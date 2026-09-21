"""Build the spoken math benchmark: <out>/meta.json and <out>/audios/<id>.wav from <out>/questions/.

    python benchmark/math/prepare_questions.py <out>
    python benchmark/math/build_audio.py <out>                  # all 3,822 items (one GPU)
    python benchmark/math/build_audio.py <out> --limit 100      # the first 100 items only, a quick build
    python benchmark/math/build_audio.py <out> --dry-run        # meta + voice assignment, no GPU, nothing written

The result is what `python -m benchmark.rag.run math <out> ...` reads.

meta.json    one row per question, sets in the order AddSub, MultiArith, SingleEq, SVAMP, GSM8K and items in
             source order: {"id", "dataset", "text": question, "answer": str, "knowledge": null}.
speech       Kyutai TTS (`pip install moshi`; model kyutai/tts-1.6b-en_fr, n_q 32, temperature 0.6, half
             precision, cfg_coef 2.0). Voices: the raw recordings of `voice-donations/` in the public HF repo
             kyutai/tts-voices (the `*_enhanced` copies are excluded), sorted by name. Per item,
             rng = random.Random(first 8 hex digits of md5(id)); agent = rng.choice(pool); user =
             rng.choice(pool without agent); the question is spoken with the user voice. The text is
             normalised first: *asterisk* wrapping stripped, double quotes removed, em dash -> ", ",
             curly apostrophe -> "'", repeated spaces collapsed. Items are generated in batches sorted by
             word count (longest first); torch is seeded from the ids of each batch, so a batch of one
             (`--batch-size 1`) is seeded per item. Sampling on GPU is not bit-exact across hardware and
             batch shapes, so a rebuild matches the released audio in content and voice, not sample by sample.
audios       24 kHz stereo PCM_16: left = 0.35 s of zeros then the speech (mono), right = Gaussian noise of
             std 7/32768 over the whole length, drawn from numpy default_rng(20260916) item after item in
             meta.json order (the HaluEvalAudio layout). The noise of an item depends on the lengths of the
             items before it, so `--sets` builds its own noise sequence, while `--limit N` is a prefix of the
             full build.

The build resumes: items whose audios/<id>.wav exists are not synthesised again, and the speech of an
interrupted run is kept in <out>/speech/ (safe to delete once audios/ is complete).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np
import soundfile as sf

SETS = ("AddSub", "MultiArith", "SingleEq", "SVAMP", "GSM8K")
VOICE_REPO = "kyutai/tts-voices"
VOICE_SUFFIX = ".wav.1e68beda@240.safetensors"   # voice embeddings of kyutai/tts-1.6b-en_fr
SR = 24000
LEAD_S = 0.35
NOISE_STD = 7 / 32768
NOISE_SEED = 20260916
CFG_COEF = 2.0
DECODE_PRIME = 8                                    # warm-up decodes of the first frame, output discarded


def load_meta(out, sets=SETS, limit=0):
    """meta.json rows from <out>/questions/<set>.jsonl, in benchmark order."""
    rows = []
    for name in SETS:
        if name not in sets:
            continue
        path = os.path.join(out, "questions", f"{name}.jsonl")
        if not os.path.exists(path):
            sys.exit(f"missing {path}: run benchmark/math/prepare_questions.py {out} first")
        for line in open(path):
            if line.strip():
                q = json.loads(line)
                rows.append({"id": q["id"], "dataset": q["dataset"], "text": q["question"],
                             "answer": str(q["answer"]), "knowledge": None})
    return rows[:limit] if limit else rows


def normalize(text):
    """TTS input: the Kyutai TTS garbles *asterisk* titles and stutters on quotes."""
    t = (text or "").strip()
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = t.replace("“", "").replace("”", "").replace('"', "")
    t = t.replace("—", ", ").replace("’", "'")
    return re.sub(r"  +", " ", t).strip()


def voice_pool(voices_dir=None):
    """Sorted repo-relative names of the raw voice-donations embeddings."""
    if voices_dir:
        names = [os.path.relpath(p, voices_dir) for p in glob.glob(os.path.join(voices_dir, "voice-donations", "*" + VOICE_SUFFIX))]
    else:
        from huggingface_hub import list_repo_files
        names = [f for f in list_repo_files(VOICE_REPO) if f.startswith("voice-donations/") and f.endswith(VOICE_SUFFIX)]
    names = sorted(n.replace(os.sep, "/") for n in names if not n.endswith("_enhanced" + VOICE_SUFFIX))
    if len(names) < 2:
        sys.exit(f"voice pool has {len(names)} voices (voices dir: {voices_dir or VOICE_REPO})")
    return names


def assign_voices(item_id, pool):
    """(agent voice, user voice) for an item; the question is spoken with the user voice."""
    rng = random.Random(int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16))
    va = rng.choice(pool)
    vu = rng.choice([p for p in pool if p != va])
    return va, vu


def _seed(ids):
    return int(hashlib.md5("|".join(ids).encode()).hexdigest()[:8], 16)


def synthesize(jobs, speech_dir, voices_dir, batch_size):
    """jobs: [(id, text, user voice)]; writes speech_dir/<id>.wav (24 kHz mono)."""
    import sphn
    import torch
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

    info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device="cuda", dtype=torch.half)
    mimi = tts.mimi
    attrs = {}

    def attr(voice):
        if voice not in attrs:
            if voices_dir:
                path = os.path.join(voices_dir, voice)
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(VOICE_REPO, voice)
            from pathlib import Path
            attrs[voice] = tts.make_condition_attributes([Path(path)], cfg_coef=CFG_COEF)
        return attrs[voice]

    jobs = sorted(jobs, key=lambda j: len(j[1].split()), reverse=True)   # stable: ties keep meta order
    os.makedirs(speech_dir, exist_ok=True)
    t_all, done = time.time(), 0
    for b in range(0, len(jobs), batch_size):
        batch = jobs[b:b + batch_size]
        torch.manual_seed(_seed([j[0] for j in batch]))
        entries = [tts.prepare_script([text], padding_between=1) for _, text, _ in batch]
        res = tts.generate(entries, [attr(v) for _, _, v in batch])
        frames = res.frames[tts.delay_steps:]
        with torch.no_grad(), mimi.streaming(len(batch)):
            if frames:
                for _ in range(DECODE_PRIME):
                    mimi.decode(frames[0][:, 1:])
            wavs = torch.cat([mimi.decode(f[:, 1:]) for f in frames], dim=-1)
        for k, (item_id, _, _) in enumerate(batch):
            es = res.end_steps[k]
            n = int(mimi.sample_rate * ((es if es is not None else wavs.shape[-1]) + tts.final_padding) / mimi.frame_rate)
            tmp = os.path.join(speech_dir, f"{item_id}.tmp.wav")
            sphn.write_wav(tmp, wavs[k, :, :n].clamp(-1, 1).cpu().numpy(), mimi.sample_rate)
            os.replace(tmp, os.path.join(speech_dir, f"{item_id}.wav"))
        done += len(batch)
        print(f"[tts] {done}/{len(jobs)}  {time.time() - t_all:.0f} s", flush=True)


def assemble(meta, out):
    """audios/<id>.wav for every item in meta order; the noise stream advances over existing files too."""
    rng = np.random.default_rng(NOISE_SEED)
    lead = np.zeros(int(LEAD_S * SR), dtype=np.float32)
    os.makedirs(os.path.join(out, "audios"), exist_ok=True)
    written = 0
    for r in meta:
        dst = os.path.join(out, "audios", f"{r['id']}.wav")
        if os.path.exists(dst):
            rng.normal(0.0, NOISE_STD, sf.info(dst).frames)
            continue
        speech, sr = sf.read(os.path.join(out, "speech", f"{r['id']}.wav"), dtype="float32")
        if sr != SR:
            sys.exit(f"{r['id']}: speech at {sr} Hz, expected {SR}")
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        left = np.concatenate([lead, speech])
        noise = rng.normal(0.0, NOISE_STD, len(left)).astype(np.float32)   # float32 before PCM_16, as released
        tmp = dst + ".part"
        sf.write(tmp, np.stack([left, noise], axis=1), SR, subtype="PCM_16", format="WAV")
        os.replace(tmp, dst)
        written += 1
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="benchmark directory holding questions/ (from prepare_questions.py)")
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated subset of the five sets")
    ap.add_argument("--limit", type=int, default=0, help="first N items in meta order (0 = all)")
    ap.add_argument("--batch-size", type=int, default=64, help="TTS batch size (lower it on small GPUs)")
    ap.add_argument("--voices-dir", default=None, help=f"local copy of {VOICE_REPO} (default: the HF Hub)")
    ap.add_argument("--dry-run", action="store_true", help="print meta counts and voice assignment; no GPU, no writes")
    ap.add_argument("--show", type=int, default=5, help="items printed by --dry-run")
    a = ap.parse_args(argv)
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    if any(s not in SETS for s in sets):
        ap.error(f"--sets must be among {', '.join(SETS)}")

    meta = load_meta(a.out, sets, a.limit)
    pool = voice_pool(a.voices_dir)
    counts = {s: sum(r["dataset"] == s for r in meta) for s in SETS if s in sets}
    print(f"{len(meta)} items {counts}; voice pool {len(pool)}")
    if a.dry_run:
        for r in meta[:a.show]:
            va, vu = assign_voices(r["id"], pool)
            print(json.dumps({"id": r["id"], "answer": r["answer"], "user_voice_file": vu,
                              "agent_voice_file": va, "tts_text": normalize(r["text"])}, ensure_ascii=False))
        return 0

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    speech_dir = os.path.join(a.out, "speech")
    jobs = []
    for r in meta:
        if os.path.exists(os.path.join(a.out, "audios", f"{r['id']}.wav")) or os.path.exists(os.path.join(speech_dir, f"{r['id']}.wav")):
            continue
        jobs.append((r["id"], normalize(r["text"]), assign_voices(r["id"], pool)[1]))
    print(f"to synthesise: {len(jobs)}")
    if jobs:
        synthesize(jobs, speech_dir, a.voices_dir, a.batch_size)
    print(f"audios written: {assemble(meta, a.out)}; {len(meta)} items in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
