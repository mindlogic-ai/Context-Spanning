#!/usr/bin/env python3
"""Adds the official Full-Duplex-Bench v3 ASR fields to our result_<provider>.json files and runs the official latency
analysis, exactly as v3/run_tool_benchmark.py and v3/analyze_tool_latency.py do for the official providers:

  nvidia/parakeet-tdt-0.6b-v2 word timestamps on the mono input -> input_transcript, input_asr_chunks,
  user_speech_end_rel (end of the word before the first gap > 2 s, else the last word);
  the same ASR on output_<provider>.wav -> transcript, asr_chunks, audio_agent_speech_start, perceived_total_latency;
  then analyze_tool_latency.py (first-response / tool-call / task-completion latency, filler sentences via gpt-4o).

Run this BEFORE evaluate_tool_calls.py: the official judge reads `transcript` (the ASR text) and the turn-taken filter
reads `asr_chunks`. Needs NeMo and OPENAI_API_KEY.

Usage: python benchmark/fdb/v3_artefacts.py --data-root DIR --provider ours --fdb-v3 PATH_TO_BENCHMARK [--device cuda]
"""
import argparse, glob, json, os, subprocess, sys, tempfile

ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def run_asr(model, wav_path):
    import soundfile as sf, librosa
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, x, 16000)
        out = model.transcribe([tmp.name], timestamps=True)
    os.unlink(tmp.name)
    chunks = []
    text = ""
    if out:
        res = out[0]
        if hasattr(res, "timestamp") and "word" in res.timestamp:
            for w in res.timestamp["word"]:
                text += w["word"] + " "
                chunks.append({"text": w["word"], "timestamp": [w["start"], w["end"]]})
        elif hasattr(res, "text"):
            text = res.text
    return {"text": text.strip(), "chunks": chunks}


def user_speech_end(chunks):
    if not chunks:
        return 0
    for cur, nxt in zip(chunks, chunks[1:]):
        if nxt["timestamp"][0] - cur["timestamp"][1] > 2.0:
            return cur["timestamp"][1]
    return chunks[-1]["timestamp"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--provider", default="ours")
    ap.add_argument("--fdb-v3", required=True, help="checkout of the Full-Duplex-Bench v3 code")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    results = sorted(glob.glob(f"{a.data_root}/**/result_{a.provider}.json", recursive=True))
    todo = [r for r in results if os.path.exists(os.path.join(os.path.dirname(r), f"output_{a.provider}.wav"))]
    print(f"[v3art] {len(results)} results, {len(todo)} with agent audio", flush=True)
    if not todo:
        sys.exit(1)
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    if a.device == "cuda":
        model = model.cuda()
    for r in todo:
        d = os.path.dirname(r)
        data = json.load(open(r))
        if "asr_chunks" in data and "user_speech_end_rel" in data:
            continue
        inp = run_asr(model, os.path.join(d, "input.wav"))
        out = run_asr(model, os.path.join(d, f"output_{a.provider}.wav"))
        data["input_transcript"] = inp["text"]
        data["input_asr_chunks"] = inp["chunks"]
        data["user_speech_end_rel"] = user_speech_end(inp["chunks"])
        data["transcript"] = out["text"]
        data["asr_chunks"] = out["chunks"]
        if out["chunks"] and data["user_speech_end_rel"]:
            start = out["chunks"][0]["timestamp"][0]
            data["audio_agent_speech_start"] = start
            data["perceived_total_latency"] = round(start - data["user_speech_end_rel"], 3)
        json.dump(data, open(r, "w"), ensure_ascii=False, indent=2)
        print(f"[v3art] {os.path.basename(d)} user_end={data['user_speech_end_rel']:.2f} agent_words={len(out['chunks'])}", flush=True)

    report = os.path.abspath(f"{a.data_root}/{a.provider}_latency_report.json")
    subprocess.run([sys.executable, "analyze_tool_latency.py", "--results-dir", os.path.abspath(a.data_root),
                    "--provider", a.provider, "--output", report], cwd=a.fdb_v3, check=True)


if __name__ == "__main__":
    main()
