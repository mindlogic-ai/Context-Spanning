"""RAG suite (MoshiRAG protocol): spoken QA through the real frame-clock stack.

    python -m benchmark.rag.run <set> <data_root> <out_root> --checkpoint ckpt.pt [--shard K/N]

set        halueval | math | llama_questions | web_questions | trivia_qa
data_root  halueval / math: a dir with meta.json [{id, text, answer, knowledge?}] and audios/<id>.wav
           OpenAudioBench sets: <root>/<set>/<set>.csv + <root>/<set>/audios/

There is one method (benchmark/README.md). Every set is run whole, in dataset order; `--shard K/N` only
splits one set over lanes that each hold their own router. The reference that answers a `<ret>` is the
moshi-rag reference generator on the router model for every set (benchmark/rag/moshirag.py). Persona
prompt, sampling temperatures, lead silence and voice assignment are fixed here and are not arguments.
Per item: output.wav (agent, 24 kHz mono), meta.json, run_events.json (text stream + span events).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import soundfile as sf

from benchmark.rag.moshirag import RAG_TIMEOUT_S, STT_WAIT_S, MoshiRagBackend
from benchmark.stack import Stack, load_mono, transcript

TAIL_S = 14.0     # answer window after the question
TEMP, TEMP_TEXT = 0.8, 0.7
# System prompt built exactly as the training corpus builds it: persona, then the capability sentence every training
# dialogue carries, then the opening-mode sentence for user-first dialogues (every QA item starts with the user's question).
# The runtime wraps it in <system> markers.
PROMPT = ("You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
          " You are moshi, a capable voice assistant who also handles any everyday request \u2014 looking things up, booking,"
          " weather, reminders, directions \u2014 competently; the role above is your warm manner and the area you know best,"
          " not the only thing you help with."
          " The user speaks first. Listen to the whole question and answer it. Do not open with a greeting.")


def load_items(mode, root):
    if mode in ("halueval", "math"):
        for r in json.load(open(f"{root}/meta.json")):
            yield r["id"], f"{root}/audios/{r['id']}.wav", r["text"], r["answer"], r.get("knowledge")
        return
    rows = list(csv.DictReader(open(f"{root}/{mode}/{mode}.csv")))
    keys = list(rows[0].keys())
    qk = next(k for k in keys if k.lower().startswith(("question", "prompt", "text", "instruction", "query")))
    ak = next((k for k in keys if "answer" in k.lower() or "gold" in k.lower()), None)
    alk = next((k for k in keys if "alias" in k.lower()), None)
    for i, r in enumerate(rows):
        wav = f"{root}/{mode}/audios/{os.path.basename(r.get('audio_filename', f'{mode}_{i}.mp3'))}"
        if os.path.exists(wav):
            gold = r[ak] if ak else ""
            if alk and r.get(alk):
                gold = f"{gold} | aliases: {r[alk]}"      # the judge sees the full valid-answer set
            yield f"{mode}_{i}", wav, r[qk], gold, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["halueval", "math", "llama_questions", "web_questions", "trivia_qa"])
    ap.add_argument("data_root"); ap.add_argument("out_root")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", default=None, help="K/N: items K, K+N, K+2N... (lanes sharing one set)")
    a = ap.parse_args(argv)
    items = list(load_items(a.mode, a.data_root))
    if a.shard:
        k, n_sh = (int(x) for x in a.shard.split("/"))
        items = items[k::n_sh]
    print(f"[rag] {a.mode}: {len(items)} items, shard={a.shard}", flush=True)
    st = Stack(a.checkpoint, TEMP, TEMP_TEXT)
    done = 0
    for sid, wav, question, answer, knowledge in items:
        odir = f"{a.out_root}/{sid}"
        if os.path.exists(f"{odir}/output.wav"):
            continue
        os.makedirs(odir, exist_ok=True)
        pcm = load_mono(wav, st.sr)
        q_end = len(pcm) / st.sr
        pcm = np.concatenate([pcm, np.zeros(int(TAIL_S * st.sr), np.float32)])
        backend = MoshiRagBackend()          # every set, HaluEval included: the reference is generated from the question
        # moshi-rag run_inference.py: every <ret> waits stt_wait_time, then the transcript so far goes to the
        # reference LLM; the reference is applied whenever it arrives within rag_timeout (no later-than-X drop).
        t0 = time.time()
        res = st.run(PROMPT, sid, pcm, backend=backend, ret_deadline_s=RAG_TIMEOUT_S, ret_fixed_wait_s=STT_WAIT_S,
                     asr_window_s=60.0)
        # the frame clock is paced to 1.0x; an engine that cannot keep up renders slower and every span then lands
        # earlier in frames than in wall time, so the speed is recorded with the item and a slow lane is flagged
        speed = round(len(pcm) / st.sr / max(1e-6, time.time() - t0), 3)
        if speed < 0.98:
            print(f"[rag] WARNING {sid} rendered at {speed}x realtime: too many engines on this GPU", flush=True)
        sf.write(f"{odir}/output.wav", res["agent"], st.sr)
        json.dump({"set": a.mode, "question": question, "answer": answer,
                   "knowledge": (str(knowledge)[:2000] if knowledge else None),
                   "question_start_s": 0.0, "question_end_s": round(q_end, 2), "reference": "moshirag_llm",
                   "render_speed_x": speed},
                  open(f"{odir}/meta.json", "w"), ensure_ascii=False)
        evs = [{"q": e.get("question"), "ref": e.get("reference"), "inject": e.get("inject"), "src": e.get("src"),
                "t": e.get("t_ret"), "t_inj": e.get("t_inj"), "frames": e.get("frames")} for e in res["events"]]
        if evs:
            evs[-1]["backend_status"] = backend.last_status     # ok | timeout | error | empty
            evs[-1]["backend_s"] = backend.last_elapsed_s
        json.dump({"transcript": transcript(st.eng.spm, res["tokens"]), "events": evs},
                  open(f"{odir}/run_events.json", "w"), ensure_ascii=False)
        done += 1
        print(f"[rag] {sid} {done}/{len(items)} ret={len(evs)}", flush=True)
    print(f"[rag] done total={done}", flush=True)


if __name__ == "__main__":
    main()
