"""RAG suite (MoshiRAG protocol): spoken QA through the real frame-clock stack.

    python -m benchmark.rag.run <set> <data_root> <out_root> --checkpoint ckpt.pt [--shard K/N]
                                [--protocol live|api] [--reference-delay-s 0.8]

set        halueval | math | llama_questions | web_questions | trivia_qa
data_root  halueval / math: a dir with meta.json [{id, text, answer, knowledge?}] and audios/<id>.wav
           OpenAudioBench sets: <root>/<set>/<set>.csv + <root>/<set>/audios/

Two protocols, both stated in benchmark/README.md section 2. Every set is run whole, in dataset order;
`--shard K/N` only splits one set over engines (lanes claim items, so several engines may share a shard list).

live   the reference LLM is a local server (the router model) and its latency is streamed: the frame clock
       runs at 1.0x while the call is in flight, as moshi-rag `run_inference.py` with a local model.
api    the reference LLM is an API model (default gpt-4.1): its reference is generated before the stream from
       the ASR transcript of the question, and it is injected exactly `--reference-delay-s` after the `<ret>`,
       replayed in frames (moshi-rag's API-backend protocol, arXiv 2604.12928 fn. 9: a uniform retrieval delay
       and no timeout). The stream is not paced to wall time, nothing is dropped, and a `<ret>` with no
       transcript is rendered again (up to three rounds) so that every `<ret>` reaches the backend.

Persona prompt, sampling temperatures, lead silence and voice assignment are fixed here and are not arguments.
Per item: output.wav (agent, 24 kHz mono), meta.json, run_events.json (text stream + span events); the api
protocol also writes references_<K>.json (item id -> reference, transcript, API seconds) next to the items.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import threading
import time

import numpy as np
import soundfile as sf

from benchmark.rag.moshirag import RAG_TIMEOUT_S, STT_WAIT_S, MoshiRagBackend
from benchmark.stack import CTX, Stack, load_mono, transcript

TAIL_S = 14.0     # answer window after the question
TEMP, TEMP_TEXT = 0.8, 0.7
# System prompt built exactly as the training corpus builds it: persona, then the capability sentence every training
# dialogue carries, then the opening-mode sentence for user-first dialogues (every QA item starts with the user's question).
# The runtime wraps it in <system> markers.
PROMPT = ("You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
          " You are moshi, a capable voice assistant who also handles any everyday request — looking things up, booking,"
          " weather, reminders, directions — competently; the role above is your warm manner and the area you know best,"
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


class ReferenceStore:
    """references_<K>.json of one shard; every shard's file under out_root is read, so lanes share the work."""

    def __init__(self, out_root, shard_k):
        self.path, self.lock, self.mine, self.d = f"{out_root}/references_{shard_k}.json", threading.Lock(), {}, {}
        for f in sorted(glob.glob(f"{out_root}/references_*.json")):
            try:
                self.d.update(json.load(open(f)))
            except Exception:
                pass
        if os.path.exists(self.path):
            self.mine = json.load(open(self.path))

    def get(self, sid):
        return self.d.get(sid)

    def put(self, sid, entry):
        with self.lock:
            self.d[sid] = entry; self.mine[sid] = entry
            json.dump(self.mine, open(self.path + ".tmp", "w"), ensure_ascii=False)
            os.replace(self.path + ".tmp", self.path)


class StoredReference:
    """The api protocol's backend for one item: hands back the stored reference; the frame at which it is
    injected is the stream's (`inject_at_f`). Exposes what the driver records."""

    def __init__(self, store, sid):
        self.store, self.sid = store, sid
        self.last_source = self.last_args = self.last_status = self.last_elapsed_s = None
        self.last_trace = []

    def retrieve(self, query, **kw):
        hit = self.store.get(self.sid) or {}
        out = hit.get("out")
        self.last_source = "moshirag-api" if out else None
        self.last_status = "ok" if out else "empty"
        self.last_elapsed_s = hit.get("api_s")
        return out


def fetch_references(st, items, store, out_root):
    """api protocol, step 1: one reference per item from the ASR transcript of its question audio, retried
    until the API answers (a timeout is retried, never injected as nothing)."""
    n, t_api, empty = 0, [], 0
    for sid, wav, question, answer, knowledge in items:
        if os.path.exists(f"{out_root}/{sid}/output.wav") or store.get(sid) is not None:
            continue
        pcm = load_mono(wav, st.sr)
        q = st.asr.transcribe(pcm, st.sr) or question
        out, t0 = None, time.time()
        for _ in range(3):
            out = MoshiRagBackend(api=True).retrieve(q, ctx=CTX, aux_context="", history=[])
            if out:
                break
        store.put(sid, {"out": out, "q": q, "api_s": round(time.time() - t0, 2)})
        n += 1; t_api.append(time.time() - t0); empty += (not out)
    if t_api:
        print(f"[rag] {n} references fetched: api p50 {np.median(t_api):.2f}s p90 {np.percentile(t_api, 90):.2f}s,"
              f" empty {empty}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["halueval", "math", "llama_questions", "web_questions", "trivia_qa"])
    ap.add_argument("data_root"); ap.add_argument("out_root")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", default=None, help="K/N: items K, K+N, K+2N... (lanes sharing one set)")
    ap.add_argument("--protocol", choices=["live", "api"], default="live")
    ap.add_argument("--reference-delay-s", type=float, default=0.8,
                    help="api: seconds from <ret> to the span, replayed in 80 ms frames (default 0.8 = 10 frames)")
    a = ap.parse_args(argv)
    items = list(load_items(a.mode, a.data_root))
    k, n_sh = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    items = items[k::n_sh]
    print(f"[rag] {a.mode}: {len(items)} items, shard={a.shard}, protocol={a.protocol}", flush=True)
    st = Stack(a.checkpoint, TEMP, TEMP_TEXT)
    os.makedirs(a.out_root, exist_ok=True)
    store = None
    if a.protocol == "api":
        hold_f = int(round(a.reference_delay_s * st.eng.frame_rate))
        for _ in range(2):        # warm the CUDA path: cold steps stall the ASR beside the engine past its timeout
            st.run(PROMPT, "warmup", np.zeros(int(6 * st.sr), np.float32), backend=StoredReference({}, "warmup"),
                   realtime=False)
        store = ReferenceStore(a.out_root, k)
        fetch_references(st, items, store, a.out_root)

    def render(items):
        done = 0
        for sid, wav, question, answer, knowledge in items:
            odir = f"{a.out_root}/{sid}"
            if os.path.exists(f"{odir}/output.wav"):
                continue
            os.makedirs(odir, exist_ok=True)
            claim = f"{odir}/.claim"           # lanes sharing a shard list: first claimant renders; a claim older than 30 min is stale
            try:
                if os.path.exists(claim) and time.time() - os.path.getmtime(claim) > 1800:
                    os.remove(claim)
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            except FileExistsError:
                continue
            pcm = load_mono(wav, st.sr)
            q_end = len(pcm) / st.sr
            pcm = np.concatenate([pcm, np.zeros(int(TAIL_S * st.sr), np.float32)])
            t0 = time.time()
            if a.protocol == "api":
                backend = StoredReference(store, sid)
                # the reference is already known; the <ret> closes the question utterance, no partial transcripts,
                # the span lands hold_f frames after the <ret> and nothing is dropped
                res = st.run(PROMPT, sid, pcm, backend=backend, realtime=False, inject_at_f=hold_f, no_partials=True,
                             ret_closes_utt=True, ret_deadline_s=None)
                meta = {"reference": "moshirag_api", "reference_model": MoshiRagBackend(api=True).model,
                        "reference_delay_s": a.reference_delay_s, "reference_delay_f": hold_f}
            else:
                backend = MoshiRagBackend()      # every set, HaluEval included: the reference is generated from the question
                # moshi-rag run_inference.py: every <ret> waits stt_wait_time, then the transcript so far goes to the
                # reference LLM; the reference is applied whenever it arrives within rag_timeout (no later-than-X drop).
                res = st.run(PROMPT, sid, pcm, backend=backend, ret_deadline_s=RAG_TIMEOUT_S, ret_fixed_wait_s=STT_WAIT_S,
                             asr_window_s=60.0)
                meta = {"reference": "moshirag_llm"}
            # the live frame clock is paced to 1.0x; an engine that cannot keep up renders slower and every span then
            # lands earlier in frames than in wall time, so the speed is recorded with the item and a slow lane is flagged
            speed = round(len(pcm) / st.sr / max(1e-6, time.time() - t0), 3)
            if a.protocol == "live" and speed < 0.98:
                print(f"[rag] WARNING {sid} rendered at {speed}x realtime: too many engines on this GPU", flush=True)
            sf.write(f"{odir}/output.wav", res["agent"], st.sr)
            json.dump({"set": a.mode, "question": question, "answer": answer,
                       "knowledge": (str(knowledge)[:2000] if knowledge else None),
                       "question_start_s": 0.0, "question_end_s": round(q_end, 2), "protocol": a.protocol,
                       "render_speed_x": speed, **meta},
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
        return done

    done = render(items)
    if a.protocol == "api":
        # a <ret> whose transcript came back empty never reached the backend (the ASR beside a cold engine can time
        # out): those items are rendered again, up to three rounds, so that every <ret> of the set is answered
        for rnd in range(3):
            bad = []
            for sid, *_ in items:
                f = f"{a.out_root}/{sid}/run_events.json"
                if not os.path.exists(f):
                    continue
                evs = json.load(open(f)).get("events", [])
                if evs and evs[-1].get("backend_status") is None:
                    bad.append(sid)
            if not bad:
                break
            for sid in bad:
                shutil.rmtree(f"{a.out_root}/{sid}", ignore_errors=True)
            print(f"[rag] round {rnd + 1}: re-rendering {len(bad)} items whose <ret> reached no backend: {bad[:8]}", flush=True)
            done += render([it for it in items if it[0] in set(bad)])
    print(f"[rag] done total={done}", flush=True)


if __name__ == "__main__":
    main()
