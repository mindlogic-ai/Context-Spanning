"""MoshiRAG RAG suite (paper: knowledge grounding): spoken QA through the real frame-clock stack.

    python -m eval.rag.run <mode> <data_root> <out_root> [--checkpoint ckpt.pt] [--limit N] [--arm ARM]

mode      halueval | math | llama_questions | web_questions | trivia_qa
data_root halueval / math: a dir with meta.json [{id, text, answer, knowledge?}] and audios/<id>.wav
          OpenAudioBench sets: <root>/<mode>/<mode>.csv + <root>/<mode>/audios/
arm       real      the deployed backend answers (MoshiRAG Table 9 protocol; default for QA sets)
          router    HaluEval: the gold passage is handed to the ROUTER as Context DB and its answer
                    is the span (the deployed 4-element path: question + Context DB + router + prompt)
          gold      HaluEval: the gold passage itself is the span (Table 8 GT-reference protocol)
          off       retrieval off: <ret> may fire, nothing arrives
Per item: output.wav (agent, 24 kHz mono), meta.json, run_events.json (transcript + span events).
"""
import argparse
import csv
import json
import os

import numpy as np
import soundfile as sf

from eval.common import CTX, NO_INFO, RealtimeBackend, Stack, load_mono, transcript

TAIL_S = 14.0     # answer window after the question
PROMPT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


class GoldSpanBackend(RealtimeBackend):
    """HaluEval Table 8 arm: the gold reference document IS the retrieval result."""
    def __init__(self, ref):
        super().__init__(cache=False); self._ref = ref
    def retrieve(self, query, **kw):
        self.last_source = "halueval_ref"; return self._ref


class RouterCondensedBackend(RealtimeBackend):
    """HaluEval deployment-faithful arm: the gold document goes to the router as Context DB text; the
    router's own answer is the span. The raw passage never reaches the speech model."""
    def __init__(self, doc):
        super().__init__(cache=False); self._doc = doc
    def retrieve(self, query, kind="auto", ctx=None, aux_context=None, history=None, convo=None):
        db = f"[context db] {self._doc}" + (f"\n{convo}" if convo else "")
        return super().retrieve(query, kind=kind, ctx=ctx, aux_context=aux_context, history=history, convo=db)


class OffBackend(RealtimeBackend):
    def __init__(self):
        super().__init__(cache=False)
    def retrieve(self, query, **kw):
        self.last_source = "off"; return None


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
    ap.add_argument("mode"); ap.add_argument("data_root"); ap.add_argument("out_root")
    ap.add_argument("--checkpoint", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arm", default=None, choices=["real", "router", "gold", "off"])
    ap.add_argument("--lead-s", type=float, default=0.0, help="silence before the question (lets the greeting end)")
    ap.add_argument("--prompt", default=PROMPT); ap.add_argument("--voice", default=None)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--temp-text", type=float, default=0.7)
    a = ap.parse_args(argv)
    arm = a.arm or ("router" if a.mode == "halueval" else "real")
    items = list(load_items(a.mode, a.data_root))
    if a.limit:
        items = items[: a.limit]
    print(f"[rag] {a.mode}: {len(items)} items, arm={arm}", flush=True)
    st = Stack(a.checkpoint, a.temp, a.temp_text, backend=OffBackend() if arm == "off" else None, voice=a.voice)
    done = 0
    for sid, wav, question, answer, knowledge in items:
        odir = f"{a.out_root}/{sid}"
        if os.path.exists(f"{odir}/output.wav"):
            continue
        os.makedirs(odir, exist_ok=True)
        pcm = load_mono(wav, st.sr)
        pcm = np.concatenate([np.zeros(int(a.lead_s * st.sr), np.float32), pcm])
        q_end = len(pcm) / st.sr
        pcm = np.concatenate([pcm, np.zeros(int(TAIL_S * st.sr), np.float32)])
        backend = None
        if a.mode == "halueval" and arm == "gold":
            backend = GoldSpanBackend(str(knowledge))
        elif a.mode == "halueval" and arm == "router":
            backend = RouterCondensedBackend(str(knowledge))
        res = st.run(a.prompt, sid, pcm, backend=backend)
        sf.write(f"{odir}/output.wav", res["agent"], st.sr)
        json.dump({"question": question, "answer": answer, "knowledge": (str(knowledge)[:2000] if knowledge else None),
                   "question_start_s": round(a.lead_s, 2), "question_end_s": round(q_end, 2), "arm": arm},
                  open(f"{odir}/meta.json", "w"), ensure_ascii=False)
        evs = [{"q": e.get("question"), "ref": e.get("reference"), "inject": e.get("inject"), "src": e.get("src"),
                "t": e.get("t_ret"), "t_inj": e.get("t_inj"), "frames": e.get("frames")} for e in res["events"]]
        json.dump({"transcript": transcript(st.eng.spm, res["tokens"]), "events": evs},
                  open(f"{odir}/run_events.json", "w"), ensure_ascii=False)
        done += 1
        print(f"[rag] {sid} {done}/{len(items)} ret={len(evs)}", flush=True)
    print(f"[rag] done total={done}", flush=True)


if __name__ == "__main__":
    main()
