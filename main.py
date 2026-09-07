"""Context Spanning — single entry point.

  python main.py infer   --input-wav q.wav --output-wav out.wav [--voice f0]
  python main.py serve   [--voice f0] [--host 0.0.0.0 --port 8080 --token ...]
  python main.py prepare --in-dir data/raw --out-dir data/prepared
  python main.py train   --data-dir data/prepared --out-dir runs/ft [--checkpoint ckpt.pt]

Backends are the DuetaSpan runtime's (contextspan/duetaspan): the tool router + 125-tool bank
(MCP_ROUTER_LLM_URL / MCP_ROUTER_LLM_MODEL / MCP_ROUTER_LLM_API=openai), the RAG LLM
(MOSHICP_RAG_LLM_URL / MOSHICP_RAG_LLM_MODEL) and the ASR endpoint (MOSHICP_ASR_URL, the
`python -m contextspan.duetaspan.runtime.asr_server` protocol). See README.
"""
import argparse
import json
import sys

import numpy as np
import soundfile as sf


def transcript(spm, toks):
    from contextspan.spans import RET_TOKEN_ID, SPAN_TOKEN_ID, TEXT_PAD
    out, run = [], []
    for t in toks:
        if t is None or t <= 2 or t == TEXT_PAD:
            continue
        if t in (RET_TOKEN_ID, SPAN_TOKEN_ID):
            if run:
                out.append(spm.decode(run)); run = []
            out.append("<ret>" if t == RET_TOKEN_ID else "<span>")
        else:
            run.append(int(t))
    if run:
        out.append(spm.decode(run))
    return " ".join(out)


def _backend_and_asr():
    """The DuetaSpan runtime backend (MCP router over the tool bank, then LLM-RAG) and its ASR client,
    prewarmed the way the DuetaSpan offline chat does: the first `<ret>` otherwise pays the MCP session
    set-up, the tool discovery and the router's first call (measured 3.5 s -> 0.2-0.5 s warm)."""
    import time
    from contextspan.duetaspan.align.asr import ASR
    from contextspan.duetaspan.runtime.backend.realtime import RealtimeBackend
    backend = RealtimeBackend(cache=False)
    t0 = time.time()
    for q in ("hello", "what time is it now?", "how is the weather today?"):
        try:
            backend.retrieve(q, ctx={}, aux_context="", history=[])
        except Exception as e:                                   # a cold backend is not fatal
            print(f"[warm] {type(e).__name__}: {e}", flush=True)
    print(f"[warm] backend {time.time() - t0:.1f}s", flush=True)
    asr = ASR()
    try:
        asr.transcribe(np.zeros(12000, dtype=np.float32), 24000)
    except Exception:
        pass
    return backend, asr


def _ctx(a):
    return {k: v for k, v in {"name": a.user_name, "city": a.user_city, "timezone": a.user_tz}.items() if v}


def infer(a):
    from contextspan.model import Engine, load_voice
    from contextspan.stream import run_stream
    eng = Engine(a.checkpoint, temp=a.temp, temp_text=a.temp_text, cpu_offload=a.cpu_offload)
    eng.set_persona(a.text_prompt, load_voice(a.voice))
    pcm, sr = sf.read(a.input_wav, dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm[:, 0]
    want = int(eng.mimi.sample_rate)
    if sr != want:
        import librosa
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=want)
    pcm = np.concatenate([np.zeros(int(a.lead_silence * want), np.float32), pcm,
                          np.zeros(int(a.tail_silence * want), np.float32)])
    backend, asr = _backend_and_asr()
    res = run_stream(eng, backend, asr, pcm, ctx=_ctx(a), asr_window_s=a.listen_s)
    n = min(len(pcm), len(res["agent"]))
    sf.write(a.output_wav, np.stack([pcm[:n], res["agent"][:n]], 1), want)      # L=user, R=agent
    text = transcript(eng.spm, res["tokens"])
    print(text)
    if a.output_text:
        json.dump({"transcript": text, "events": res["events"]}, open(a.output_text, "w"), indent=1)


def serve_cmd(a):
    from contextspan.model import Engine, load_voice
    from contextspan.serve import serve
    eng = Engine(a.checkpoint, temp=a.temp, temp_text=a.temp_text)
    backend, asr = _backend_and_asr()
    serve(eng, backend, asr, a.text_prompt, load_voice(a.voice),
          host=a.host, port=a.port, token=a.token, listen_s=a.listen_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="stream a user wav through the model")
    p.add_argument("--input-wav", required=True)
    p.add_argument("--output-wav", default="output.wav")
    p.add_argument("--output-text", default=None)
    p.add_argument("--checkpoint", default=None, help="local .pt; default downloads the released weights")
    p.add_argument("--voice", default="f0", help="voice: f0 / f1 / f2 from the weights repo, or a local .pt of agent-voice Mimi codes")
    p.add_argument("--text-prompt", default="You are a helpful and friendly voice assistant.")
    p.add_argument("--lead-silence", type=float, default=4.0)
    p.add_argument("--tail-silence", type=float, default=8.0)
    p.add_argument("--listen-s", type=float, default=12.0, help="seconds of user audio transcribed on <ret>")
    p.add_argument("--user-name", default="", help="user profile for the Context DB")
    p.add_argument("--user-city", default="")
    p.add_argument("--user-tz", default="", help="IANA timezone, e.g. Asia/Seoul")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--temp-text", type=float, default=0.7)
    p.add_argument("--cpu-offload", action="store_true")
    p.set_defaults(fn=infer)

    p = sub.add_parser("serve", help="live conversation in the browser")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--voice", default="f0")
    p.add_argument("--text-prompt", default="You are a helpful and friendly voice assistant.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--token", default="", help="if set, /ws requires ?token=<this>")
    p.add_argument("--listen-s", type=float, default=12.0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--temp-text", type=float, default=0.7)
    p.set_defaults(fn=serve_cmd)

    p = sub.add_parser("prepare", help="encode dialogues (json + stereo wav) into training tensors")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=lambda a: __import__("contextspan.train", fromlist=["prepare"]).prepare(a.in_dir, a.out_dir))

    p = sub.add_parser("train", help="fine-tune on prepared tensors")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="start from a local .pt (default: PersonaPlex base)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--context", type=int, default=3000)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--w-ret", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=lambda a: __import__("contextspan.train", fromlist=["train"]).train(
        a.data_dir, a.out_dir, a.checkpoint, a.steps, a.lr, a.accum, a.context, a.ckpt_every, a.w_ret, a.seed))

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
