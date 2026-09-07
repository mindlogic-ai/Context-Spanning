"""Frame-clock streaming loop: user audio in, agent audio out, spans injected as they arrive.

The `<ret>` handling is the DuetaSpan live runtime's (duetaspan/runtime/live_server.py): the
recent user audio is transcribed; the backend is asked with the user's profile, what the agent has
already said, the requests already answered and the Context DB working text; a tool result or a
direct answer is injected as a Context Span; a `<ret>` with no transcript, or a router "nothing to
do" that no tool produced, injects NOTHING (the benchmark harness's gate: an abstain span exists in
training only when a real tool ran and came back empty). The frame clock never waits for any of it.
"""
import json
import os
import threading
import time

import numpy as np

from .duetaspan.runtime.backend.context_db import ContextDB, ContextProfile

# Wall-clock budget from <ret> to injection. The training corpus's ret->span delays (deploy_ret_delay.json,
# live harness 2026-09-04) have p50 0.72 s, p99 2.3 s, max 2.4 s: a span later than that is outside what the
# model was trained to wait for, and by then it has usually answered without it (ContextSpanning #12).
RET_DEADLINE_S = float(os.environ.get("CS_RET_DEADLINE_S", "2.5"))


def retrieve_for_ret(backend, asr, clip, sample_rate, ctx, db, said_text, events, on_question=None):
    """One `<ret>`: transcribe -> ask the backend -> decide what is injected.

    Returns dict(question, reference, inject, src, args): `inject` is the span text, or None when
    nothing is injected; `reference` is the backend's raw answer (None when nothing was asked)."""
    question = asr.transcribe(clip, sample_rate) or ""
    if on_question is not None:
        on_question(question)
    if not question.strip():
        return {"question": "", "reference": None, "inject": None, "src": None, "args": None}
    if db is not None and question != db.last_user_text():
        db.add_user_turn(question)
    answered = [f"{e['question']!r} -> {e['src'] or 'no tool'}" for e in events if e.get("reference") is not None]
    convo = db.working_text() if db is not None else None
    ref = backend.retrieve(question or said_text, ctx=ctx, aux_context=said_text, history=answered,
                           convo=convo)
    src, args = backend.last_source, getattr(backend, "last_args", None)
    if ref and "no information found" in str(ref).lower() and not (src or "").startswith("mcp:"):
        inject = None                     # router had nothing to do: no tool ran, nothing to ground on
    else:
        inject = ref or None
        if db is not None and (src or "").startswith("mcp:"):
            db.add_tool_turn(f"{src[4:]}({json.dumps(args or {}, ensure_ascii=False)}) => {str(ref or '')[:240]}",
                             tool=src[4:], args=args)
    return {"question": question, "reference": ref, "inject": inject, "src": src, "args": args}


def run_stream(eng, backend, asr, pcm, ctx=None, asr_window_s=12.0, realtime=True, verbose=True):
    """Streams `pcm` (float32 mono at Mimi's rate) through the engine, one 80 ms frame per step.
    `ctx` = user profile (name, city, timezone, lat, lon, notes). Returns dict(agent, tokens, events)."""
    fs, sr = eng.frame_size, int(eng.mimi.sample_rate)
    n = len(pcm) // fs
    ctx = dict(ctx or {})
    db = ContextDB(ContextProfile(**ctx))
    lock, queue, events, tokens, out = threading.Lock(), [], [], [], []
    pending = [False]

    def kick(i):
        lo = max(0, (i + 1) * fs - int(asr_window_s * sr))
        said = eng.decode_text([t for t in tokens[-64:] if t is not None])
        r = retrieve_for_ret(backend, asr, pcm[lo:(i + 1) * fs], sr, ctx, db, said, events)
        r["t_ret"] = i / eng.frame_rate
        with lock:
            queue.append(r)
        pending[0] = False

    t_start = time.time()
    for i in range(n):
        with lock:
            ready = queue.pop(0) if queue else None
        if ready is not None:
            ready["t_inj"] = i / eng.frame_rate
            ready["late"] = ready["inject"] is not None and (ready["t_inj"] - ready["t_ret"]) > RET_DEADLINE_S
            if ready["late"]:
                ready["inject"] = None
            ready["frames"] = eng.inject_context_span(ready["inject"]) if ready["inject"] else 0
            events.append(ready)
            if verbose:
                print(f"[span @{ready['t_inj']:.1f}s] q={ready['question']!r} src={ready['src']} "
                      f"-> {('(late, dropped)' if ready['late'] else '(no span)') if ready['inject'] is None else ready['inject'][:100]!r}", flush=True)
        o = eng.step(pcm[i * fs:(i + 1) * fs])
        out.append(o["agent_pcm"] if o["agent_pcm"] is not None else np.zeros(fs, np.float32))
        tokens.append(o["text_token"])
        if o["is_ret"] and not pending[0]:
            pending[0] = True
            if verbose:
                print(f"[ret @{i / eng.frame_rate:.1f}s]", flush=True)
            threading.Thread(target=kick, args=(i,), daemon=True).start()
        if realtime:
            dt = t_start + (i + 1) / eng.frame_rate - time.time()
            if dt > 0:
                time.sleep(dt)
    return {"agent": np.concatenate(out), "tokens": tokens, "events": events}
