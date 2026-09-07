"""Frame-clock streaming loop: user audio in, agent audio out, spans injected as they arrive.

The `<ret>` handling is the DuetaSpan live runtime's (duetaspan/runtime/live_server.py): the
recent user audio is transcribed; the backend is asked with the user's profile, what the agent has
already said, the requests already answered and the Context DB working text; a tool result or a
direct answer is injected as a Context Span; a `<ret>` with no transcript, or a router "nothing to
do" that no tool produced, is closed with an EMPTY marker pair (the surface the model was trained
on for "issued, nothing arrived"). The frame clock never waits for any of it.
"""
import json
import threading
import time

import numpy as np

from .duetaspan.runtime.backend.context_db import ContextDB, ContextProfile


def retrieve_for_ret(backend, asr, clip, sample_rate, ctx, db, said_text, events, on_question=None):
    """One `<ret>`: transcribe -> ask the backend -> decide what is injected.

    Returns dict(question, reference, inject, src, args): `inject` is the text to put in the span
    ("" = empty marker pair), `reference` the backend's raw answer (None when nothing was asked)."""
    question = asr.transcribe(clip, sample_rate) or ""
    if on_question is not None:
        on_question(question)
    if not question.strip():
        return {"question": "", "reference": None, "inject": "", "src": None, "args": None}
    if db is not None and question != db.last_user_text():
        db.add_user_turn(question)
    answered = [f"{e['question']!r} -> {e['src'] or 'no tool'}" for e in events if e.get("reference") is not None]
    convo = db.working_text() if db is not None else None
    ref = backend.retrieve(question or said_text, ctx=ctx, aux_context=said_text, history=answered,
                           convo=convo)
    src, args = backend.last_source, getattr(backend, "last_args", None)
    if ref and "no information found" in str(ref).lower() and not (src or "").startswith("mcp:"):
        inject = ""                       # router had nothing to do: no tool ran, nothing to ground on
    else:
        inject = ref or ""
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
            ready["frames"] = eng.inject_context_span(ready["inject"])
            events.append(ready)
            if verbose:
                print(f"[span @{ready['t_inj']:.1f}s] q={ready['question']!r} src={ready['src']} "
                      f"-> {ready['inject'][:100]!r}", flush=True)
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
