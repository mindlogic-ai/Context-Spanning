"""Frame-clock streaming loop: user audio in, agent audio out, spans injected as they arrive."""
import threading
import numpy as np
import time

from .backend import NO_INFO


def run_stream(eng, backend, asr, pcm, listen_s=12.0, realtime=True, verbose=True):
    """Streams `pcm` (float32 mono at Mimi's rate) through the engine, one 80 ms frame per step.

    On `<ret>` (when no retrieval is pending) the last `listen_s` seconds of user audio are
    transcribed, the backend is asked, and the reference is injected on the next frame. With
    `realtime` the loop keeps the 1.0x frame clock. Returns dict(agent, tokens, events).
    """
    fs, sr = eng.frame_size, int(eng.mimi.sample_rate)
    n = len(pcm) // fs
    lock, queue, events, tokens, out = threading.Lock(), [], [], [], []
    pending = [False]

    def kick(i):
        lo = max(0, (i + 1) * fs - int(listen_s * sr))
        q = asr.transcribe(pcm[lo:(i + 1) * fs], sr)
        ev = {"t_ret": i / eng.frame_rate, "question": q, "reference": None}
        if q.strip():
            ev["reference"] = backend.retrieve(q)
            if ev["reference"] != NO_INFO:
                with lock:
                    queue.append(ev)
        events.append(ev)
        pending[0] = False

    t_start = time.time()
    for i in range(n):
        with lock:
            ready = queue.pop(0) if queue else None
        if ready is not None:
            ready["t_inj"] = i / eng.frame_rate
            eng.inject_context_span(ready["reference"])
            if verbose:
                print(f"[span @{ready['t_inj']:.1f}s] {ready['reference'][:100]!r}", flush=True)
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
