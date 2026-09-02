"""Frame-clock streaming loop: user audio in, agent audio out, spans arriving asynchronously."""
import threading
import time
import numpy as np

from .backend import NO_INFO


def run_stream(eng, backend, asr, pcm, debounce_s=0.5, reroute_s=1.5, ret_cooldown_s=2.0,
               vad_thresh=0.01, verbose=True):
    """Streams `pcm` (float32 mono at Mimi's rate) through the engine at 1.0x frame clock.

    On `<ret>`: wait until the user has been silent for `debounce_s`, transcribe the current
    utterance, ask the backend, and queue the reference; it is injected on the next frame.
    If the user keeps talking within `reroute_s` after a span was issued, the utterance is
    transcribed again and the span replaced (re-route). Returns dict(agent, tokens, events).
    """
    fs, sr = eng.frame_size, int(eng.mimi.sample_rate)
    n = len(pcm) // fs
    rms = np.sqrt(np.mean(pcm[:n * fs].reshape(n, fs) ** 2, axis=1) + 1e-12)
    now, lock, queue, events, tokens, out = [0], threading.Lock(), [], [], [], []
    last_ret = [-10 ** 9]

    def utt_start(j):
        k = j
        while k > 0 and (j - k) < int(30 * eng.frame_rate):
            if k >= 12 and (rms[k - 12:k] < vad_thresh).all():
                break
            k -= 1
        return max(0, k)

    def wait_silence(j, cap=6.0):
        deadline = time.time() + cap
        while time.time() < deadline:
            j = min(n, max(j, now[0]))
            k = int(debounce_s * eng.frame_rate)
            if j >= n or (j >= k and (rms[j - k:j] < vad_thresh).all()):
                return j
            time.sleep(0.05)
        return j

    def kick(i):
        j = wait_silence(i)
        lo = utt_start(j)
        q = asr.transcribe(pcm[lo * fs:j * fs], sr)
        if not q.strip():
            return
        ref = backend.retrieve(q)
        ev = {"t_ret": i / eng.frame_rate, "question": q, "reference": ref}
        if ref != NO_INFO:
            with lock:
                queue.append(ev)
        events.append(ev)
        if reroute_s > 0:                          # user kept talking: re-transcribe + re-route
            t0 = time.time()
            while time.time() - t0 < reroute_s:
                j2 = min(n, max(j, now[0]))
                if (rms[j:j2] >= vad_thresh).sum() >= int(0.6 * eng.frame_rate):
                    j2 = wait_silence(j2)
                    q2 = asr.transcribe(pcm[lo * fs:j2 * fs], sr)
                    if len(q2.split()) >= len(q.split()) + 2:
                        ref2 = backend.retrieve(q2)
                        ev2 = {"t_ret": i / eng.frame_rate, "question": q2, "reference": ref2, "reroute": True}
                        with lock:
                            queue[:] = [e for e in queue if e is not ev]
                            if ref2 != NO_INFO:
                                queue.append(ev2)
                        events.append(ev2)
                    return
                if j2 >= n:
                    return
                time.sleep(0.05)

    t_start = time.time()
    for i in range(n):
        target = t_start + (i + 1) / eng.frame_rate
        with lock:
            ready = queue.pop(0) if queue else None
        if ready is not None:
            ready["t_inj"] = i / eng.frame_rate
            eng.inject_context_span(ready["reference"])
            if verbose:
                print(f"[span @{ready['t_inj']:.1f}s] {ready['reference'][:100]!r}", flush=True)
        o = eng.step(pcm[i * fs:(i + 1) * fs])
        now[0] = i + 1
        out.append(o["agent_pcm"] if o["agent_pcm"] is not None else np.zeros(fs, np.float32))
        tokens.append(o["text_token"])
        if o["is_ret"] and (i - last_ret[0]) > ret_cooldown_s * eng.frame_rate:
            last_ret[0] = i
            if verbose:
                print(f"[ret @{i / eng.frame_rate:.1f}s]", flush=True)
            threading.Thread(target=kick, args=(i,), daemon=True).start()
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)
    return {"agent": np.concatenate(out), "tokens": tokens, "events": events}
