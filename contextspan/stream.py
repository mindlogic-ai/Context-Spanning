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

# The user's speech is transcribed continuously, by utterance, the way the DuetaSpan live server does it:
# a frame is voiced above RMS_SPEECH, an utterance ends after UTT_END_F silent frames (0.72 s), a partial
# transcript is taken every PARTIAL_EVERY_F frames (1.6 s) while it runs. On <ret> the freshest transcript
# (< UTT_CACHE_S old) is the question — no second ASR call; if the user is still mid-sentence the question
# waits for the utterance to end, at most RET_UTT_WAIT_S (MoshiRAG waits a fixed second; here the wait ends
# with the sentence). Only with no utterance at all does the old fixed window get transcribed.
RMS_SPEECH = 0.01
UTT_END_F = 9
PARTIAL_EVERY_F = 20
UTT_CACHE_S = 3.0
RET_UTT_WAIT_S = float(os.environ.get("CS_RET_UTT_WAIT_S", "1.0"))


class Utterances:
    """Causal utterance segmentation over user frames. `feed(i, frame)` returns the events due at frame i:
    ("partial", u0, i) while an utterance runs, ("final", u0, u1) when it has ended."""

    def __init__(self, min_len_f=3):
        self.u0, self.usil, self.last_partial, self.min_len_f = None, 0, 0, min_len_f

    @property
    def speaking(self):
        return self.u0 is not None

    def feed(self, i, frame):
        out = []
        rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        if rms >= RMS_SPEECH:
            if self.u0 is None:
                self.u0, self.last_partial = i, i
            self.usil = 0
        elif self.u0 is not None:
            self.usil += 1
            if self.usil >= UTT_END_F:
                u0, u1 = self.u0, i - self.usil + 1
                self.u0, self.usil = None, 0
                if u1 - u0 >= self.min_len_f:
                    out.append(("final", u0, u1))
                return out
        if self.u0 is not None and i - self.last_partial >= PARTIAL_EVERY_F:
            self.last_partial = i
            out.append(("partial", self.u0, i))
        return out


def retrieve_for_ret(backend, asr, clip, sample_rate, ctx, db, said_text, events, on_question=None, question=None):
    """One `<ret>`: (transcribe unless `question` is already known) -> ask the backend -> decide what is
    injected. Returns dict(question, reference, inject, src, args): `inject` is the span text, or None
    when nothing is injected; `reference` is the backend's raw answer (None when nothing was asked)."""
    if question is None:
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
    utts, cache = Utterances(), {"text": None, "t": -1e9, "final": False}
    ret_wait = [None]                      # frame index of a <ret> waiting for the running utterance to end
    pending = [False]
    transcripts = []                       # (t_end_s, partial|final, text) for the caller's record

    def kick(i, question=None):
        lo = max(0, (i + 1) * fs - int(asr_window_s * sr))
        said = eng.decode_text([t for t in tokens[-64:] if t is not None])
        r = retrieve_for_ret(backend, asr, pcm[lo:(i + 1) * fs], sr, ctx, db, said, events, question=question)
        r["t_ret"] = i / eng.frame_rate
        with lock:
            queue.append(r)
        pending[0] = False

    def transcribe_utt(kind, u0, u1):
        text = asr.transcribe(pcm[u0 * fs:(u1 + 1) * fs], sr) or ""
        with lock:
            if text.strip():
                cache.update(text=text, t=u1 / eng.frame_rate, final=(kind == "final"))
                transcripts.append((round(u1 / eng.frame_rate, 2), kind, text))
                if kind == "final":
                    db.add_user_turn(text) if text != db.last_user_text() else None
            waiting = ret_wait[0]
            if kind == "final" and waiting is not None:
                ret_wait[0] = None
                threading.Thread(target=kick, args=(waiting, text or None), daemon=True).start()

    def start_ret(i):
        """The question for this <ret>: the fresh utterance transcript, else wait for the sentence, else the window."""
        pending[0] = True
        if cache["text"] and (i / eng.frame_rate - cache["t"]) < UTT_CACHE_S and not utts.speaking:
            threading.Thread(target=kick, args=(i, cache["text"]), daemon=True).start()
        elif utts.speaking:
            ret_wait[0] = i                # resolved by the utterance's final transcript or by the cap below
        else:
            threading.Thread(target=kick, args=(i,), daemon=True).start()

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
        frame = pcm[i * fs:(i + 1) * fs]
        for kind, u0, u1 in utts.feed(i, frame):
            threading.Thread(target=transcribe_utt, args=(kind, u0, u1), daemon=True).start()
        with lock:                         # check-and-clear under the lock: the utterance thread may claim it first
            waiting = ret_wait[0]
            if waiting is not None and (i - waiting) / eng.frame_rate >= RET_UTT_WAIT_S:
                ret_wait[0] = None
            else:
                waiting = None
        if waiting is not None:
            threading.Thread(target=kick, args=(i,), daemon=True).start()   # sentence did not end in time: the window
        o = eng.step(frame)
        out.append(o["agent_pcm"] if o["agent_pcm"] is not None else np.zeros(fs, np.float32))
        tokens.append(o["text_token"])
        if o["is_ret"] and not pending[0]:
            if verbose:
                print(f"[ret @{i / eng.frame_rate:.1f}s]", flush=True)
            start_ret(i)
        if realtime:
            dt = t_start + (i + 1) / eng.frame_rate - time.time()
            if dt > 0:
                time.sleep(dt)
    return {"agent": np.concatenate(out), "tokens": tokens, "events": events, "transcripts": transcripts}
