"""Frame-clock streaming loop for a wav file: user audio in, agent audio out, spans injected as they arrive.

The `<ret>` handling is the DuetaSpan live runtime's: the recent user audio is transcribed; the
backend is asked with the user's profile, what the agent has already said, the requests already
answered and the Context DB working text; a tool result or a direct answer is injected as a Context
Span; a `<ret>` with no transcript, or a router "nothing to do" that no tool produced, injects
NOTHING (an abstain span exists in training only when a real tool ran and came back empty). The
frame clock never waits for any of it.
"""
import json
import os
import threading
import time

import numpy as np

from ..duetaspan.runtime.backend.context_db import ContextDB, ContextProfile

# Wall-clock budget from <ret> to injection. The training corpus's ret->span delays have
# p50 0.72 s, p99 2.3 s, max 2.4 s: a span later than that is outside what the model
# was trained to wait for, and by then it has usually answered without it.
RET_DEADLINE_S = float(os.environ.get("CS_RET_DEADLINE_S", "2.5"))

# The user's speech is transcribed continuously, by utterance: a frame is voiced above RMS_SPEECH, an
# utterance ends after UTT_END_F silent frames (0.72 s), a partial transcript is taken every
# PARTIAL_EVERY_F frames (1.6 s) while it runs. On <ret> the freshest transcript (< UTT_CACHE_S old) is
# the question - no second ASR call; if the user is still mid-sentence the question waits for the
# utterance to end, at most RET_UTT_WAIT_S (MoshiRAG waits a fixed second; here the wait ends with the
# sentence). Only with no utterance at all does the fixed window get transcribed.
RMS_SPEECH = 0.01
RMS_FLOOR = 0.008      # -42 dBFS after levelling: below this nothing is speech, whatever the VAD says
UTT_END_F = 9
PARTIAL_EVERY_F = 20
UTT_CACHE_S = 3.0
RET_UTT_WAIT_S = float(os.environ.get("CS_RET_UTT_WAIT_S", "1.0"))
# Once the model has emitted <ret> and is waiting for the running utterance to end, the utterance ends after
# RET_CUT_S of silence instead of the full UTT_END_F (0.72 s): the <ret> is the model's own signal that the
# question is over, so the final transcript starts up to 0.32 s earlier. 0.4 s is the debounce that keeps the
# question whole; 0 disables the cut.
RET_CUT_S = float(os.environ.get("CS_RET_CUT_S", "0.4"))
RET_CUT_F = int(round(RET_CUT_S / 0.08))


def ret_question_plan(cache, t_now, speaking):
    """What a `<ret>` at time `t_now` uses as its question.

    "cache"  - the last utterance ended, its FINAL transcript is in the cache and fresh: that is the question.
    "wait"   - the user is still speaking, or the utterance ended but only a PARTIAL transcript is cached
               (the final one is still in the ASR): wait for the final, bounded by RET_UTT_WAIT_S.
    "window" - no utterance around: transcribe the fixed window before the <ret>.
    A partial is cut wherever its 1.6 s tick fell ("...arthur's magazine or?"), so it is never the question."""
    fresh = bool(cache.get("text")) and (t_now - cache.get("t", -1e9)) < UTT_CACHE_S
    if fresh and cache.get("final") and not speaking:
        return "cache"
    if speaking or (fresh and not cache.get("final")):
        return "wait"
    return "window"


class Utterances:
    """Causal utterance segmentation over user frames. `feed(i, frame)` returns the events due at frame i:
    ("partial", u0, i) while an utterance runs, ("final", u0, u1) when it has ended."""

    def __init__(self, min_len_f=3, sample_rate=24000):
        self.u0, self.usil, self.last_partial, self.min_len_f = None, 0, 0, min_len_f
        self.sr = sample_rate
        # WebRTC VAD (mode 3, the most selective) on 20 ms sub-frames at 16 kHz. An absolute RMS
        # threshold cannot tell amplified room noise from speech once the channel is levelled; the
        # VAD can. Falls back to the RMS rule if the library is missing.
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(3)
        except Exception:
            self.vad = None

    @property
    def speaking(self):
        return self.u0 is not None

    def voiced(self, frame):
        rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        if self.vad is None or rms < RMS_FLOOR:
            return rms >= RMS_SPEECH
        from scipy.signal import resample_poly
        x = resample_poly(frame.astype(np.float32), 2, 3) if self.sr == 24000 else frame  # 24 kHz -> 16 kHz
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
        n, hits = 0, 0
        for off in range(0, len(pcm) - 640 + 1, 640):            # 20 ms = 320 samples = 640 bytes
            n += 1
            hits += self.vad.is_speech(pcm[off:off + 640], 16000)
        return n > 0 and hits * 2 >= n                            # at least half the sub-frames

    def end_now(self, i):
        """End the running utterance at frame i without the full UTT_END_F silence (the <ret> cut).
        Returns its ("final", u0, u1) event, or None when nothing (long enough) is running."""
        if self.u0 is None:
            return None
        u0, u1 = self.u0, i - self.usil + 1
        self.u0, self.usil = None, 0
        return ("final", u0, u1) if u1 - u0 >= self.min_len_f else None

    def feed(self, i, frame):
        out = []
        if self.voiced(frame):
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


def run_stream(eng, backend, asr, pcm, ctx=None, asr_window_s=12.0, realtime=True, verbose=True,
               ret_deadline_s=RET_DEADLINE_S, ret_fixed_wait_s=None):
    """Streams `pcm` (float32 mono at Mimi's rate) through the engine, one 80 ms frame per step.
    `ctx` = user profile (name, city, timezone, lat, lon, notes). Returns dict(agent, tokens, events).
    `ret_fixed_wait_s`: benchmark only (moshi-rag run_inference.py `stt_wait_time`): every `<ret>` waits
    exactly this long, then the audio window so far is transcribed and sent; the utterance plan and the
    `<ret>` cut are not used. None (serving) keeps the utterance-transcript path."""
    fs, sr = eng.frame_size, int(eng.mimi.sample_rate)
    n = len(pcm) // fs
    ctx = dict(ctx or {})
    db = ContextDB(ContextProfile(**ctx))
    lock, queue, events, tokens, out = threading.Lock(), [], [], [], []
    utts, cache = Utterances(), {"text": None, "t": -1e9, "final": False}
    ret_wait = [None]                      # frame index of a <ret> waiting for the running utterance to end
    pending = [False]
    transcripts = []                       # (t_end_s, partial|final, text) for the caller's record

    def kick(i, question=None, end=None):
        """`i` = the <ret> frame (t_ret); the audio window ends at `end` (default `i`)."""
        end = i if end is None else end
        lo = max(0, (end + 1) * fs - int(asr_window_s * sr))
        said = eng.decode_text([t for t in tokens[-64:] if t is not None])
        r = retrieve_for_ret(backend, asr, pcm[lo:(end + 1) * fs], sr, ctx, db, said, events, question=question)
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
                if kind == "final" and text != db.last_user_text():
                    db.add_user_turn(text)
            waiting = ret_wait[0]
            if kind == "final" and waiting is not None:
                ret_wait[0] = None
                threading.Thread(target=kick, args=(waiting, text or None), daemon=True).start()

    fixed_kick = [None, None]              # (frame at which a fixed-wait <ret> fires, its <ret> frame) (benchmark)

    def start_ret(i):
        """The question for this <ret>: the fresh utterance transcript, else wait for the sentence, else the window."""
        pending[0] = True
        if ret_fixed_wait_s is not None:
            fixed_kick[0], fixed_kick[1] = i + int(round(ret_fixed_wait_s * eng.frame_rate)), i
            return
        plan = ret_question_plan(cache, i / eng.frame_rate, utts.speaking)
        if plan == "cache":
            threading.Thread(target=kick, args=(i, cache["text"]), daemon=True).start()
        elif plan == "wait":
            ret_wait[0] = i                # resolved by the utterance's final transcript or by the cap below
        else:
            threading.Thread(target=kick, args=(i,), daemon=True).start()

    t_start = time.time()
    for i in range(n):
        with lock:
            ready = queue.pop(0) if queue else None
        if ready is not None:
            ready["t_inj"] = i / eng.frame_rate
            ready["late"] = ready["inject"] is not None and (ready["t_inj"] - ready["t_ret"]) > ret_deadline_s
            if ready["late"]:
                ready["inject"] = None
            ready["frames"] = eng.inject_context_span(ready["inject"]) if ready["inject"] else 0
            ready["prefill_ms"] = round(eng.last_prefill_ms, 1) if ready["inject"] else 0.0
            events.append(ready)
            if verbose:
                shown = ("(late, dropped)" if ready["late"] else "(no span)") if ready["inject"] is None \
                    else repr(ready["inject"][:100])
                print(f"[span @{ready['t_inj']:.1f}s] q={ready['question']!r} src={ready['src']} -> {shown}", flush=True)
        frame = pcm[i * fs:(i + 1) * fs]
        for kind, u0, u1 in utts.feed(i, frame):
            threading.Thread(target=transcribe_utt, args=(kind, u0, u1), daemon=True).start()
        if fixed_kick[0] is not None and i >= fixed_kick[0]:
            ret_i, fixed_kick[0] = fixed_kick[1], None
            threading.Thread(target=kick, args=(ret_i, None, i), daemon=True).start()   # window up to now
        if RET_CUT_F and ret_wait[0] is not None and utts.speaking and utts.usil >= RET_CUT_F:
            ev = utts.end_now(i)               # <ret> already out and the user quiet for RET_CUT_S: the question is over
            if ev is not None:
                threading.Thread(target=transcribe_utt, args=ev, daemon=True).start()
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
