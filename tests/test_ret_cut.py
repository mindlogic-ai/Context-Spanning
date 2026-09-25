"""The <ret> cut: once the model has asked for retrieval, the running utterance ends after RET_CUT_S of silence
instead of the full UTT_END_F, so the final transcript (the question) starts earlier."""
import numpy as np

from contextspan.runtime import frame_stream as F


def _frames(voiced, silent, fs=1920):
    rng = np.random.default_rng(0)
    return [rng.normal(0, 0.1, fs).astype(np.float32) for _ in range(voiced)] + [np.zeros(fs, np.float32)] * silent


def test_end_now_returns_the_running_utterance_and_clears_it():
    u = F.Utterances()
    u.vad = None                                  # RMS rule: deterministic
    i = 0
    for i, fr in enumerate(_frames(10, F.RET_CUT_F)):
        u.feed(i, fr)
    assert u.speaking and u.usil == F.RET_CUT_F   # still running: UTT_END_F (9) silent frames not yet reached
    assert F.RET_CUT_F < F.UTT_END_F
    ev = u.end_now(i)
    assert ev == ("final", 0, 10)                 # frames 0..9 voiced, u1 = first silent frame
    assert not u.speaking and u.end_now(i) is None


def test_end_now_drops_a_too_short_utterance():
    u = F.Utterances(min_len_f=3)
    u.vad = None
    for i, fr in enumerate(_frames(2, 2)):
        u.feed(i, fr)
    assert u.end_now(3) is None and not u.speaking


def test_cut_is_configured_in_frames():
    assert F.RET_CUT_F == int(round(F.RET_CUT_S / 0.08))
    assert F.RET_CUT_S == 0.0 or F.RET_CUT_F >= 1
