"""The question a <ret> uses: never a partial transcript."""
from contextspan.runtime.frame_stream import ret_question_plan


def test_final_still_in_asr_waits():
    # the utterance ended, its final transcript has not returned yet, the cache holds a partial
    cache = {"text": "which magazine was started first, arthur's magazine or?", "t": 4.2, "final": False}
    assert ret_question_plan(cache, 5.0, speaking=False) == "wait"


def test_fresh_final_is_the_question():
    cache = {"text": "which magazine was started first, arthur's magazine or first for women?", "t": 5.8, "final": True}
    assert ret_question_plan(cache, 6.0, speaking=False) == "cache"


def test_still_speaking_waits_even_with_a_final_cached():
    cache = {"text": "earlier sentence", "t": 4.9, "final": True}
    assert ret_question_plan(cache, 5.0, speaking=True) == "wait"


def test_stale_or_empty_cache_uses_the_window():
    assert ret_question_plan({"text": "old final", "t": 1.0, "final": True}, 5.0, speaking=False) == "window"
    assert ret_question_plan({"text": None, "t": -1e9, "final": False}, 5.0, speaking=False) == "window"
