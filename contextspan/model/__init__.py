"""The speech model side of Context Spanning.

  weights.py              the PersonaPlex base, the released checkpoint and voices (Hub or local)
  engine.py               `Engine`: persona prefix, one frame per step, `<ret>`, span reads
  context_span_block.py   the Context Span block read into the KV stream in one forward
  prefix_prefill.py       the per-connection prefix (voice codes, persona text) read in one forward
  sequence_convention.py  token ids, placeholders, prefix and training-sequence layout
"""
from .engine import Engine
from .weights import BASE_REPO, WEIGHTS_FILE, WEIGHTS_REPO, hf_path, load_model, load_voice

__all__ = ["Engine", "BASE_REPO", "WEIGHTS_FILE", "WEIGHTS_REPO", "hf_path", "load_model", "load_voice"]
