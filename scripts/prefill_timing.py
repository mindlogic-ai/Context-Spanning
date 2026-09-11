"""Wall-clock of a Context Span block read + the next step against the span length, on the release engine.

    python scripts/prefill_timing.py <checkpoint.pt> <out.json>     # from the repository root, one idle GPU
"""
import json
import os
import statistics as st
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from contextspan.model import Engine                                        # noqa: E402
from contextspan.model import context_span_block as block                  # noqa: E402
from contextspan.model.sequence_convention import SPAN_CLOSE_ID, SPAN_OPEN_ID   # noqa: E402

CK = sys.argv[1]; OUT = sys.argv[2]
eng = Engine(checkpoint=CK, device="cuda")
spm = eng.spm
fs = eng.frame_size
rng = np.random.default_rng(0)
PASSAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefill_passage.txt")).read()
body = spm.encode(PASSAGE)
print("passage tokens", len(body), flush=True)

def frame():                       # quiet room-tone, the way an idle user sounds
    return (rng.standard_normal(fs) * 1e-3).astype(np.float32)

def sync(): torch.cuda.synchronize()

@torch.no_grad()
def timed_step():
    sync(); t = time.perf_counter(); eng.step(frame()); sync(); return (time.perf_counter() - t) * 1e3

@torch.no_grad()
def timed_prefill(ids):
    sync(); t = time.perf_counter()
    eng._pending_exit_cb0 = block.pending_exit_cb0(eng.lm_gen)
    n = block.prefill(eng.lm_gen, ids, eng._sil, eng._sine)
    sync(); return (time.perf_counter() - t) * 1e3, n

res = {"gpu": torch.cuda.get_device_name(0), "ckpt": CK, "frame_ms": 1000 / eng.frame_rate}
with torch.no_grad():
    for _ in range(60): eng.step(frame())          # warm ring + kernels
base = [timed_step() for _ in range(100)]
res["step_only_ms"] = dict(p50=st.median(base), p95=float(np.percentile(base, 95)), max=max(base))
print("step only", res["step_only_ms"], flush=True)
res["by_n"] = {}
for n in (25, 50, 100, 150, 200, 250, 300, 350, 400):
    ids = [SPAN_OPEN_ID] + body[: n - 2] + [SPAN_CLOSE_ID]
    pre, nxt, tot = [], [], []
    for r in range(12):
        with torch.no_grad():
            for _ in range(25): eng.step(frame())  # live frames between injections
        p, k = timed_prefill(ids)
        s = timed_step()
        if r >= 2: pre.append(p); nxt.append(s); tot.append(p + s)
    q = lambda v: dict(p50=round(st.median(v), 2), p95=round(float(np.percentile(v, 95)), 2), max=round(max(v), 2))
    res["by_n"][n] = dict(tokens=len(ids), prefill_ms=q(pre), next_step_ms=q(nxt), prefill_plus_step_ms=q(tot),
                          fits_80ms=max(tot) <= 80.0)
    print(n, res["by_n"][n], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
