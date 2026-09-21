"""The batched persona prefix leaves the stream exactly where stepping it would.

Needs the weights, so it is skipped unless CS_TEST_CHECKPOINT points at a Context Spanning .pt.
Runs on CPU in float32 by default (CS_TEST_DEVICE=cuda when a GPU is free): both paths from a fresh
stream, then the cache ring, the provided flags, the offset, every streaming-state tensor of the
backbone and the next five sampled frames (same seed) are compared."""
import dataclasses
import os

import pytest
import torch

CKPT = os.environ.get("CS_TEST_CHECKPOINT")
DEVICE = os.environ.get("CS_TEST_DEVICE", "cpu")
PERSONA = "You are a warm, concise voice assistant who lives in Seoul and likes jazz."


def _state_tensors(module):
    """Every tensor held by a streaming state anywhere under `module`, keyed by module path and field."""
    out = {}
    for name, m in module.named_modules():
        st = getattr(m, "_streaming_state", None)
        if st is None:
            continue
        fields = dataclasses.fields(st) if dataclasses.is_dataclass(st) else []
        for f in fields:
            v = getattr(st, f.name)
            if torch.is_tensor(v):
                out[f"{name}.{f.name}"] = v.detach().clone()
            elif isinstance(v, (int, float)):
                out[f"{name}.{f.name}"] = torch.tensor(v)
    return out


@pytest.fixture(scope="module")
def engine():
    if not CKPT:
        pytest.skip("CS_TEST_CHECKPOINT not set")
    from contextspan.model.engine import Engine
    eng = Engine(checkpoint=CKPT, device=DEVICE)
    if DEVICE == "cpu":
        eng.lm.float()
    return eng


def _run(eng, stepwise, voice, seed=0):
    eng.reset()
    torch.manual_seed(seed)
    (eng.set_persona_stepwise if stepwise else eng.set_persona)(PERSONA, voice)
    snap = _state_tensors(eng.lm)
    ring = eng.lm_gen._streaming_state
    snap["ring.cache"] = ring.cache.clone(); snap["ring.provided"] = ring.provided.clone()
    snap["ring.offset"] = torch.tensor(int(ring.offset))
    torch.manual_seed(seed + 1)
    frames = [eng.lm_gen.step(input_tokens=eng._sine) for _ in range(5)]
    frames = [f.clone() for f in frames if f is not None]
    return snap, frames


def test_batched_prefix_matches_stepping(engine):
    torch.manual_seed(123)
    card = int(engine.lm.card)
    voice = torch.randint(0, card, (8, 24), dtype=torch.long)       # a short synthetic voice prompt
    ref_state, ref_frames = _run(engine, stepwise=True, voice=voice)
    new_state, new_frames = _run(engine, stepwise=False, voice=voice)
    assert ref_state.keys() == new_state.keys()
    for k in ref_state:
        a, b = ref_state[k], new_state[k]
        assert a.shape == b.shape, k
        if a.is_floating_point():
            assert torch.allclose(a, b, atol=2e-4, rtol=1e-4), f"{k}: max |diff| {(a - b).abs().max().item():.3e}"
        else:
            assert torch.equal(a, b), k
    assert len(ref_frames) == len(new_frames) > 0
    for i, (a, b) in enumerate(zip(ref_frames, new_frames)):
        assert torch.equal(a, b), f"frame {i} differs"


def test_prefix_without_voice(engine):
    ref_state, _ = _run(engine, stepwise=True, voice=None)
    new_state, _ = _run(engine, stepwise=False, voice=None)
    assert torch.equal(ref_state["ring.cache"], new_state["ring.cache"])
    assert int(ref_state["ring.offset"]) == int(new_state["ring.offset"])
