"""The Context Span block: the backend's reference read into the running stream in one forward.

The block is `n` columns, one per span token, each carrying the span token on the text row, the
SILENCE placeholder on the agent audio rows and the SINE placeholder on the user audio rows - the
same arrangement the training assembler wrote. `prefill` advances the backbone over the whole block
in a single forward; no live frame is copied, rewritten or delayed.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import sequence_convention as seq

__all__ = ["span_ids", "prefill", "pending_exit_cb0"]


def span_ids(reference_text: str, spm) -> list[int]:
    """Text-row ids of the block for the reference exactly as the backend returned it."""
    return seq.context_span_ids((reference_text or "").strip(), spm)


def pending_exit_cb0(lm_gen) -> Optional[torch.Tensor]:
    """Semantic code of the live frame the block is about to swallow (the readout is one frame
    late, so that frame's rendering is never returned to the caller). Returns None if the state is
    not available."""
    state = getattr(lm_gen, "_streaming_state", None)
    if state is None:
        return None
    ring = state.cache.shape[2]
    return state.cache[:, 1, (state.offset - 1) % ring].clone()


def prefill(lm_gen, ids: Sequence[int], agent_sil: torch.Tensor, user_sine: torch.Tensor) -> int:
    """Read the block with ONE batched backbone forward instead of `n` forced steps.

    Every block column is fully forced - span token, SILENCE agent placeholder, SINE user
    placeholder - so the depformer and the text sampler are no-ops on it: the only thing a forced
    step accomplishes is advancing the backbone KV by one position. A streaming transformer
    forwarding `n` positions in one chunk is the same computation as `n` single steps, so the block
    is read in one forward (~30 ms) instead of n x ~27 ms.

    Columns here are already delayed coordinates: the block is a constant column repeated `n`
    times, differing only in the text row. The model input for block column j is column j-1, so
    the forward is fed [last live column] + block[:n-1].
    """
    state = lm_gen._streaming_state
    lm = lm_gen.lm_model
    ring = state.cache.shape[2]
    n = len(ids)
    if n == 0:
        return 0
    device = agent_sil.device
    k_audio = agent_sil.shape[1]
    column = torch.empty(1, lm.num_codebooks, 1, dtype=torch.long, device=device)
    column[0, 1:1 + k_audio, 0] = agent_sil[0, :, 0]
    column[0, 1 + k_audio:, 0] = user_sine[0, :, 0]
    block = column.repeat(1, 1, n)
    block[0, 0, :] = torch.as_tensor(list(ids), dtype=torch.long, device=device)

    prev = state.cache[:, :, (state.offset - 1) % ring][:, :, None].clone()
    lm.forward_codes(torch.cat([prev, block[:, :, :n - 1]], dim=2))

    for j in range(max(0, n - ring), n):                  # keep the ring consistent for live steps
        col = (state.offset + j) % ring
        state.cache[:, :, col] = block[:, :, j]
        state.provided[:, :, col] = False
    state.offset += n
    state.provided[:, :, state.offset % ring] = False     # next live target is free to be sampled
    return n
