"""The per-connection prefix (voice codes, persona text) read into the stream in one forward.

Forcing the prefix through `lm_gen.step` one column at a time costs ~100 voice frames + ~70 persona
tokens + separators = ~180 steps of ~30 ms, ~5.5 s before the page can listen; one batched forward
takes ~0.1 s.
Every prefix column is fully forced (text row, agent audio rows, user audio rows), so a step does
nothing but write the column into the cache ring and advance the backbone KV by one position; the
depformer and the samplers are no-ops on it. The bookkeeping is kept exactly as `step` does it - the
same `prepare_step_input` writes the same ring positions, initial tokens and `provided` flags - and
only the backbone forward is batched: one chunk over the ~180 input columns instead of 180 calls.
The KV state, the ring and the offset afterwards are identical to stepping (tests/test_persona_prefill.py).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = ["prefill_forced_steps"]

Step = tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]   # (user codes, agent codes, text token)


@torch.no_grad()
def prefill_forced_steps(lm_gen, steps: Sequence[Step]) -> int:
    """Apply `steps` - each the (input_tokens, moshi_tokens, text_token) triple that would go to
    `lm_gen.step`, every row given - and run the backbone once over their input columns.
    Returns the number of columns forwarded (the first step of a fresh stream only seeds the ring)."""
    state = lm_gen._streaming_state
    inputs = []
    for user_codes, agent_codes, text_token in steps:
        prepared = lm_gen.prepare_step_input(user_codes, agent_codes, text_token)
        if prepared is None:               # offset 0: the initial column is written, nothing to predict yet
            continue
        input_, provided_, _target, model_input_position, _target_position = prepared
        if not bool(provided_.all()):
            raise ValueError("prefix columns must be fully forced; a sampled row needs the step-by-step path")
        inputs.append(input_.clone())
        # what process_transformer_output does to the state when every target row is provided
        state.provided[:, :, model_input_position] = False
        state.offset += 1
    if inputs:
        lm_gen.lm_model.forward_codes(torch.cat(inputs, dim=2))
    return len(inputs)
