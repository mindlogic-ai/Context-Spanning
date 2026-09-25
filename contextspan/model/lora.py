"""LoRA checkpoints: load an adapter checkpoint exactly as it was trained, without merging.

A LoRA checkpoint stores, for every adapted Linear `<name>`, the frozen base weight as
`<name>.base.weight` and the adapter as `<name>.lora_A` [r, in] / `<name>.lora_B` [out, r], plus
`lora = {"r": r, "alpha": alpha}` at the top level. The layers are rebuilt here with the same
forward the trainer used:

    y = base(x) + B(A(x)) * alpha / r

Modules that read `.weight` directly (the gated FFN hands `linear_in` / `linear_out` weights to a
fused kernel) get `base.weight + (B @ A) * alpha / r` in the base dtype, which is also what they
saw in training.

Merging the adapter into bf16 weights ahead of time is not equivalent: with a delta around 0.2 to
0.9 percent of the weight norm, bf16 rounding drops 20 to 60 percent of it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.bfloat16, device=dev), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.bfloat16, device=dev), requires_grad=False)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    @property
    def weight(self):
        return self.base.weight + (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale

    @property
    def bias(self):
        return self.base.bias


def adapted_layers(state_dict):
    """Names of the Linears a checkpoint adapts (keys already stripped of `._orig_mod.`)."""
    return sorted(k[: -len(".lora_A")] for k in state_dict if k.endswith(".lora_A"))


def wrap(lm: nn.Module, state_dict, r: int, alpha: float) -> int:
    """Replace every adapted Linear of `lm` with a LoRALinear so `state_dict` loads strictly."""
    names = adapted_layers(state_dict)
    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = lm.get_submodule(parent_name) if parent_name else lm
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, not nn.Linear")
        setattr(parent, attr, LoRALinear(base, r, alpha))
    return len(names)
