"""Where the weights come from: the PersonaPlex base, the Context Spanning checkpoint, the voices.

`CS_BASE_DIR` / `CS_WEIGHTS_DIR` point at local copies of the two Hub repositories; otherwise the
files are downloaded with `huggingface_hub`.
"""
import os

import sentencepiece
import torch
from huggingface_hub import hf_hub_download

from . import lora
from ..moshi.models import loaders

BASE_REPO = "nvidia/personaplex-7b-v1"
WEIGHTS_REPO = "mindlogicinc/context-spanning-7b"
WEIGHTS_FILE = "context_spanning_7b.pt"


def hf_path(repo, name):
    """Local path of `name` in `repo`: the CS_*_DIR copy when present, else the Hub download."""
    local = os.environ.get("CS_BASE_DIR" if repo == BASE_REPO else "CS_WEIGHTS_DIR")
    if local and os.path.exists(os.path.join(local, name)):
        return os.path.join(local, name)
    return hf_hub_download(repo, name)


def load_mimi(device="cuda"):
    """The Mimi codec of the PersonaPlex base."""
    return loaders.get_mimi(hf_path(BASE_REPO, loaders.MIMI_NAME), device)


def load_tokenizer():
    """The SentencePiece text tokenizer of the PersonaPlex base."""
    return sentencepiece.SentencePieceProcessor(hf_path(BASE_REPO, loaders.TEXT_TOKENIZER_NAME))


def load_model(checkpoint=None, device="cuda", cpu_offload=False):
    """Returns (lm, mimi, spm). checkpoint: local .pt path, or None to fetch the released weights."""
    mimi = load_mimi(device)
    spm = load_tokenizer()
    lm = loaders.get_moshi_lm(hf_path(BASE_REPO, loaders.MOSHI_NAME), device=device, cpu_offload=cpu_offload)
    ck = checkpoint or hf_path(WEIGHTS_REPO, WEIGHTS_FILE)
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    sd = {k.replace("._orig_mod.", "."): v for k, v in ckpt.get("model", ckpt).items()}
    if lora.adapted_layers(sd):
        meta = ckpt.get("lora") if isinstance(ckpt, dict) else None
        if not meta:
            raise ValueError(f"{ck} carries LoRA adapters but no 'lora': {{'r', 'alpha'}} entry")
        lora.wrap(lm, sd, int(meta["r"]), float(meta["alpha"]))
    lm.load_state_dict(sd)
    lm.eval()
    return lm, mimi, spm


def load_voice(name_or_path="f0"):
    """Voice prompt = agent-voice Mimi codes [8, P] saved as {'codes': LongTensor}. A bare name
    (f0-f3, m0-m3, seonghyeon, or any file under the repo's voices/) is fetched as voices/<name>.pt.

    The voices are looked up on the Hub weights repo, never under CS_WEIGHTS_DIR: that
    variable points at a *checkpoint*, and a checkpoint snapshot may carry the voice files
    that were current when it was uploaded, so honouring it here would make "f0" a different
    voice depending on which checkpoint is loaded, with no error and no log line. CS_VOICES_DIR points at a local
    voices/ directory when one is wanted."""
    if os.path.exists(name_or_path):
        path = name_or_path
    else:
        local = os.environ.get("CS_VOICES_DIR")
        cand = os.path.join(local, f"{name_or_path}.pt") if local else ""
        path = cand if cand and os.path.exists(cand) else hf_hub_download(WEIGHTS_REPO, f"voices/{name_or_path}.pt")
    return torch.load(path, map_location="cpu")["codes"].long()
