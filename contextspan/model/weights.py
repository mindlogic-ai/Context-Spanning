"""Where the weights come from: the PersonaPlex base, the Context Spanning checkpoint, the voices.

`CS_BASE_DIR` / `CS_WEIGHTS_DIR` point at local copies of the two Hub repositories; otherwise the
files are downloaded with `huggingface_hub`.
"""
import os

import sentencepiece
import torch
from huggingface_hub import hf_hub_download

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
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    lm.load_state_dict({k.replace("._orig_mod.", "."): v for k, v in sd.items()})
    lm.eval()
    return lm, mimi, spm


def load_voice(name_or_path="f0"):
    """Voice prompt = agent-voice Mimi codes [8, P] saved as {'codes': LongTensor}. A bare name
    (f0-f3, m0-m3, seonghyeon) is fetched as voices/<name>.pt from the weights repo."""
    path = name_or_path if os.path.exists(name_or_path) else hf_path(WEIGHTS_REPO, f"voices/{name_or_path}.pt")
    return torch.load(path, map_location="cpu")["codes"].long()
