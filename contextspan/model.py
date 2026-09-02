"""Model loading and the streaming engine (PersonaPlex base + Context Spanning checkpoint)."""
import os
import numpy as np
import sentencepiece
import torch
from huggingface_hub import hf_hub_download
from moshi.models import LMGen, loaders

from .spans import N_AUDIO_CB, RET_TOKEN_ID, SILENCE_TOKENS, SINE_TOKENS, context_span_ids, persona_prompt

BASE_REPO = "nvidia/personaplex-7b-v1"
WEIGHTS_REPO = "mindlogicinc/context-spanning-7b"
WEIGHTS_FILE = "context_spanning_7b.pt"


def _hf(repo, name):
    """CS_BASE_DIR / CS_WEIGHTS_DIR point at local copies; otherwise download from the Hub."""
    local = os.environ.get("CS_BASE_DIR" if repo == BASE_REPO else "CS_WEIGHTS_DIR")
    if local and os.path.exists(os.path.join(local, name)):
        return os.path.join(local, name)
    return hf_hub_download(repo, name)


def load_model(checkpoint=None, device="cuda", cpu_offload=False):
    """Returns (lm, mimi, spm). checkpoint: local .pt path, or None to fetch the released weights."""
    mimi = loaders.get_mimi(_hf(BASE_REPO, loaders.MIMI_NAME), device)
    spm = sentencepiece.SentencePieceProcessor(_hf(BASE_REPO, loaders.TEXT_TOKENIZER_NAME))
    lm = loaders.get_moshi_lm(_hf(BASE_REPO, loaders.MOSHI_NAME), device=device, cpu_offload=cpu_offload)
    ck = checkpoint or _hf(WEIGHTS_REPO, WEIGHTS_FILE)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    lm.load_state_dict({k.replace("._orig_mod.", "."): v for k, v in sd.items()})
    lm.eval()
    return lm, mimi, spm


def load_voice(path):
    """Voice prompt = agent-voice Mimi codes [8, P] saved as {'codes': LongTensor}."""
    return torch.load(path, map_location="cpu")["codes"].long()


class Engine:
    """One user frame in, one agent frame out; `<ret>` triggers the backend; spans are injected
    as masked blocks exactly as in training."""

    def __init__(self, checkpoint=None, device="cuda", temp=0.8, temp_text=0.7, cpu_offload=False):
        self.device = device
        self.lm, self.mimi, self.spm = load_model(checkpoint, device, cpu_offload)
        self.user_mimi = loaders.get_mimi(_hf(BASE_REPO, loaders.MIMI_NAME), device)
        self.frame_rate = float(self.mimi.frame_rate)
        self.frame_size = int(round(self.mimi.sample_rate / self.frame_rate))
        kw = dict(sample_rate=int(self.mimi.sample_rate), frame_rate=self.frame_rate)
        self.lm_gen = (LMGen(self.lm, device=device, use_sampling=False, **kw) if temp <= 0
                       else LMGen(self.lm, device=device, temp=temp, temp_text=temp_text, **kw))
        self._sil = torch.tensor(SILENCE_TOKENS, device=device)[None, :, None]
        self._sine = torch.tensor(SINE_TOKENS, device=device)[None, :, None]
        self._prev = None
        self.frames = 0
        self.reset()
        with torch.no_grad():
            for _ in range(4):
                self.lm_gen.step(input_tokens=self._sine)
        self.reset()

    def reset(self):
        for m in (self.lm_gen, self.mimi, self.user_mimi):
            try:
                m._stop_streaming()
            except Exception:
                pass
            m.streaming_forever(1)
        self._prev, self.frames = None, 0

    def _forced(self, text_id, agent=None, user=None):
        tt = torch.tensor([text_id], dtype=torch.long, device=self.device)
        self.lm_gen.step(input_tokens=self._sine if user is None else user,
                         moshi_tokens=self._sil if agent is None else agent, text_token=tt)

    @torch.no_grad()
    def set_persona(self, system_prompt, voice_codes=None):
        """Prefix: voice codes -> silence -> persona text -> silence (all forced, no audio out)."""
        pad = int(self.lm_gen.zero_text_code)
        if voice_codes is not None:
            vc = voice_codes.to(self.device)
            for p in range(vc.shape[1]):
                self._forced(pad, agent=vc[:, p][None, :, None])
            self._forced(pad)
        ids = self.spm.encode(persona_prompt(system_prompt))
        for t in ids:
            self._forced(int(t))
        if ids:
            self._forced(pad)

    @torch.no_grad()
    def step(self, user_pcm):
        """user_pcm: float32 [frame_size] -> {'agent_pcm', 'text_token', 'is_ret'}."""
        u = torch.as_tensor(user_pcm, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        uc = self.user_mimi.encode(u)
        tok = self.lm_gen.step(input_tokens=uc)
        if tok is None:
            return {"agent_pcm": None, "text_token": None, "is_ret": False}
        ac = tok[:, 1:1 + N_AUDIO_CB]
        self._prev = (ac, uc)
        self.frames += 1
        t = int(tok[0, 0, 0])
        return {"agent_pcm": self.mimi.decode(ac)[0, 0].cpu().numpy(), "text_token": t,
                "is_ret": t == RET_TOKEN_ID}

    @torch.no_grad()
    def inject_context_span(self, reference: str) -> int:
        """Force the span tokens as a masked block; the last column carries the previous real
        acoustic codes (the training-time MOVE convention)."""
        ids = context_span_ids(reference, self.spm)
        for i, tid in enumerate(ids):
            if i == len(ids) - 1 and self._prev is not None:
                pa, pu = self._prev
                self._forced(tid, agent=torch.cat([self._sil[:, :1], pa[:, 1:]], 1),
                             user=torch.cat([self._sine[:, :1], pu[:, 1:]], 1))
            else:
                self._forced(tid)
        return len(ids)
