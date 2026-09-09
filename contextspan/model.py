"""Model loading and the streaming engine (PersonaPlex base + Context Spanning checkpoint)."""
import os
import numpy as np
import sentencepiece
import torch
from huggingface_hub import hf_hub_download
from .moshi.models import LMGen, loaders

from . import inject as inject_mod
from .spans import (N_AUDIO_CB, RET_TOKEN_ID, SILENCE_TOKENS, SINE_TOKENS, SPAN_TOKEN_ID, TEXT_PAD,
                    persona_prompt)

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


def load_voice(name_or_path="f0"):
    """Voice prompt = agent-voice Mimi codes [8, P] saved as {'codes': LongTensor}. A bare name
    ("f0", "f1", "f2") is fetched as voices/<name>.pt from the weights repo."""
    path = name_or_path if os.path.exists(name_or_path) else _hf(WEIGHTS_REPO, f"voices/{name_or_path}.pt")
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
        self._pending_exit_cb0 = None
        self.frames = 0
        self.reset()
        with torch.no_grad():
            for _ in range(4):
                self.lm_gen.step(input_tokens=self._sine)
            # The span block is read through `forward_codes`, a batched path the per-frame
            # loop never touches, so its one-time initialisation (~0.5 s on an RTX PRO 6000) would otherwise
            # land on the first span of every conversation as a hole in the agent's audio
            # (#28). Read one throwaway block here; reset() below leaves no trace of it.
            self.lm.forward_codes(torch.zeros(1, self.lm.num_codebooks, 32,
                                              dtype=torch.long, device=device))
        self.reset()

    def reset(self):
        for m in (self.lm_gen, self.mimi, self.user_mimi):
            try:
                m._stop_streaming()
            except Exception:
                pass
            m.streaming_forever(1)
        self._pending_exit_cb0, self.frames = None, 0

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
        tok = self.lm_gen.step(input_tokens=self.user_mimi.encode(u))
        if tok is None:
            return {"agent_pcm": None, "text_token": None, "is_ret": False}
        t = int(tok[0, 0, 0])
        ac = tok[:, 1:1 + N_AUDIO_CB]
        if self._pending_exit_cb0 is not None:
            # The readout runs one frame late, so the live frame a span swallows would otherwise come
            # back carrying the block's placeholder; restore its real semantic code.
            if t == SPAN_TOKEN_ID:
                ac = ac.clone()
                ac[:, 0, 0] = self._pending_exit_cb0
            self._pending_exit_cb0 = None
        self.frames += 1
        return {"agent_pcm": self.mimi.decode(ac)[0, 0].cpu().numpy(), "text_token": t,
                "is_ret": t == RET_TOKEN_ID}

    @torch.no_grad()
    def inject_context_span(self, reference: str) -> int:
        """Read the reference into the stream as a masked Context Span block in one batched forward.
        Returns the number of frames the block consumed."""
        ids = inject_mod.span_ids(reference, self.spm)
        self._pending_exit_cb0 = inject_mod.pending_exit_cb0(self.lm_gen)
        return inject_mod.prefill(self.lm_gen, ids, self._sil, self._sine)

    @torch.no_grad()
    def clone_voice(self, pcm, sample_rate: int) -> torch.Tensor:
        """Agent-voice Mimi codes [8, P] from a recording (resampled, -24 LUFS), the way PersonaPlex
        conditions on a voice. Resets the streams; call it between conversations."""
        from .moshi.models.lm import normalize_audio
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        want = int(self.mimi.sample_rate)
        if sample_rate != want:
            import librosa
            x = librosa.resample(x, orig_sr=sample_rate, target_sr=want)
        x = normalize_audio(x[None, :], want, -24.0)
        for m in (self.mimi, self.user_mimi):
            try:
                m._stop_streaming()
            except Exception:
                pass
        codes = self.mimi.encode(torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, 1, -1))
        self.reset()
        return codes[0].to(torch.long)

    def decode_text(self, tokens) -> str:
        skip = {TEXT_PAD, 0, 1, 2, RET_TOKEN_ID, SPAN_TOKEN_ID}
        return self.spm.decode([int(t) for t in tokens if t is not None and int(t) not in skip])
