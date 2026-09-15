"""The streaming engine: one user frame in, one agent frame out, Context Spans read on arrival.

`Engine` wraps the PersonaPlex model with the Context Spanning checkpoint loaded: it builds the
masked persona prefix, steps the model one 80 ms frame at a time, reports `<ret>` and reads a
reference into the stream as a Context Span block exactly as the training assembler wrote it.
"""
import time

import numpy as np
import torch

from ..moshi.models import LMGen
from . import context_span_block as block
from . import prefix_prefill as prefix
from .sequence_convention import (N_AUDIO_CB, RET_TOKEN_ID, SILENCE_TOKENS, SINE_TOKENS, SPAN_CLOSE_ID,
                                  SPAN_OPEN_ID, TEXT_PAD, persona_prompt)
from .weights import load_mimi, load_model


class Engine:
    """One user frame in, one agent frame out; `<ret>` triggers the backend; spans are injected
    as masked blocks exactly as in training."""

    def __init__(self, checkpoint=None, device="cuda", temp=0.8, temp_text=0.7, cpu_offload=False):
        self.device = device
        self.lm, self.mimi, self.spm = load_model(checkpoint, device, cpu_offload)
        self.user_mimi = load_mimi(device)
        self.frame_rate = float(self.mimi.frame_rate)
        self.frame_size = int(round(self.mimi.sample_rate / self.frame_rate))
        kw = dict(sample_rate=int(self.mimi.sample_rate), frame_rate=self.frame_rate)
        self.lm_gen = (LMGen(self.lm, device=device, use_sampling=False, **kw) if temp <= 0
                       else LMGen(self.lm, device=device, temp=temp, temp_text=temp_text, **kw))
        self._sil = torch.tensor(SILENCE_TOKENS, device=device)[None, :, None]
        self._sine = torch.tensor(SINE_TOKENS, device=device)[None, :, None]
        self._pending_exit_cb0 = None
        self.frames = 0
        self.last_prefill_ms = 0.0
        self.reset()
        with torch.no_grad():
            for _ in range(4):
                self.lm_gen.step(input_tokens=self._sine)
            # The block forward is a multi-position path the single-step loop never exercises; its
            # first call in a process costs ~0.7 s (measured 741 ms vs 58 ms for the second call,
            # 2026-09-09). Pay it here, not on the first span of a conversation.
            block.prefill(self.lm_gen, [SPAN_OPEN_ID] + [TEXT_PAD] * 6 + [SPAN_CLOSE_ID], self._sil, self._sine)
        self.reset()

    def reset(self):
        """Restart the streams for a new conversation."""
        for m in (self.lm_gen, self.mimi, self.user_mimi):
            try:
                m._stop_streaming()
            except Exception:
                pass
            m.streaming_forever(1)
        self._pending_exit_cb0, self.frames = None, 0

    def _prefix_steps(self, system_prompt, voice_codes=None):
        """The prefix as forced step triples: voice codes -> silence -> persona text -> silence."""
        pad = int(self.lm_gen.zero_text_code)
        text = lambda t: torch.tensor([t], dtype=torch.long, device=self.device)
        steps = []
        if voice_codes is not None:
            vc = voice_codes.to(self.device)
            steps += [(self._sine, vc[:, p][None, :, None], text(pad)) for p in range(vc.shape[1])]
            steps.append((self._sine, self._sil, text(pad)))
        ids = self.spm.encode(persona_prompt(system_prompt))
        steps += [(self._sine, self._sil, text(int(t))) for t in ids]
        if ids:
            steps.append((self._sine, self._sil, text(pad)))
        return steps

    @torch.no_grad()
    def set_persona(self, system_prompt, voice_codes=None):
        """Prefix: voice codes -> silence -> persona text -> silence, all forced, read in one backbone
        forward (~0.1 s instead of ~5.5 s of single steps, #38). Same state as `set_persona_stepwise`."""
        prefix.prefill_forced_steps(self.lm_gen, self._prefix_steps(system_prompt, voice_codes))

    @torch.no_grad()
    def set_persona_stepwise(self, system_prompt, voice_codes=None):
        """Reference path: the same prefix forced one `lm_gen.step` at a time (what the batched read is checked against)."""
        for user_codes, agent_codes, text_token in self._prefix_steps(system_prompt, voice_codes):
            self.lm_gen.step(input_tokens=user_codes, moshi_tokens=agent_codes, text_token=text_token)

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
            if t == SPAN_CLOSE_ID:
                ac = ac.clone()
                ac[:, 0, 0] = self._pending_exit_cb0
            self._pending_exit_cb0 = None
        self.frames += 1
        return {"agent_pcm": self.mimi.decode(ac)[0, 0].cpu().numpy(), "text_token": t,
                "is_ret": t == RET_TOKEN_ID}

    @torch.no_grad()
    def inject_context_span(self, reference: str) -> int:
        """Read the reference into the stream as a masked Context Span block in one batched forward.
        Returns the number of frames the block consumed; `last_prefill_ms` holds its wall-clock cost.
        Measured 2026-09-09 (RTX PRO 6000 Blackwell): 27 ms up to 250 tokens, 32/38/41 ms at 300/350/400,
        so prefill + the next 32 ms step stays inside the 80 ms frame up to 400 tokens."""
        ids = block.span_ids(reference, self.spm)
        self._pending_exit_cb0 = block.pending_exit_cb0(self.lm_gen)
        t0 = time.perf_counter()
        n = block.prefill(self.lm_gen, ids, self._sil, self._sine)
        if self.device != "cpu":
            torch.cuda.synchronize()
        self.last_prefill_ms = (time.perf_counter() - t0) * 1e3
        return n

    @torch.no_grad()
    def clone_voice(self, pcm, sample_rate: int) -> torch.Tensor:
        """Agent-voice Mimi codes [8, P] from a recording (resampled, -24 LUFS), the way PersonaPlex
        conditions on a voice. Resets the streams; call it between conversations."""
        from ..moshi.models.lm import normalize_audio
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
        """Words only: control tokens and padding dropped."""
        skip = {TEXT_PAD, 0, 1, 2, RET_TOKEN_ID, SPAN_OPEN_ID, SPAN_CLOSE_ID}
        return self.spm.decode([int(t) for t in tokens if t is not None and int(t) not in skip])
