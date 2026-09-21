"""The release stack for the benchmark: Engine + DuetaSpan backend + ASR, warmed, with a
deterministic voice per sample. Evaluation is separate from the runtime package: nothing in
`contextspan/` imports this folder. Run the runners from the repository root (`python -m benchmark...`)."""
import hashlib

import numpy as np

from contextspan.duetaspan.align.asr import ASR, _resample
from contextspan.duetaspan.runtime.backend.realtime import RealtimeBackend
from contextspan.model import Engine, load_voice
from contextspan.model.sequence_convention import transcript
from contextspan.runtime.frame_stream import run_stream

__all__ = ["ASR", "CTX", "Engine", "RealtimeBackend", "Stack", "VOICES", "load_mono", "load_voice", "run_stream", "transcript"]

CTX = {"city": "Seoul", "timezone": "Asia/Seoul"}      # the profile every paper run used
VOICES = ("f0", "f1", "f2")                            # the released voice prompts


def load_mono(path, want):
    import soundfile as sf
    pcm, sr = sf.read(path, dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm.mean(1)
    if sr != want:
        pcm = _resample(pcm, sr, want)
    return pcm


class Stack:
    """Engine + backend + ASR, warmed. `voice_for(key)` is deterministic per sample: a prefix without a
    voice block is outside the training distribution, and the paper runs drew the voice from the
    training voice pool by a hash of the sample key; here the pool is the released f0/f1/f2."""

    def __init__(self, checkpoint=None, temp=0.8, temp_text=0.7, backend=None, voice=None):
        self.eng = Engine(checkpoint, temp=temp, temp_text=temp_text)
        self.backend = backend if backend is not None else RealtimeBackend()
        self.asr = ASR()
        self.sr, self.fs = int(self.eng.mimi.sample_rate), self.eng.frame_size
        self._voices = {v: load_voice(v) for v in ((voice,) if voice else VOICES)}
        try:                                               # first-call warm-up off the item path
            self.asr.transcribe(np.zeros(self.sr, np.float32), self.sr)
            if isinstance(self.backend, RealtimeBackend):
                self.backend.retrieve("what time is it now?", ctx=CTX, aux_context="", history=[])
        except Exception as e:
            print(f"[warm] skipped: {e}", flush=True)

    def voice_for(self, key):
        names = list(self._voices)
        return self._voices[names[int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) % len(names)]]

    def run(self, persona, key, pcm, backend=None, ctx=CTX, verbose=False, **stream_kw):
        self.eng.reset()
        self.eng.set_persona(persona, self.voice_for(key))
        return run_stream(self.eng, backend or self.backend, self.asr, pcm, ctx=ctx, verbose=verbose, **stream_kw)
