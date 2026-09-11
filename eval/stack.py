"""Shared pieces of the evaluation harness: the release stack (Engine + DuetaSpan backend + ASR),
resampling, a deterministic voice per sample, the transcript helper. Evaluation is separate from the
runtime package: nothing in `contextspan/` imports this folder."""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from contextspan.duetaspan.align.asr import ASR, _resample            # noqa: E402
from contextspan.duetaspan.runtime.backend.realtime import RealtimeBackend, NO_INFO   # noqa: E402
from contextspan.model import Engine, load_voice                        # noqa: E402
from contextspan.stream import run_stream                               # noqa: E402
from main import transcript                                             # noqa: E402

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
        self.backend = backend if backend is not None else RealtimeBackend(cache=False)
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

    def run(self, persona, key, pcm, backend=None, ctx=CTX, verbose=False):
        self.eng.reset()
        self.eng.set_persona(persona, self.voice_for(key))
        return run_stream(self.eng, backend or self.backend, self.asr, pcm, ctx=ctx, verbose=verbose)
