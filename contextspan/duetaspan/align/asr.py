"""ASR client: posts the user's audio to the Qwen3-ASR endpoint and returns the transcript.

`MOSHICP_ASR_URL` names the endpoint (`scripts/env.sh` exports it). Two server flavours are spoken:
`qwen-asr-serve` (vLLM, the OpenAI audio API at `/v1/audio/transcriptions`) and the transformers server
in `runtime/asr_server.py` (`POST /transcribe`). The URL is trusted as given: a connect probe at start-up
times out whenever the box is momentarily busy, and a client demoted at that moment returns '' for the
rest of the process. Transient failures are absorbed per request instead.
"""

from __future__ import annotations

import io
import os
import re

import numpy as np
import requests
import soundfile as sf


def _resample(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D float32 PCM array from src_sr to dst_sr using scipy."""
    if src_sr == dst_sr:
        return pcm
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(pcm.astype(np.float32), up, down).astype(np.float32)


class ASR:
    def __init__(self) -> None:
        self.url = os.environ.get("MOSHICP_ASR_URL", "").strip() or "http://localhost:8990/transcribe"
        print(f"[ASR] Qwen3-ASR endpoint {self.url}", flush=True)

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 24_000) -> str:
        """Transcribe a mono float32 PCM array. Returns lowercased text, '' when the request fails."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1:
            raise ValueError(f"pcm must be 1-D, got shape {pcm.shape}")
        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        # The language may be forced; a context or hotword prompt may not: biasing the decoder toward
        # expected proper nouns makes it emit them from silence.
        data = {}
        lang = os.environ.get("MOSHICP_ASR_LANGUAGE", "").strip()
        if lang:
            data["language"] = lang
        # The OpenAI audio API needs a `model` field and prefixes the text with
        # "language English<asr_text>".
        if "/v1/audio/transcriptions" in self.url:
            data["model"] = os.environ.get("MOSHICP_ASR_MODEL", "qwen3-asr")
        try:
            r = requests.post(self.url, files={"file": ("a.wav", buf, "audio/wav")},
                              data=data or None, timeout=10)
            text = (r.json().get("text") or "").strip()
            text = re.sub(r"^language\s+\w+<asr_text>\s*", "", text)
            # MOSHICP_ASR_KEEP_CASE=1 keeps the case of spelled-out IDs ("A B C one two three") all the
            # way to the router's argument filling; the default is the lowercase contract.
            if os.environ.get("MOSHICP_ASR_KEEP_CASE", "").strip().lower() in ("1", "true", "on"):
                return text
            return text.lower()
        except Exception:
            return ""
