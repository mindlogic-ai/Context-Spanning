"""ASR adapter for MoshiCP.

Backend priority:
  1. faster_whisper  (WhisperModel, CPU int8) — used when available
  2. openai-whisper  (whisper.load_model)
  3. stub            — returns "" with a printed warning
"""

from __future__ import annotations

import importlib.util
import os
import re
import warnings
from typing import Optional

import numpy as np
from contextspan.duetaspan.common import paths

# Process-wide loaded ASR model (shared across all ASR() instances — avoids reloading
# large weights / re-allocating VRAM on every <ret> fire).
_SHARED: dict = {"model": None, "backend": None}


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
    """Lazy-loading ASR adapter.

    Parameters
    ----------
    device:
        Target device for the model ("cpu" recommended; GPU is not claimed
        until *after* lazy load).
    model:
        Optional pre-constructed model object.  When supplied the backend
        detection step is skipped and this object is used directly.
    """

    _FASTER_WHISPER_SR = 16_000
    _WHISPER_SR = 16_000

    def __init__(self, device: str = "cpu", model=None) -> None:
        self.device = device
        self._model = model          # may be None until first transcribe()
        self._backend: Optional[str] = None  # "faster_whisper" | "whisper" | "stub"

        if model is not None:
            # Caller supplied a ready model — detect which kind it is
            self._backend = ("faster_whisper"
                             if importlib.util.find_spec("faster_whisper") else "whisper")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Detect available backend and load the model (called once).

        Model is a process-wide singleton (_SHARED) so repeated ASR() instances —
        e.g. one per <ret> fire — reuse a single loaded WhisperModel instead of
        reloading the (large) weights and re-allocating VRAM each time.
        """
        # 0. reuse already-loaded shared model
        if _SHARED["backend"] is not None:
            self._model, self._backend = _SHARED["model"], _SHARED["backend"]
            return

        # 0b. Qwen3-ASR vLLM HTTP endpoint (fast ~30-60ms, more accurate) — preferred for serving.
        #     Isolated vllm env; we just POST audio. Falls through to faster-whisper if unreachable.
        if os.environ.get("MOSHICP_ASR_BACKEND", "qwen").lower() in ("qwen", "qwen_http", "qwen3"):
            url = os.environ.get("MOSHICP_ASR_URL", "http://localhost:8990/transcribe")
            # If MOSHICP_ASR_URL is set explicitly, trust it unconditionally (2026-08-03
            # incident). The 1-second connect probe times out whenever the box is momentarily
            # busy (engine loading, etc.), and the process is then demoted to the stub so
            # **every transcript is '' for the rest of its lifetime** — the cause of two FDB v3
            # shards coming out entirely call-less (calls=[]). An explicit env var is the
            # operator's intent, so no demotion; transient failures are absorbed by the
            # per-request timeout (10s) and the except -> "" path.
            if os.environ.get("MOSHICP_ASR_URL", "").strip():
                self._model, self._backend = url, "qwen_http"
                _SHARED["model"], _SHARED["backend"] = url, "qwen_http"
                print(f"[ASR] Qwen3-ASR vLLM endpoint {url} (explicit env; probe skipped)", flush=True)
                return
            try:
                import socket
                from urllib.parse import urlparse
                u = urlparse(url)
                socket.create_connection((u.hostname, u.port or 80), timeout=1.0).close()
                self._model, self._backend = url, "qwen_http"
                _SHARED["model"], _SHARED["backend"] = url, "qwen_http"
                print(f"[ASR] Qwen3-ASR vLLM endpoint {url}", flush=True)
                return
            except Exception as e:
                warnings.warn(f"[ASR] Qwen endpoint unreachable ({e!r}); using faster-whisper",
                              RuntimeWarning, stacklevel=2)

        # 1. faster_whisper — large-v3-turbo on GPU (float16) for quality+low latency,
        #    CPU int8 fallback. Override via MOSHICP_ASR_MODEL / MOSHICP_ASR_DEVICE.
        try:
            from faster_whisper import WhisperModel
            name = os.environ.get("MOSHICP_ASR_MODEL", "large-v3")
            dev = os.environ.get("MOSHICP_ASR_DEVICE", "")
            tried = []
            if dev:
                tried.append((dev, "float16" if dev == "cuda" else "int8"))
            else:
                try:
                    import ctranslate2
                    if ctranslate2.get_cuda_device_count() > 0:
                        tried.append(("cuda", "float16"))
                except Exception:
                    pass
                tried.append(("cpu", "int8"))
            last = None
            for d, ct in tried:
                try:
                    self._model = WhisperModel(name, device=d, compute_type=ct)
                    self._backend = "faster_whisper"
                    _SHARED["model"], _SHARED["backend"] = self._model, self._backend
                    print(f"[ASR] faster-whisper {name!r} on {d}/{ct}", flush=True)
                    return
                except Exception as e:  # GPU libs missing / OOM → try next
                    last = e
            if last:
                warnings.warn(f"[ASR] faster-whisper load failed: {last!r}", RuntimeWarning, stacklevel=2)
        except ImportError:
            pass

        # 2. openai-whisper
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model("base.en", device=self.device)
            self._backend = "whisper"
            _SHARED["model"], _SHARED["backend"] = self._model, self._backend
            return
        except ImportError:
            pass

        # 3. stub
        warnings.warn(
            "[ASR] No ASR backend available (install faster-whisper or "
            "openai-whisper).  transcribe() will return empty strings.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._backend = "stub"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 24_000) -> str:
        """Transcribe a mono float32 PCM array and return lowercased text.

        Parameters
        ----------
        pcm:
            1-D float32 numpy array (mono audio samples).
        sample_rate:
            Sample rate of *pcm* in Hz (default 24 000 Hz).

        Returns
        -------
        str
            Recognised text, lowercased and stripped.  Empty string if no
            backend is available or the audio is silent/empty.
        """
        if self._model is None and self._backend is None:
            self._load()

        if self._backend == "stub":
            return ""

        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1:
            raise ValueError(f"pcm must be 1-D, got shape {pcm.shape}")

        if self._backend == "qwen_http":
            import io
            import requests
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            # Language may be forced; a context/hotword bias prompt may NOT. Biasing the
            # decoder toward expected proper nouns makes it emit them from silence, which is
            # exactly the hallucination the QA gates exist to catch. Older servers ignore the
            # field they do not know, so this stays forward-compatible.
            data = {}
            _lang = os.environ.get("MOSHICP_ASR_LANGUAGE", "").strip()
            if _lang:
                data["language"] = _lang
            # vLLM's `qwen-asr-serve` speaks the OpenAI audio API (`/v1/audio/transcriptions`): it
            # needs a `model` field and prefixes the text with "language English<asr_text>". Same
            # weights, ~2x faster per request and it batches, so the final transcript of a question no
            # longer queues behind the partial one (measured 0.45 s -> 0.13 s median on the bench box).
            if "/v1/audio/transcriptions" in self._model:
                data["model"] = os.environ.get("MOSHICP_ASR_MODEL", "qwen3-asr")
            try:
                r = requests.post(self._model, files={"file": ("a.wav", buf, "audio/wav")},
                                  data=data or None, timeout=10)
                text = (r.json().get("text") or "").strip()
                text = re.sub(r"^language\s+\w+<asr_text>\s*", "", text)
                # The default keeps the previous lowercase contract. With
                # MOSHICP_ASR_KEEP_CASE=1 the original case is preserved — this carries the case
                # information of spelled-out IDs ("A B C one two three") all the way to the
                # router and argument filling (an opt-in repair of the path where lowercasing
                # destroyed what the ASR actually heard).
                if os.environ.get("MOSHICP_ASR_KEEP_CASE", "").strip().lower() in ("1", "true", "on"):
                    return text
                return text.lower()
            except Exception:
                return ""

        if self._backend == "faster_whisper":
            audio = _resample(pcm, sample_rate, self._FASTER_WHISPER_SR)
            segments, _info = self._model.transcribe(
                audio,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip().lower()
            return text

        if self._backend == "whisper":
            audio = _resample(pcm, sample_rate, self._WHISPER_SR)
            result = self._model.transcribe(audio, fp16=False, language="en")
            return result["text"].strip().lower()

        return ""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import soundfile as sf

    wav_files = sorted(
        glob.glob(
            f"{paths.DATA}/moshicp/audio/dialogue/*.wav"
        )
    )
    if not wav_files:
        raise FileNotFoundError(
            "No .wav files found in datasets/moshicp/audio/dialogue/"
        )

    wav_path = wav_files[0]
    print(f"Loading: {wav_path}")

    stereo, sr = sf.read(wav_path, dtype="float32")
    # stereo shape: (samples, 2)  — channel 0 = agent (L), channel 1 = user (R)
    user_pcm = stereo[:, 1]
    print(f"Audio: {stereo.shape}, sr={sr}, user channel shape={user_pcm.shape}")

    asr = ASR(device="cpu")
    transcript = asr.transcribe(user_pcm, sample_rate=sr)

    print(f"\nASR backend : {asr._backend}")
    print(f"Transcript  : {transcript!r}")
