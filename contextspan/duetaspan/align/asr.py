"""ASR adapter for MoshiCP.

Backend priority:
  1. faster_whisper  (WhisperModel, CPU int8) — used when available
  2. openai-whisper  (whisper.load_model)
  3. stub            — returns "" with a printed warning
"""

from __future__ import annotations

import importlib.util
import os
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
            # MOSHICP_ASR_URL 이 명시돼 있으면 무조건 신뢰한다 (2026-08-03 실사고).
            # 1초짜리 연결 프로브는 박스가 바쁜 순간(엔진 로딩 등) 타임아웃하고, 그러면
            # 이 프로세스는 stub 로 강등돼 **수명 내내 전사가 '' 가 된다** — FDB v3 두 샤드가
            # 통째로 무콜(calls=[])이 된 원인. 명시 env = 운영자 의도이므로 강등 금지;
            # 일시 장애는 요청별 timeout(10s)과 except → "" 가 흡수한다.
            if os.environ.get("MOSHICP_ASR_URL", "").strip():
                self._model, self._backend = url, "qwen_http"
                _SHARED["model"], _SHARED["backend"] = url, "qwen_http"
                print(f"[ASR] Qwen3-ASR vLLM endpoint {url} (명시 env — 프로브 생략)", flush=True)
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
            try:
                r = requests.post(self._model, files={"file": ("a.wav", buf, "audio/wav")},
                                  data=data or None, timeout=10)
                text = (r.json().get("text") or "").strip()
                # 기본은 종전과 동일한 lowercase 계약. MOSHICP_ASR_KEEP_CASE=1 이면 원문 케이스
                # 유지 — 철자 ID("A B C one two three")의 케이스 정보를 라우터/인자채움까지
                # 보존한다(소문자화가 ASR이 실제 들은 정보를 파괴하던 경로의 opt-in 수리).
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
