"""External backends: an LLM that writes the reference (MoshiRAG-style) and an ASR endpoint.

Both talk to OpenAI-compatible HTTP servers (e.g. vLLM), so any model can sit behind them.
"""
import io
import os
import numpy as np
import requests
import soundfile as sf

NO_INFO = "(no information found)"
REF_SYSTEM = ("You are the knowledge backend of a spoken assistant. Given the user's question, "
              "reply with ONE factual spoken sentence of at most 20 words that answers it (only the fact asked, no background). If you do not "
              f"know, reply exactly {NO_INFO}. Plain text, no markdown.")


class LLMReferenceBackend:
    def __init__(self, url=None, model=None, api_key=None, timeout=30):
        self.url = url or os.environ.get("CS_LLM_URL", "http://localhost:8000/v1/chat/completions")
        self.model = model or os.environ.get("CS_LLM_MODEL", "google/gemma-3-27b-it")
        self.key = api_key or os.environ.get("CS_LLM_API_KEY", "")     # hosted APIs (e.g. GPT-Luna)
        self.timeout = timeout

    def retrieve(self, question: str) -> str:
        body = {"model": self.model, "messages": [{"role": "system", "content": REF_SYSTEM},
                                                  {"role": "user", "content": question}]}
        if os.environ.get("CS_LLM_REASONING", "auto") == "1" or (
                os.environ.get("CS_LLM_REASONING", "auto") == "auto" and any(k in self.model.lower() for k in ("gpt-5", "luna", "o1", "o3", "o4"))):
            body["max_completion_tokens"] = 256          # reasoning models: no temperature / max_tokens
        else:
            body.update(temperature=0, max_tokens=96)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else None
        r = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
        return (r.json()["choices"][0]["message"]["content"] or "").strip() or NO_INFO


class ASR:
    """POST wav to an OpenAI-style /v1/audio/transcriptions endpoint (e.g. vLLM qwen-asr-serve)."""

    def __init__(self, url=None, model=None, timeout=10):
        self.url = url or os.environ.get("CS_ASR_URL", "http://localhost:8901/v1/audio/transcriptions")
        self.model = model or os.environ.get("CS_ASR_MODEL", "qwen3-asr")
        self.timeout = timeout

    def transcribe(self, pcm: np.ndarray, sr: int) -> str:
        buf = io.BytesIO()
        sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        try:
            r = requests.post(self.url, files={"file": ("a.wav", buf, "audio/wav")},
                              data={"model": self.model}, timeout=self.timeout)
            text = (r.json().get("text") or "").strip()
            return text.split("<asr_text>", 1)[1].strip() if "<asr_text>" in text else text
        except Exception:
            return ""
