# -*- coding: utf-8 -*-
"""Qwen3-ASR HTTP server (:8990) — the endpoint contextspan.duetaspan.align.asr.ASR prefers.

Runs Qwen3-ASR-1.7B via the transformers backend (no vLLM engine, ~4GB) so it can share
GPU2 with the gemma router/RAG server. POST /transcribe with a wav file (multipart `file`
field, as asr.py sends, or a raw wav body) -> {"text": ...}. GET /health -> ok.

Launch: CUDA_VISIBLE_DEVICES=<gpu> python -m contextspan.duetaspan.runtime.asr_server
"""
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf
import torch

from contextspan.duetaspan.common import paths
MODEL_DIR = os.environ.get("QWEN_ASR_DIR", f"{paths.MODELS}/Qwen3-ASR-1.7B")
PORT = int(os.environ.get("ASR_PORT", "8990"))
# No context/hotword bias is accepted or applied (user decree, 2026-08-25): a biased
# transcript invents the words it was primed with, which is exactly the failure this
# endpoint exists to measure. `language` is the only conditioning left.

print(f"[asr_server] loading {MODEL_DIR} (transformers backend, bf16)", flush=True)
from qwen_asr import Qwen3ASRModel  # noqa: E402  (heavy import after banner)
MODEL = Qwen3ASRModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0")
print(f"[asr_server] ready on :{PORT}", flush=True)

# Qwen3-ASR wants a language *name* ("Korean"), and the client passes MOSHICP_ASR_LANGUAGE
# through untouched, so the natural setting "ko" was rejected on every request and the
# client turned that into an empty transcript with no error in the runtime log.
_LANG_NAMES = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
               "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
               "it": "Italian", "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
               "ja": "Japanese"}


def _norm_lang(value):
    v = (value or "").strip()
    if not v:
        return "English"
    key = v.lower().replace("_", "-").split("-")[0]
    if key in _LANG_NAMES:
        return _LANG_NAMES[key]
    return v[0].upper() + v[1:].lower()        # "korean", "KOREAN" -> "Korean"



def _extract_wav_bytes(headers, body: bytes):
    """(wav_bytes, fields) — the audio part plus the text fields (`language`)."""
    ctype = headers.get("Content-Type", "")
    fields = {}
    wav = body
    if ctype.startswith("multipart/"):
        import email
        msg = email.message_from_bytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
        for part in msg.walk():
            if part.get_filename() or part.get_content_maintype() == "audio":
                wav = part.get_payload(decode=True)
            else:
                name = part.get_param("name", header="Content-Disposition")
                if name and part.get_content_maintype() == "text" or name == "language":
                    try:
                        fields[str(name)] = (part.get_payload(decode=True) or b"").decode(
                            "utf-8", "ignore").strip()
                    except Exception:
                        pass
    return wav, fields


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"ok": True})

    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            wav, fields = _extract_wav_bytes(self.headers, body)
            pcm, sr = sf.read(io.BytesIO(wav), dtype="float32")
            if pcm.ndim == 2:
                pcm = pcm.mean(axis=1)
            if len(pcm) < sr * 0.2:                       # too short for ASR
                return self._send(200, {"text": ""})
            language = _norm_lang(fields.get("language"))
            res = MODEL.transcribe((np.ascontiguousarray(pcm), int(sr)), language=language)
            self._send(200, {"text": (res[0].text or "").strip()})
        except Exception as e:  # keep the server alive on any bad request
            self._send(500, {"text": "", "error": str(e)})


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
