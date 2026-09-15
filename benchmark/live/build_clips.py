"""TTS the benchmark questions into the clips the fake microphone plays.

    python -m benchmark.live.build_clips benchmark/live/cases.json <clips_dir>

Each clip is lead-in silence, the question, then a long tail of silence so the session has room to
answer while the fake device keeps playing. macOS `say` is the default because it is deterministic
and needs nothing installed; `CS_TTS` overrides the command (it receives TEXT and OUT_AIFF).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import wave

LEAD_S, TAIL_S, SR = 1.5, 26.0, 24000


def build(cases_path: str, out_dir: str) -> int:
    cases = json.load(open(cases_path))
    os.makedirs(out_dir, exist_ok=True)
    if not shutil.which("say") or not shutil.which("afconvert"):
        sys.exit("needs macOS `say` + `afconvert`; set CS_TTS to use another engine")
    for c in cases:
        aiff, raw, out = (os.path.join(out_dir, c["id"] + s) for s in (".aiff", ".raw.wav", ".wav"))
        subprocess.run(["say", "-v", os.environ.get("CS_TTS_VOICE", "Samantha"),
                        "-o", aiff, c["question"]], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", aiff, raw], check=True)
        with wave.open(raw, "rb") as r:
            sr, data = r.getframerate(), r.readframes(r.getnframes())
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(b"\x00\x00" * int(sr * LEAD_S) + data + b"\x00\x00" * int(sr * TAIL_S))
        os.remove(aiff); os.remove(raw)
    shutil.copy(cases_path, os.path.join(out_dir, "cases.json"))
    print(f"built {len(cases)} clips in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(build(sys.argv[1], sys.argv[2]))
