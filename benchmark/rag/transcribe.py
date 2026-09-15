"""OpenAudioBench-protocol response transcription: whisper-large-v3 over a run dir.

    python -m benchmark.rag.transcribe <run_dir>          (needs `transformers`, a GPU)

Writes `hyp_whisper` into each item's run_events.json and <run_dir>/whisper_hyps.json; benchmark.rag.score
uses it when present. The decoded array is passed to the pipeline (not the file path): the pipeline's
own long-form loader dropped the whole answer segment on clips with a long pause after the greeting.
"""
import glob
import json
import os
import sys

import soundfile as sf
import torch
from transformers import pipeline


def main(run):
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                    device="cuda:0", model_kwargs={"attn_implementation": "sdpa"})
    out = {}
    dirs = sorted(d for d in glob.glob(f"{run}/*") if os.path.exists(f"{d}/output.wav"))
    for i, d in enumerate(dirs):
        x, sr = sf.read(f"{d}/output.wav", dtype="float32")
        if x.ndim == 2:
            x = x.mean(1)
        hyp = pipe({"raw": x, "sampling_rate": sr}, return_timestamps=True,
                   generate_kwargs={"language": "english", "task": "transcribe"})["text"].strip()
        ev = json.load(open(f"{d}/run_events.json")); ev["hyp_whisper"] = hyp
        json.dump(ev, open(f"{d}/run_events.json", "w"), ensure_ascii=False)
        out[os.path.basename(d)] = hyp
        if (i + 1) % 25 == 0:
            print(f"[whisper] {i + 1}/{len(dirs)}", flush=True)
    json.dump(out, open(f"{run}/whisper_hyps.json", "w"), ensure_ascii=False)
    print(f"[whisper] done {len(dirs)}")


if __name__ == "__main__":
    main(sys.argv[1])
