#!/usr/bin/env python3
"""Full-Duplex-Bench v3 turn-taking metrics (paper Table 2: take-turn, latency, interruption, filler) for our runs.

The benchmark's own pipeline (run_tool_benchmark.py) transcribes the user input and the agent output with
nvidia/parakeet-tdt-0.6b-v2 word timestamps, stores `user_speech_end_rel` (end of the first turn: the word before
the first gap > 2 s, else the last word) and `asr_chunks`, and analyze_tool_latency.py then computes the latencies
and asks gpt-4o to separate the filler sentence from the key information. This script adds the same fields to our
result_<provider>.json files (from output_<provider>.wav written by v3_run.py), runs the benchmark's
analyze_tool_latency.py unchanged, and reports the four Table 2 numbers:

  take-turn     scenarios with any agent speech / all scenarios
  interruption  turn-taken scenarios whose first agent word starts before the user finished / turn-taken scenarios
  latency       mean task completion latency over turn-taken, non-interrupted scenarios (seconds)
  filler        scenarios with a filler sentence / turn-taken, non-interrupted scenarios
(the denominators reproduce the paper's GPT-Realtime and Gemini Live rows exactly: 13/96, 14/83, 15/78, 20/63).

Needs NeMo (parakeet) and OPENAI_API_KEY (gpt-4o, as in the benchmark).

Usage: python benchmark/fdb/v3_turn_taking.py --data-root DIR --provider ours --fdb-v3 PATH_TO_BENCHMARK [--device cuda]
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile

ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def asr_chunks(model, wav_path):
    import soundfile as sf, numpy as np, librosa
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, x, 16000)
        out = model.transcribe([tmp.name], timestamps=True)
    os.unlink(tmp.name)
    if not out:
        return "", []
    res = out[0]
    chunks = []
    if hasattr(res, "timestamp") and "word" in res.timestamp:
        chunks = [{"text": w["word"], "timestamp": [w["start"], w["end"]]} for w in res.timestamp["word"]]
    return " ".join(c["text"] for c in chunks), chunks


def user_speech_end(chunks):
    """The benchmark's rule: end of the word before the first gap > 2 s, else the last word's end."""
    if not chunks:
        return 0
    for cur, nxt in zip(chunks, chunks[1:]):
        if nxt["timestamp"][0] - cur["timestamp"][1] > 2.0:
            return cur["timestamp"][1]
    return chunks[-1]["timestamp"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--provider", default="ours")
    ap.add_argument("--fdb-v3", required=True, help="checkout of the Full-Duplex-Bench v3 code (analyze_tool_latency.py)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    results = sorted(glob.glob(f"{a.data_root}/**/result_{a.provider}.json", recursive=True))
    todo = [r for r in results if os.path.exists(os.path.join(os.path.dirname(r), f"output_{a.provider}.wav"))]
    missing = len(results) - len(todo)
    print(f"[v3tt] {len(results)} results, {len(todo)} with agent audio, {missing} without (re-run v3_run.py to write it)", flush=True)
    if not todo:
        sys.exit(1)

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    if a.device == "cuda":
        model = model.cuda()
    for r in todo:
        d = os.path.dirname(r)
        data = json.load(open(r))
        if "asr_chunks" in data and "user_speech_end_rel" in data:
            continue
        backup = r.replace(".json", ".pre_turn_taking.json")
        if not os.path.exists(backup):
            shutil.copy(r, backup)
        _, in_chunks = asr_chunks(model, os.path.join(d, "input.wav"))
        out_text, out_chunks = asr_chunks(model, os.path.join(d, f"output_{a.provider}.wav"))
        data["input_asr_chunks"] = in_chunks
        data["user_speech_end_rel"] = user_speech_end(in_chunks)
        data["asr_chunks"] = out_chunks
        data["output_asr_text"] = out_text          # the model's own text stream stays in `transcript`
        json.dump(data, open(r, "w"), ensure_ascii=False, indent=2)
        print(f"[v3tt] {os.path.basename(d)} user_end={data['user_speech_end_rel']:.2f} agent_words={len(out_chunks)}", flush=True)

    report = os.path.abspath(f"{a.data_root}/{a.provider}_latency_report.json")
    subprocess.run([sys.executable, "analyze_tool_latency.py", "--results-dir", os.path.abspath(a.data_root),
                    "--provider", a.provider, "--output", report], cwd=a.fdb_v3, check=True)

    per = []
    for r in todo:
        m = os.path.join(os.path.dirname(r), f"latency_tool_analysis_{a.provider}.json")
        if os.path.exists(m):
            per.append(json.load(open(m)))
    n = len(per)
    taken = [m for m in per if m.get("turn_take_success")]
    interrupted = [m for m in taken if (m.get("first_response_latency_s") is not None and m["first_response_latency_s"] < 0)]
    valid = [m for m in taken if m.get("first_response_latency_s") is not None and m["first_response_latency_s"] >= 0]
    task = [m["task_completion_latency_s"] for m in valid if m.get("task_completion_latency_s") is not None]
    filler = [m for m in valid if m.get("filler_sentence")]
    summary = {
        "scenarios": n,
        "take_turn_rate_pct": round(100 * len(taken) / n, 1) if n else None,
        "latency_s": round(sum(task) / len(task), 2) if task else None,
        "interruption_rate_pct": round(100 * len(interrupted) / len(taken), 1) if taken else None,
        "filler_rate_pct": round(100 * len(filler) / len(valid), 1) if valid else None,
        "counts": {"turn_taken": len(taken), "interrupted": len(interrupted), "valid": len(valid), "with_filler": len(filler),
                   "task_latency_n": len(task)},
    }
    out = f"{a.data_root}/{a.provider}_turn_taking_report.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
