"""MoshiRAG latency metrics (TTFAT / KD / E2EKD) for a run dir.

    python -m eval.rag.latency <run_dir>       (needs nemo_toolkit[asr] for parakeet word timestamps, a GPU)

TTFAT = question end -> first voiced agent audio; KD = response onset -> first mention of the answer
keyword; E2EKD = TTFAT + KD. Keyword = MoshiRAG Table 17 extraction (LLM judge) over the transcript;
its onset from nvidia/parakeet-tdt-0.6b-v2 word timestamps (the paper's protocol).
Writes <run_dir>/latency_report.json.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import requests
import soundfile as sf

KW_PROMPT = ("You are helping evaluate a question answering model.\n"
             "Identify the single keyword or short phrase in the model answer that directly expresses any of the "
             "answer aliases. If the model answer does not directly express any of the answer aliases, return the "
             "keyword/phrase that the model intends to answer the question. Respond with that keyword/phrase only.\n"
             "Example:\nQuestion: Give me a capital of an European country?\n"
             "Model answer: Berlin is the capital of Germany.\n"
             'Valid answer aliases: ["Paris", "Madrid", "Budapest", "Lisbon"]\nResponse: Berlin\n'
             'Input:\nQuestion: {question}\nModel answer: {answer}\nValid answer aliases: ["{gold}"]\nResponse:')


def voiced_onset(path, t0, thr=-38):
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(1)
    hop = int(0.05 * sr); n = len(x) // hop
    r = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    v = 20 * np.log10(r) > thr
    for i in range(int(t0 / 0.05), n):
        if v[i]:
            return i * 0.05
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_LLM_URL", "http://localhost:8004/v1/chat/completions"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_LLM_MODEL", "google/gemma-3-27b-it"))
    a = ap.parse_args(argv)
    import nemo.collections.asr as nemo_asr
    asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

    def keyword(question, answer_text, gold):
        try:
            r = requests.post(a.judge_url, json={"model": a.judge_model, "temperature": 0, "max_tokens": 24,
                                                 "messages": [{"role": "user", "content": KW_PROMPT.format(
                                                     question=question, answer=answer_text, gold=gold)}]}, timeout=60)
            return r.json()["choices"][0]["message"]["content"].strip().strip('"')
        except Exception:
            return str(gold)

    def keyword_time(path, kw):
        h = asr.transcribe([path], timestamps=True)[0]
        words = getattr(h, "timestamp", {}).get("word", []) if hasattr(h, "timestamp") else []
        kws = [w for w in re.findall(r"[A-Za-z0-9']+", str(kw).lower()) if len(w) > 2] or re.findall(r"[A-Za-z0-9']+", str(kw).lower())
        for w in words:
            tok = re.sub(r"[^a-z0-9']", "", str(w.get("word", "")).lower())
            if tok and any(k.startswith(tok) or tok.startswith(k) for k in kws):
                return float(w.get("start", w.get("start_offset", 0)))
        return None

    rows = []
    for d in sorted(glob.glob(f"{a.run_dir}/*")):
        if not (os.path.exists(f"{d}/output.wav") and os.path.exists(f"{d}/meta.json")):
            continue
        m = json.load(open(f"{d}/meta.json")); ev = json.load(open(f"{d}/run_events.json"))
        qe = float(m["question_end_s"]); on = voiced_onset(f"{d}/output.wav", qe)
        kt = keyword_time(f"{d}/output.wav", keyword(m.get("question", ""), ev.get("hyp_whisper") or ev.get("transcript", ""), m.get("answer", "")))
        rows.append({"id": os.path.basename(d), "q_end": qe,
                     "ttfat": round(on - qe, 2) if on is not None else None,
                     "kd": round(kt - on, 2) if (kt is not None and on is not None and kt >= on) else None,
                     "e2ekd": round(kt - qe, 2) if (kt is not None and kt >= qe) else None})
    avg = lambda k: (round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3) if any(r[k] is not None for r in rows) else None)
    summary = {"n": len(rows), "TTFAT": avg("ttfat"), "KD": avg("kd"), "E2EKD": avg("e2ekd"),
               "kw_found_rate": round(sum(1 for r in rows if r["e2ekd"] is not None) / max(len(rows), 1), 3)}
    json.dump({"summary": summary, "rows": rows}, open(f"{a.run_dir}/latency_report.json", "w"), indent=1)
    print("[lat] SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
