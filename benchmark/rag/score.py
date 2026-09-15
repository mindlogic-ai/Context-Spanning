"""RAG-suite scorer (MoshiRAG Table 1 protocol): ref. / resp. accuracy split + latency.

    python -m benchmark.rag.score <run_dir> [--judge-url URL] [--judge-model M] [--asr-url URL] [--math]

resp acc  = LLM judge (MoshiRAG Table 16 prompt, verbatim): does the transcript of what the agent said
            contain the gold answer?  The transcript is `hyp_whisper` from benchmark.rag.transcribe when
            present (OpenAudioBench protocol: whisper-large-v3), else the ASR endpoint.
ref acc   = literal containment of the gold string in the arrived span text, else the same judge
ret latency = t_ret - question_end ; inj latency = t_inj - t_ret ; answer onset = first voiced agent frame
Writes <run_dir>/rag_report.json and prints the summary.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re

import numpy as np
import requests
import soundfile as sf

JUDGE_SYS = ("You are a professional QA evaluation expert. You need to assess whether the model's answer is "
             "correct based on the standard answer.")
JUDGE_BODY = ("## Scoring Criteria\n"
              "Correct: The answer matches or is equivalent to the standard answer, or contains the same core concept.\n"
              "Incorrect: The answer is wrong or irrelevant to the question\n\n"
              "## Evaluation Guidelines\n"
              "1. The expression of answers can be flexible, not requiring exact matches. For example:\n"
              "   - Numbers can be expressed in either Arabic numerals or words\n"
              "   - Differences in punctuation or simple spelling mistakes can be ignored\n"
              "2. Focus on whether the core meaning of the answer is correct\n\n"
              "## Output Format\n"
              'Provide the reasoning for your score, then generate the result in "[]" format and make sure it contains '
              '"the score is [Correct]" or "the score is [Incorrect]".\n\n')
MATH_PROMPT = ("Question: {question}\nGold answer: {gold}\nModel answer: {answer}\n"
               "Is the model answer correct? Reply Yes or No.")


def gold_view(gold):
    g = str(gold)
    if ";" in g and " | aliases:" not in g:
        parts = [x.strip() for x in g.split(";") if x.strip()]
        if len(parts) > 1:
            return "any ONE of these is a correct answer: " + " / ".join(parts[:12])
    return g


class Judge:
    def __init__(self, url, model, math=False):
        self.url, self.model, self.math = url, model, math

    def __call__(self, question, gold, text, kind):
        gold = gold_view(gold)
        if not text or not text.strip():
            return 0
        if self.math:
            try:
                r = requests.post(self.url, json={"model": self.model, "temperature": 0, "max_tokens": 4,
                                                  "messages": [{"role": "user", "content": MATH_PROMPT.format(
                                                      question=question, gold=gold, answer=text)}]}, timeout=120)
                return int(r.json()["choices"][0]["message"]["content"].strip().lower().startswith("yes"))
            except Exception:
                return 0
        if kind == "ref":
            n = lambda x: " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split()) + " "
            alts = [x for x in re.split(r"\s*(?:;|\|\s*aliases:|/)\s*", str(gold)) if x.strip()]
            if any(n(x) in n(text) for x in alts if len(n(x).strip()) >= 2):
                return 1
        tgt = "Model's Answer" if kind == "resp" else "Model's Answer (retrieved reference)"
        um = JUDGE_BODY + f"## Question:\n{question}\n## Standard Answer:\n{gold}\n## {tgt}:\n{text}"
        try:
            r = requests.post(self.url, json={"model": self.model, "temperature": 0, "max_tokens": 300,
                                              "messages": [{"role": "system", "content": JUDGE_SYS},
                                                           {"role": "user", "content": um}]}, timeout=90)
            return 1 if "[correct]" in r.json()["choices"][0]["message"]["content"].lower() else 0
        except Exception:
            return 1 if str(gold).lower() in text.lower() else 0


def first_voiced_after(path, t0, thr=-38):
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(1)
    hop = int(0.05 * sr); n = len(x) // hop
    r = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    v = 20 * np.log10(r) > thr
    for i in range(int(t0 / 0.05), n):
        if v[i]:
            return i * 0.05 - t0
    return None


def asr_file(url, path):
    data = {"model": os.environ.get("MOSHICP_ASR_MODEL", "qwen3-asr")} if "/v1/audio/transcriptions" in url else None
    with open(path, "rb") as f:
        r = requests.post(url, files={"file": ("a.wav", f, "audio/wav")}, data=data, timeout=120)
    j = r.json()
    text = j.get("text", j.get("transcript", "")) if isinstance(j, dict) else str(j)
    return re.sub(r"^language\s+\w+<asr_text>\s*", "", text.strip())   # vLLM qwen-asr-serve prefix


def score_moshirag(a, dirs):
    """MoshiRAG-isomorphic scoring (see benchmark/rag/moshirag.py). The judged answer is the model's text
    stream with the <ret>/<span> markers removed; the reference is the injected span text. A judge
    returns None on empty text and -1 when unparseable; both are excluded from the averages, as in
    moshi-rag's evaluate/score.py. P(resp|ref) and the timing fields are kept as extra columns."""
    from benchmark.rag.moshirag import MoshiRagJudge, strip_tags
    mode = a.mode or os.path.basename(os.path.normpath(a.run_dir))
    judge = MoshiRagJudge(mode)

    def one(d):
        m = json.load(open(f"{d}/meta.json")); ev = json.load(open(f"{d}/run_events.json"))
        events = ev.get("events", []); spans = [s for s in events if s.get("inject")]
        text = strip_tags(ev.get("transcript", ""))
        ref_text = " ".join(str(s.get("inject") or "") for s in spans)
        q, gold, qe = m["question"], m["answer"], m["question_end_s"]
        row = {"id": os.path.basename(d), "resp": judge(q, gold, text), "ref": judge(q, gold, ref_text) if spans else None,
               "n_span": len(spans), "n_ret": len(events), "text": text[:300], "hyp": (ev.get("hyp_whisper") or "")[:300],
               "backend_status": events[-1].get("backend_status") if events else None}
        if spans:
            s0 = spans[0]
            if s0.get("t") is not None:
                row["ret_lat_s"] = round(float(s0["t"]) - qe, 2)
            if s0.get("t_inj") is not None and s0.get("t") is not None:
                row["inj_lat_s"] = round(float(s0["t_inj"]) - float(s0["t"]), 2)
        return row

    with cf.ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(one, dirs))
    n = max(len(rows), 1)
    judged = lambda k: [r[k] for r in rows if r.get(k) is not None and r[k] >= 0]
    avg = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    both = [r for r in rows if r.get("ref") == 1 and r.get("resp") is not None and r["resp"] >= 0]
    lat = lambda k: avg([r[k] for r in rows if r.get(k) is not None])
    summary = {"run": a.run_dir, "protocol": "moshirag", "n": len(rows),
               "resp_acc": avg(judged("resp")), "n_resp_judged": len(judged("resp")),
               "ref_acc": avg(judged("ref")), "n_ref_judged": len(judged("ref")),
               "P(resp|ref)": avg([r["resp"] for r in both]), "n_ref_correct": len(both),
               "resp_acc_all": round(sum(1 for r in rows if r.get("resp") == 1) / n, 4),
               "ref_acc_all": round(sum(1 for r in rows if r.get("ref") == 1) / n, 4),
               "span_rate": round(sum(1 for r in rows if r["n_span"]) / n, 4), "ret_rate": round(sum(1 for r in rows if r["n_ret"]) / n, 4),
               "backend_timeout": sum(1 for r in rows if r.get("backend_status") == "timeout"),
               "ret_lat_s": lat("ret_lat_s"), "inj_lat_s": lat("inj_lat_s")}
    json.dump({"summary": summary, "rows": rows}, open(f"{a.run_dir}/rag_report_moshirag.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_LLM_URL", "http://localhost:8004/v1/chat/completions"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_LLM_MODEL", "google/gemma-3-27b-it"))
    ap.add_argument("--asr-url", default=os.environ.get("MOSHICP_ASR_URL", "http://localhost:8990/transcribe"))
    ap.add_argument("--math", action="store_true", help="MoshiRAG math judge (Yes/No)")
    ap.add_argument("--protocol", default="ours", choices=["ours", "moshirag"],
                    help="moshirag: judge the model's own text stream with the moshi-rag judges "
                         "(benchmark/rag/moshirag.py) and average as moshi-rag's score.py does")
    ap.add_argument("--mode", default=None, help="dataset name for --protocol moshirag (default: run_dir basename)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    judge = Judge(a.judge_url, a.judge_model, a.math)
    dirs = sorted(d for d in glob.glob(f"{a.run_dir}/*") if os.path.exists(f"{d}/output.wav"))
    if a.protocol == "moshirag":
        return score_moshirag(a, dirs)

    def one(d):
        m = json.load(open(f"{d}/meta.json")); ev = json.load(open(f"{d}/run_events.json"))
        spans = [s for s in ev.get("events", []) if s.get("inject")]
        hyp = ev.get("hyp_whisper") or ""
        if not hyp:
            try:
                hyp = asr_file(a.asr_url, f"{d}/output.wav")
            except Exception as e:
                print(f"[score] ASR fail {d}: {e}", flush=True)
        q, gold, qe = m["question"], m["answer"], m["question_end_s"]
        row = {"id": os.path.basename(d), "resp": judge(q, gold, hyp, "resp"),
               "ref": judge(q, gold, " ".join(str(s.get("inject") or "") for s in spans), "ref") if spans else 0,
               "n_span": len(spans), "n_ret": len(ev.get("events", [])), "hyp": hyp[:300]}
        if spans:
            s0 = spans[0]
            if s0.get("t") is not None:
                row["ret_lat_s"] = round(float(s0["t"]) - qe, 2)
            if s0.get("t_inj") is not None and s0.get("t") is not None:
                row["inj_lat_s"] = round(float(s0["t_inj"]) - float(s0["t"]), 2)
        on = first_voiced_after(f"{d}/output.wav", qe)
        if on is not None:
            row["answer_onset_s"] = round(on, 2)
        return row

    with cf.ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(one, dirs))
    n = max(len(rows), 1)
    avg = lambda k: (round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 4)
                     if any(r.get(k) is not None for r in rows) else None)
    summary = {"run": a.run_dir, "n": len(rows),
               "resp_acc": round(sum(r["resp"] for r in rows) / n, 4), "ref_acc": round(sum(r["ref"] for r in rows) / n, 4),
               "span_rate": round(sum(1 for r in rows if r["n_span"]) / n, 4), "ret_rate": round(sum(1 for r in rows if r["n_ret"]) / n, 4),
               "ret_lat_s": avg("ret_lat_s"), "inj_lat_s": avg("inj_lat_s"), "answer_onset_s": avg("answer_onset_s")}
    json.dump({"summary": summary, "rows": rows}, open(f"{a.run_dir}/rag_report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
