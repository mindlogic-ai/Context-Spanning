"""RAG suite scoring (MoshiRAG protocol, benchmark/README.md section 4).

    python -m benchmark.rag.score <run_dir>

The set is read from the items' meta.json. The judged answer is the model's own text stream with the
<ret>/<span> markers removed; the judged reference is the injected span text. Judges and averages are
moshi-rag's (evaluate/judge, evaluate/score.py): a judge returns None on empty text and -1 when its reply
is unparseable, and both are left out of the averages. Writes <run_dir>/rag_report.json.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os

import numpy as np

from benchmark.rag.moshirag import MoshiRagJudge, strip_tags


def score_item(judge, d):
    m = json.load(open(f"{d}/meta.json")); ev = json.load(open(f"{d}/run_events.json"))
    events = ev.get("events", []); spans = [s for s in events if s.get("inject")]
    text = strip_tags(ev.get("transcript", ""))
    ref_text = " ".join(str(s.get("inject") or "") for s in spans)
    q, gold, qe = m["question"], m["answer"], m["question_end_s"]
    row = {"id": os.path.basename(d), "resp": judge(q, gold, text), "ref": judge(q, gold, ref_text) if spans else None,
           "n_span": len(spans), "n_ret": len(events), "text": text[:300],
           "backend_status": events[-1].get("backend_status") if events else None}
    if spans:
        s0 = spans[0]
        if s0.get("t") is not None:
            row["ret_lat_s"] = round(float(s0["t"]) - qe, 2)
        if s0.get("t_inj") is not None and s0.get("t") is not None:
            row["inj_lat_s"] = round(float(s0["t_inj"]) - float(s0["t"]), 2)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    dirs = sorted(d for d in glob.glob(f"{a.run_dir}/*") if os.path.exists(f"{d}/output.wav"))
    if not dirs:
        raise SystemExit(f"no items under {a.run_dir}")
    mode = json.load(open(f"{dirs[0]}/meta.json"))["set"]
    judge = MoshiRagJudge(mode)
    with cf.ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(lambda d: score_item(judge, d), dirs))
    n = max(len(rows), 1)
    judged = lambda k: [r[k] for r in rows if r.get(k) is not None and r[k] >= 0]
    avg = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    both = [r for r in rows if r.get("ref") == 1 and r.get("resp") is not None and r["resp"] >= 0]
    lat = lambda k: avg([r[k] for r in rows if r.get(k) is not None])
    summary = {"run": a.run_dir, "set": mode, "judge": judge.model, "n": len(rows),
               "resp_acc": avg(judged("resp")), "n_resp_judged": len(judged("resp")),
               "ref_acc": avg(judged("ref")), "n_ref_judged": len(judged("ref")),
               "P(resp|ref)": avg([r["resp"] for r in both]), "n_ref_correct": len(both),
               "resp_acc_all": round(sum(1 for r in rows if r.get("resp") == 1) / n, 4),
               "ref_acc_all": round(sum(1 for r in rows if r.get("ref") == 1) / n, 4),
               "span_rate": round(sum(1 for r in rows if r["n_span"]) / n, 4),
               "ret_rate": round(sum(1 for r in rows if r["n_ret"]) / n, 4),
               "backend_timeout": sum(1 for r in rows if r.get("backend_status") == "timeout"),
               "ret_lat_s": lat("ret_lat_s"), "inj_lat_s": lat("inj_lat_s")}
    json.dump({"summary": summary, "rows": rows}, open(f"{a.run_dir}/rag_report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
