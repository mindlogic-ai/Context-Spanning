#!/usr/bin/env python3
"""The Full-Duplex-Bench v3 table numbers, read from the official reports only:
  <data_root>/<provider>_latency_report.json   (analyze_tool_latency.py)   take-turn rate, latencies, filler
  <evaluation report>                          (evaluate_tool_calls.py)    tool selection, argument accuracy,
                                                                          response quality, interruption rate
  <pass-rate report>                           (evaluate_pass_rate.py)     pass rate
Usage: python benchmark/fdb/v3_summary.py --data-root DIR --provider ours --evaluation E.json --pass-rate P.json [--out S.json]
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--provider", default="ours")
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--pass-rate", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lat = json.load(open(f"{a.data_root}/{a.provider}_latency_report.json"))["aggregate"]
    ev = json.load(open(a.evaluation))
    pr = json.load(open(a.pass_rate))
    fs = lat["filler_stats"]
    n_f = fs["with_filler"] + fs["without_filler"]
    agg = ev.get("aggregate", ev)
    by = agg.get("by_metric", agg)
    lr = agg.get("latency", agg.get("latency_report", {}))

    def pct(x):
        return None if x is None else round(100 * float(x), 1)

    s = {
        "scenarios": lat["total_analyzed"],
        "tool_selection_pct": pct(by.get("tool_selection_acc", {}).get("mean") if isinstance(by.get("tool_selection_acc"), dict) else by.get("tool_selection_acc")),
        "argument_acc_pct": pct(by.get("argument_acc", {}).get("mean") if isinstance(by.get("argument_acc"), dict) else by.get("argument_acc")),
        "response_qual_pct": pct(by.get("response_qual", {}).get("mean") if isinstance(by.get("response_qual"), dict) else by.get("response_qual")),
        "pass_rate_pct": pct(pr.get("overall_pass_rate", pr.get("pass_rate"))),
        "take_turn_rate_pct": pct(lat["turn_taking"]["turn_take_rate"]),
        "interruption_rate_pct": pct(lr.get("interruption_rate")),
        "latency_s": lat["task_completion_latency"].get("mean"),
        "first_response_latency_s": lat["first_response_latency"].get("mean"),
        "tool_call_latency_s": lat["tool_call_latency"].get("mean"),
        "filler_rate_pct": round(100 * fs["with_filler"] / n_f, 1) if n_f else None,
        "counts": {"turn_taken": lat["turn_taking"]["turn_taken"], "latency_n": lat["task_completion_latency"].get("count", 0),
                   "with_filler": fs["with_filler"], "without_filler": fs["without_filler"],
                   "interruptions": lr.get("interruption_count"), "latency_samples": lr.get("total_samples")},
    }
    if a.out:
        json.dump(s, open(a.out, "w"), indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
