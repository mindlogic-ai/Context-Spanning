"""Checks that no argument value of the Full-Duplex-Bench v3 gold tool calls appears in the router prompt or the
toolpack. Values that the official tool docstrings themselves use as examples are allowed (they are what the
official agent sees too).

    FDB_V3_DIR=<clone>/v3 python -m benchmark.fdb.v3_leakcheck
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [os.path.join(HERE, "v3_toolpack.py"),
         os.path.join(HERE, "..", "..", "contextspan", "duetaspan", "runtime", "mcp", "client.py")]


def main():
    v3 = os.environ["FDB_V3_DIR"]
    gold = json.load(open(os.path.join(v3, "benchmark_data_v2.json")))["scenarios"]
    official = open(os.path.join(v3, "lk_agent_tool.py")).read().lower()
    values = {}
    for sc in gold:
        for c in sc["expected_tool_calls"]:
            for k, v in c["args"].items():
                if isinstance(v, str) and v.startswith("$"):
                    continue
                t = str(v).strip().lower()
                if len(t) < 3 or t in official or re.fullmatch(r"[\d.]+", t):
                    continue   # pure numbers (amounts, prices) are not treated as leaks
                values.setdefault(t, set()).add(f"{sc['id']}.{c['function']}.{k}")
    hits = []
    for f in FILES:
        text = open(f).read().lower()
        for t, where in values.items():
            for m in re.finditer(re.escape(t), text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append((os.path.relpath(f, HERE), line, t, sorted(where)[:3]))
    for h in hits:
        print("LEAK", *h)
    print(f"{len(values)} gold values checked against {len(FILES)} files: {len(hits)} hits")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
