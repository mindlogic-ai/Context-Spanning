"""Download the five math word-problem test sets of the math benchmark and normalise them.

    python benchmark/math/prepare_questions.py <out>            # -> <out>/questions/<set>.jsonl
    python benchmark/math/prepare_questions.py <out> --raw DIR  # parse local copies instead of downloading

Sets (the STITCH / MoshiRAG math evaluation), 3,822 questions in total:

| set        | items | public source |
| ---------- | ----- | ------------- |
| AddSub     |   395 | kojima-takeshi188/zero_shot_cot `dataset/AddSub/AddSub.json` |
| MultiArith |   600 | kojima-takeshi188/zero_shot_cot `dataset/MultiArith/MultiArith.json` |
| SingleEq   |   508 | kojima-takeshi188/zero_shot_cot `dataset/SingleEq/questions.json` |
| SVAMP      |  1000 | arkilpatel/SVAMP `SVAMP.json` (all 1000 problems) |
| GSM8K      |  1319 | openai/grade-school-math `grade_school_math/data/test.jsonl` (same rows as HF openai/gsm8k main/test) |

Each row is {"id", "dataset", "question", "answer"}: `id` is `<set>-<index>` (the source's own index for
AddSub / MultiArith / SingleEq / SVAMP, the line number for GSM8K), `question` has its whitespace collapsed
(SVAMP: Body + Question, with a full stop added to a Body that lacks final punctuation), and `answer` is
the final number as a string (integers without a decimal point: "43", not "43.0").
The downloaded files are kept in <out>/questions/raw/ and reused on the next run (`--force` re-downloads).
A set that cannot be downloaded or parsed is reported and the script exits non-zero; rows are never invented.
"""
import argparse
import json
import os
import re
import sys
import urllib.request


def _num(x):
    """Canonical numeric string: 43.0 -> '43', 0.5 -> '0.5', '1,234' -> '1234'."""
    if isinstance(x, str):
        x = x.replace(",", "").replace("$", "").replace("%", "").strip()
    v = float(x)
    return str(int(v)) if abs(v - round(v)) < 1e-9 else repr(v)


def _clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def parse_mawps(raw, name):
    """zero_shot_cot dump schema (AddSub, MultiArith, SingleEq): iIndex / sQuestion / lSolutions."""
    return [{"id": f"{name}-{r.get('iIndex', i)}", "dataset": name,
             "question": _clean(r["sQuestion"]), "answer": _num(r["lSolutions"][0])}
            for i, r in enumerate(json.loads(raw))]


def parse_svamp(raw, name):
    rows = []
    for i, r in enumerate(json.loads(raw)):
        q = _clean(r["Body"])
        if q and not q.endswith((".", "?", "!")):
            q += "."
        rows.append({"id": f"{name}-{r.get('ID', i)}", "dataset": name,
                     "question": _clean(q + " " + r["Question"]), "answer": _num(r["Answer"])})
    return rows


def parse_gsm8k(raw, name):
    rows = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"id": f"{name}-{i}", "dataset": name, "question": _clean(r["question"]),
                     "answer": _num(r["answer"].split("####")[-1].strip())})
    return rows


_ZSC = "https://raw.githubusercontent.com/kojima-takeshi188/zero_shot_cot/main/dataset"
# set -> (url, raw file name, parser, expected count). The order is the benchmark's item order.
SOURCES = {
    "AddSub": (f"{_ZSC}/AddSub/AddSub.json", "AddSub.json", parse_mawps, 395),
    "MultiArith": (f"{_ZSC}/MultiArith/MultiArith.json", "MultiArith.json", parse_mawps, 600),
    "SingleEq": (f"{_ZSC}/SingleEq/questions.json", "SingleEq.json", parse_mawps, 508),
    "SVAMP": ("https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json", "SVAMP.json", parse_svamp, 1000),
    "GSM8K": ("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
              "gsm8k_test.jsonl", parse_gsm8k, 1319),
}


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "contextspan-math-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="benchmark directory; questions go to <out>/questions/")
    ap.add_argument("--raw", default=None, help="directory holding the raw files (AddSub.json, MultiArith.json, "
                                                "SingleEq.json, SVAMP.json, gsm8k_test.jsonl) to parse instead of downloading")
    ap.add_argument("--force", action="store_true", help="download again even if <out>/questions/raw/ has the file")
    a = ap.parse_args(argv)
    qdir = os.path.join(a.out, "questions")
    raw_dir = os.path.join(qdir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    failed, total = [], 0
    for name in SOURCES:
        url, raw_name, parse, expected = SOURCES[name]
        cached = os.path.join(a.raw or raw_dir, raw_name)
        try:
            if os.path.exists(cached) and (a.raw or not a.force):
                raw, src = open(cached, "rb").read(), cached
            else:
                raw, src = fetch(url), url
            rows = parse(raw, name)
        except Exception as e:  # report and continue with the other sets
            print(f"{name:10s} FAILED  {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(name)
            continue
        if src == url:
            open(os.path.join(raw_dir, raw_name), "wb").write(raw)
        with open(os.path.join(qdir, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        note = "" if len(rows) == expected else f"  (expected {expected})"
        print(f"{name:10s} {len(rows):5d}  <- {src}{note}")
        total += len(rows)
    print(f"total: {total} questions in {qdir}")
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
