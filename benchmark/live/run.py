"""Live-session benchmark: score the demo the way a person uses it.

Every other runner in `benchmark/` streams a benchmark's own audio through the library. This one drives
the **browser page** with a fake microphone, one question per session, and scores what the page
actually rendered — so it measures the thing that ships: ASR, the router, the deadline, the span,
and what the model said with it.

    python -m benchmark.live.run <url> <clips_dir> <out_dir> [--limit N]

`clips_dir` holds `<id>.wav` (the spoken question, already padded with lead-in and tail silence) and
`cases.json` (`[{id, question, kind, expect: [...], not_expect: [...]}]`). Per case it records the
rendered turns, the transcript events, the retrieval timings, and three verdicts:

  span     a span arrived and was injected                (retrieval worked at all)
  ontime   it arrived inside the deadline                  (it was usable)
  answer   the agent's words contain the expected fact     (it was used)

`grounded` is the join that matters: a span arrived on time AND the answer is right. `invented` is
the failure that matters: no span arrived and the agent answered anyway with something specific.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time

from playwright.async_api import async_playwright

WORD = re.compile(r"[a-z0-9]+")
NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
             "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
             "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
             "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90"}


def norm(s: str) -> str:
    """Lowercase words, with spoken numerals folded to digits so 'eighteen forty-two' can be
    compared with '1842' — the agent speaks numbers and the gold answers are written."""
    toks = WORD.findall((s or "").lower().replace("-", " "))
    return " ".join(NUM_WORDS.get(t, t) for t in toks)


_SCALE_RUN = re.compile(
    r"\b((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|and|point)\s+)+)"
)


def norm_scaled(s: str) -> str:
    """`norm`, but a run of number words that carries a scale word (hundred, thousand, million) is
    folded to its VALUE with word2number: "eight thousand eight hundred and forty-eight point eight
    six" -> "8848.86". `norm` alone folds token by token ("8 1000 8 100 and 40 8"), which is right
    for years and clock times ("eighteen forty-two" -> "1842") and wrong for magnitudes; the two
    renderings are complementary, so `contains` tries both."""
    from word2number import w2n
    text = (s or "").lower().replace("-", " ")

    def fold(m):
        run = m.group(1)
        if not re.search(r"\b(hundred|thousand|million|billion)\b", run):
            return run
        try:
            v = w2n.word_to_num(run.strip())
        except ValueError:
            return run
        return f"{v} "

    return norm(_SCALE_RUN.sub(fold, text + " "))


def contains(hay: str, needle: str) -> bool:
    """The agent SPEAKS its answer and the gold is written, so the two never match literally.
    Compare on normalised words, then again with the spaces removed, which is what makes "six
    fifty-one P M" match "PM"; and on the scale-folded rendering, which is what makes "8848"
    match "eight thousand eight hundred and forty-eight"."""
    n = norm(needle)
    for h in (norm(hay), norm_scaled(hay)):
        if n in h or n.replace(" ", "") in h.replace(" ", ""):
            return True
    return False


async def one(page, url: str, wav: str, case: dict, seconds: float) -> dict:
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page.fill("#name", "Seonghyeon")
    await page.fill("#loc", "Seoul")
    errs: list[str] = []
    page.on("pageerror", lambda e: errs.append(str(e)[:120]))
    await page.click("#go")
    t0 = time.time()
    while time.time() - t0 < seconds:
        await asyncio.sleep(1.0)
    turns = await page.evaluate("""() => [...document.querySelectorAll('#thread .bubble')]
        .map(b => ({who: b.dataset.who, text: b.innerText.trim()}))""")
    await page.click("#go")
    await page.wait_for_selector("#rec:not([hidden])", timeout=15000)
    await asyncio.sleep(0.6)
    js = await page.evaluate("""async () => {
        const a = document.querySelector('#recjson');
        return a && a.href ? await (await fetch(a.href)).text() : null; }""")
    tr = json.loads(js) if js else {"events": []}
    ev = tr.get("events", [])

    agent = " ".join(t["text"] for t in turns if t["who"] == "agent")
    heard = " ".join(t["text"] for t in turns if t["who"] == "user")
    spans = [e for e in ev if e["type"] == "span"]
    rets = [e for e in ev if e["type"] == "ret"]
    qs = [e for e in ev if e["type"] == "question"]
    lat = None
    if rets and (spans or [e for e in ev if e["type"] in ("no_span", "late")]):
        end = (spans or [e for e in ev if e["type"] in ("no_span", "late")])[0]
        lat = round(end["t"] - rets[0]["t"], 3)

    expect = case.get("expect", [])
    got = bool(expect) and any(contains(agent, e) for e in expect)
    bad = any(contains(agent, b) for b in case.get("not_expect", []))
    return {
        "id": case["id"], "kind": case.get("kind", ""), "question": case["question"],
        "heard": heard, "agent": agent,
        "span": bool(spans), "span_text": spans[0]["text"] if spans else None,
        "late": any(e["type"] == "late" for e in ev), "no_span": any(e["type"] == "no_span" for e in ev),
        "ret_count": len(rets), "asr_ms": qs[0].get("ms") if qs else None,
        "retrieval_s": lat, "answer_ok": got and not bad, "contradicted": bad,
        "page_errors": errs, "events": ev, "turns": turns,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("clips"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()
    cases = json.load(open(os.path.join(a.clips, "cases.json")))
    if a.limit:
        cases = cases[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    async with async_playwright() as p:
        for i, c in enumerate(cases, 1):
            wav = os.path.abspath(os.path.join(a.clips, c["id"] + ".wav"))
            br = await p.chromium.launch(headless=True, args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}%noloop",
                "--autoplay-policy=no-user-gesture-required"])
            ctx = await br.new_context(permissions=["microphone"])
            page = await ctx.new_page()
            try:
                r = await one(page, a.url, wav, c, a.seconds)
            except Exception as e:
                r = {"id": c["id"], "kind": c.get("kind", ""), "question": c["question"],
                     "error": f"{type(e).__name__}: {e}"[:200], "span": False, "answer_ok": False}
            await br.close()
            rows.append(r)
            mark = "ok " if r.get("answer_ok") else "MISS"
            print(f"  [{i}/{len(cases)}] {mark} {c['id']:16s} span={str(r.get('span')):5s} "
                  f"lat={r.get('retrieval_s')} :: {str(r.get('agent'))[:70]}", flush=True)
            await asyncio.sleep(2.0)          # the engine holds one conversation at a time

    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1, ensure_ascii=False)
    n = len(rows)
    lats = [r["retrieval_s"] for r in rows if r.get("retrieval_s") is not None]
    by = {}
    for r in rows:
        k = r.get("kind", "")
        d = by.setdefault(k, {"n": 0, "span": 0, "ok": 0})
        d["n"] += 1; d["span"] += bool(r.get("span")); d["ok"] += bool(r.get("answer_ok"))
    summary = {
        "n": n,
        "span_rate": round(sum(bool(r.get("span")) for r in rows) / n, 3),
        "late_rate": round(sum(bool(r.get("late")) for r in rows) / n, 3),
        "answer_accuracy": round(sum(bool(r.get("answer_ok")) for r in rows) / n, 3),
        "grounded": round(sum(bool(r.get("span")) and bool(r.get("answer_ok")) for r in rows) / n, 3),
        "invented": round(sum((not r.get("span")) and bool(r.get("agent")) and not r.get("answer_ok")
                              for r in rows) / n, 3),
        "contradicted_span": round(sum(bool(r.get("contradicted")) for r in rows) / n, 3),
        "retrieval_s_p50": round(statistics.median(lats), 2) if lats else None,
        "retrieval_s_p90": round(sorted(lats)[int(len(lats) * 0.9) - 1], 2) if len(lats) >= 2 else None,
        "by_kind": {k: {"n": v["n"], "span_rate": round(v["span"] / v["n"], 2),
                        "answer_accuracy": round(v["ok"] / v["n"], 2)} for k, v in by.items()},
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
