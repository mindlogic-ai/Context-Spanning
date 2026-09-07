"""Call every one of the 125 bank tools exactly once and report what happened.

``eval_125`` samples: it proves the tools it touches are right, but it never
touches some of them at all — the 6 PayPal tools had no case in it. This walks
the whole bank, so every name is exercised and every outcome is stated.

Outcomes:

  ok            the tool ran and returned a value
  abstain       the tool ran and truthfully found nothing
  refused       the tool rejected the call (missing a required slot) — correct
  unsupported   no honest backend on this machine; the reason is printed
  ERROR         the tool was supposed to work and did not

Arguments come from the real world wherever possible: an SGD tool replays an
argument set some dialogue actually sent, and where the corpus has none the
slots are filled from a catalog row so the call is answerable. The 6 pay tools
have no corpus at all, so their arguments are synthesized from their schemas.

    python -m contextspan.duetaspan.runtime.mcp.coverage --sgd-dir <clone>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

from contextspan.duetaspan.runtime.mcp import adapters_fs
from contextspan.duetaspan.eval.mcp_toolbank import (
    FIXTURE, _sgd_calls, abroad_cases, browser_cases, fs_cases, live_cases, maps_cases,
    search_cases,
)
from contextspan.duetaspan.runtime.mcp.registry import Unsupported, get_registry
from contextspan.duetaspan.runtime.mcp.world import WorldError
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)

# The pay tools have no corpus. Minimal arguments that satisfy each schema's
# ``required``; the world records them as transactions, no money moves.
PAY_ARGS = {
    "create_product": {"name": "Context Span license", "type": "DIGITAL"},
    "create_order": {"currencyCode": "USD",
                     "items": [{"name": "Context Span license", "quantity": 1}]},
    "create_invoice": {"detail": {"invoice_date": "2026-07-09", "currency_code": "USD"}},
    "create_subscription_plan": {
        "product_id": "PROD-0001", "name": "Monthly",
        "billing_cycles": [{"frequency": {"interval_unit": "MONTH", "interval_count": 1}}],
    },
    "create_shipment_tracking": {"tracking_number": "1Z999", "transaction_id": "TXN-0001"},
    "create_refund": {"capture_id": "CAP-0001"},
}


def _placeholder(slot: str) -> str:
    if "date" in slot:
        return "2019-03-01"
    if "time" in slot:
        return "19:00"
    if slot in ("number_of_tickets", "num_passengers", "number_of_adults",
                "number_of_seats", "number_of_rooms", "stay_length"):
        return "2"
    if slot in ("trip_protection", "add_insurance", "additional_luggage"):
        return "False"
    return "Example"


def _args_from_catalog(reg, function, rng) -> dict:
    """Fill a world tool's required slots from a real catalog row, so it can answer."""
    required = reg.world.required(function)
    if not required:
        return {}
    service = reg.world._meta[function]["service"]
    rows = reg.world._conn.execute(
        "SELECT doc FROM entity WHERE service = ? LIMIT 400", (service,)
    ).fetchall()
    docs = [json.loads(row["doc"]) for row in rows]
    rng.shuffle(docs)
    for doc in docs:
        if all(slot in doc for slot in required):
            return {slot: doc[slot] for slot in required}
    # No single row carries every required slot — an action's slots describe the
    # request, not the entity. Take what the catalog has, fill the rest.
    args = {}
    for slot in required:
        hit = next((doc[slot] for doc in docs if slot in doc), None)
        args[slot] = hit if hit is not None else _placeholder(slot)
    return args


def _fixed_args() -> dict[str, dict]:
    """Reuse the argument sets eval_125 already vetted."""
    args: dict[str, dict] = {}
    every = (maps_cases() + abroad_cases() + fs_cases() + live_cases()
             + search_cases() + browser_cases())
    for case in every:
        args.setdefault(case.function, case.args)
    return args


# This sweep walks the bank in its own order, which reaches `end_codegen_session` before
# `start_codegen_session` ever runs. eval_125 drives the tools as a sequence; here each
# one is called cold, so the state they assume has to be laid down first.
BROWSER_ARGS = {
    "start_codegen_session": {"options": {}},
    "end_codegen_session": {"sessionId": "codegen-001"},
    "get_codegen_session": {"sessionId": "codegen-002"},
    "clear_codegen_session": {"sessionId": "codegen-003"},
    # eval_125 navigates to example.com first to record a response; the fixture below
    # does that instead, so the sweep's own navigate should land on the fixture page.
    "playwright_navigate": {"url": FIXTURE},
    "puppeteer_navigate": {"url": FIXTURE},
}

# Browser tools are stateful, so alphabetical order would click before it navigates.
# eval_125 already encodes a working order; reuse it.
_BROWSER_ORDER = {case.function: i for i, case in enumerate(browser_cases())}


def _sweep_key(reg, function: str) -> tuple:
    domain = reg.bank[function]["domain"]
    return (domain, _BROWSER_ORDER.get(function, 0), function)


def _lay_browser_fixture(reg) -> None:
    if reg.backend_of("playwright_navigate") != "browser":
        return
    from contextspan.duetaspan.runtime.mcp import adapters_browser as browser

    for _ in range(3):                       # codegen-001 .. codegen-003
        browser.start_codegen()
    browser.navigate("https://example.com")  # records a response for assert_response
    browser.expect_response(id="r1", url="example.com")
    browser.navigate(FIXTURE)                # the page every selector below expects


def build_args(reg, sgd_dir, rng) -> dict[str, dict]:
    calls = _sgd_calls(sgd_dir) if os.path.isdir(sgd_dir) else {}
    args = _fixed_args()
    args.update(PAY_ARGS)
    args.update(BROWSER_ARGS)
    for function in reg.bank:
        if function in args or reg.backend_of(function) != "world":
            continue
        recorded = [c for c in calls.get(function, []) if c[0]]
        args[function] = (rng.choice(recorded)[0] if recorded
                          else _args_from_catalog(reg, function, rng))
    return args


def classify(reg, function, args) -> tuple[str, str]:
    try:
        result = reg.dispatch(function, args)
    except Unsupported:
        return "unsupported", reg.unsupported_reason(function)
    except WorldError as exc:
        return "refused", str(exc)
    except Exception as exc:
        return "ERROR", f"{type(exc).__name__}: {str(exc)[:60]}"
    if result == "(no information found)":
        return "abstain", ""
    return "ok", result[:58]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgd-dir", default=os.path.join(DATA, "moshicp", "_sgd_src"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    ap.add_argument("--json", help="also write the per-tool results here")
    opts = ap.parse_args()

    rng = random.Random(opts.seed)
    adapters_fs.reset_sandbox()
    # The sweep visits tools alphabetically, so read_file/edit_file/get_file_info
    # run before write_file would have created their target. Lay the fixture down
    # first: those tools legitimately require a file that already exists.
    adapters_fs.write_file("proj/src/a.txt", "line one\nline two\nline three\n")
    adapters_fs.write_file("proj/src/b.txt", "bee")

    reg = get_registry()
    _lay_browser_fixture(reg)
    args_by_tool = build_args(reg, opts.sgd_dir, rng)

    outcomes = collections.Counter()
    rows, records = [], []
    for function in sorted(reg.bank, key=lambda f: _sweep_key(reg, f)):
        args = args_by_tool.get(function, {})
        outcome, detail = classify(reg, function, args)
        outcomes[outcome] += 1
        spec = reg.bank[function]
        rows.append((spec["domain"], function, outcome, detail))
        records.append({
            "function": function, "domain": spec["domain"], "source": spec["source"],
            "kind": reg.world.kind(function) or spec["kind"],
            "backend": reg.backend_of(function), "outcome": outcome,
            "detail": detail, "args": args,
            "reason": (reg.unsupported_reason(function)
                       if outcome == "unsupported" else ""),
        })

    if opts.json:
        with open(opts.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1, default=str)
        print(f"wrote {len(records)} tool results -> {opts.json}")

    if not opts.quiet:
        domain = None
        for dom, function, outcome, detail in rows:
            if dom != domain:
                print(f"\n=== {dom} ===")
                domain = dom
            print(f"  {outcome:12s} {function:34s} {detail}")

    total = sum(outcomes.values())
    ran = outcomes["ok"] + outcomes["abstain"] + outcomes["refused"]
    print(f"\ncalled all {total} bank tools: {dict(sorted(outcomes.items()))}")
    print(f"executed for real: {ran}   unsupported: {outcomes['unsupported']}   "
          f"broken: {outcomes['ERROR']}")
    if outcomes["ERROR"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
