"""Seed the MCP world SQLite from the Google SGD corpus.

The tool bank took its 38 SGD function schemas from BFCL. This takes the other
half of the same dataset: the ``service_results`` those services actually
returned across 26k real tool calls — real venues, phone numbers, prices — and
makes them the catalog our tools read from.

Two fields are taken from SGD's own schema rather than trusted from the bank:

* ``kind`` — the bank derives read/action from the verb, which misreads
  ``RideSharing_2_GetRide`` and ``Trains_1_GetTrainTickets``. SGD marks both
  ``is_transactional``: they hail a car and buy tickets. Here they are actions.
* ``required`` — SGD's ``required_slots`` is what the service itself enforces.

Bank tools in the ``pay`` domain (the shipped bank has none) have no corpus, so they
get no catalog: they are pure actions whose effect is the transaction row they write.

Usage::

    python -m contextspan.duetaspan.runtime.mcp.build_world                 # clones SGD if needed
    python -m contextspan.duetaspan.runtime.mcp.build_world --sgd-dir ./sgd --force
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sqlite3
import subprocess

from contextspan.duetaspan.runtime.mcp.world import DEFAULT_DB, _REQUEST_SLOTS, _ekey
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)
TOOL_BANK = os.path.join(DATA, "moshicp", "mcp_tool_bank.json")
SGD_REPO = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git"
SPLITS = ("train", "dev", "test")

# Services whose entities belong to the *user*, not to a public catalog. Seeding
# them with strangers' rows from the corpus would make Alarm_1_GetAlarms answer
# with someone else's alarms; the user starts with none and adds their own.
USER_STATE_SERVICES = {"Alarm_1"}

_SCHEMA = """
CREATE TABLE meta (
    function TEXT PRIMARY KEY,
    service  TEXT NOT NULL,
    method   TEXT NOT NULL,
    kind     TEXT NOT NULL,
    required TEXT NOT NULL
);
CREATE TABLE entity (
    service TEXT NOT NULL,
    ekey    TEXT NOT NULL,
    doc     TEXT NOT NULL,
    PRIMARY KEY (service, ekey)
);
CREATE TABLE txn (
    txn_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    service TEXT NOT NULL,
    method  TEXT NOT NULL,
    args    TEXT NOT NULL,
    record  TEXT,
    ts      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX entity_by_service ON entity(service);
"""


def _ensure_sgd(sgd_dir: str) -> str:
    if os.path.isdir(os.path.join(sgd_dir, "train")):
        return sgd_dir
    print(f"[world] cloning SGD -> {sgd_dir}", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "-q", SGD_REPO, sgd_dir], check=True)
    return sgd_dir


def _sgd_intents(sgd_dir: str) -> dict[tuple[str, str], dict]:
    """(service, intent) -> {transactional, required} straight from SGD's schema."""
    out: dict[tuple[str, str], dict] = {}
    for split in SPLITS:
        path = os.path.join(sgd_dir, split, "schema.json")
        for service in json.load(open(path, encoding="utf-8")):
            for intent in service["intents"]:
                out[(service["service_name"], intent["name"])] = {
                    "transactional": bool(intent["is_transactional"]),
                    "required": list(intent.get("required_slots") or []),
                }
    return out


def _harvest(sgd_dir: str, wanted: set[str], intents: dict) -> dict[str, list[dict]]:
    """service -> entity docs, mined from every non-transactional service_results."""
    docs: dict[str, list[dict]] = collections.defaultdict(list)
    calls = 0
    for split in SPLITS:
        for path in sorted(glob.glob(os.path.join(sgd_dir, split, "dialogues_*.json"))):
            for dialogue in json.load(open(path, encoding="utf-8")):
                for turn in dialogue["turns"]:
                    for frame in turn.get("frames", []):
                        service = frame.get("service")
                        call = frame.get("service_call")
                        if service not in wanted or not call:
                            continue
                        calls += 1
                        spec = intents.get((service, call["method"]))
                        # Only a READ's results describe catalog entities; an
                        # action's result just confirms what it did.
                        if not spec or spec["transactional"]:
                            continue
                        if service in USER_STATE_SERVICES:
                            continue
                        for result in frame.get("service_results") or []:
                            doc = {
                                k: v for k, v in result.items()
                                if k not in _REQUEST_SLOTS and str(v).strip()
                            }
                            if doc:
                                docs[service].append(doc)
    print(f"[world] scanned {calls} real service calls", flush=True)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgd-dir", default=os.path.join(DATA, "moshicp", "_sgd_src"))
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.db):
        if not args.force:
            raise SystemExit(f"{args.db} exists — pass --force to rebuild")
        os.remove(args.db)

    bank = json.load(open(TOOL_BANK, encoding="utf-8"))
    sgd_tools = [t for t in bank if t["source"] == "BFCL-v3/Google-SGD"]
    pay_tools = [t for t in bank if t["domain"] == "pay"]
    if not sgd_tools:
        raise SystemExit("tool bank has no SGD tools")

    sgd_dir = _ensure_sgd(args.sgd_dir)
    intents = _sgd_intents(sgd_dir)

    rows, services, fixed = [], set(), []
    for tool in sgd_tools:
        parts = tool["function"].split("_")
        service, method = "_".join(parts[:2]), parts[2]
        spec = intents.get((service, method))
        if spec is None:
            raise SystemExit(f"{tool['function']}: no SGD schema for {service}.{method}")
        kind = "action" if spec["transactional"] else "read"
        if kind != tool["kind"]:
            fixed.append(f"{tool['function']}: bank={tool['kind']} -> sgd={kind}")
        services.add(service)
        rows.append((tool["function"], service, method, kind, json.dumps(spec["required"])))
    for tool in pay_tools:
        required = (tool.get("parameters") or {}).get("required") or []
        rows.append((tool["function"], "pay", tool["function"], "action",
                     json.dumps(list(required))))

    docs = _harvest(sgd_dir, services, intents)

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.executescript(_SCHEMA)
    conn.executemany("INSERT INTO meta VALUES (?, ?, ?, ?, ?)", rows)

    seeded = 0
    for service in sorted(docs):
        seen, batch = set(), []
        for doc in docs[service]:
            key = _ekey(doc)
            if key in seen:
                continue
            seen.add(key)
            batch.append((service, key, json.dumps(doc, ensure_ascii=False)))
        conn.executemany("INSERT INTO entity VALUES (?, ?, ?)", batch)
        seeded += len(batch)
        print(f"  {service:16s} {len(docs[service]):6d} results -> {len(batch):5d} unique")
    conn.commit()
    conn.close()

    print(f"\n[world] {len(rows)} tools ({len(sgd_tools)} SGD + {len(pay_tools)} pay), "
          f"{seeded} entities -> {args.db}")
    if fixed:
        print("[world] kind corrected from SGD is_transactional:")
        for line in fixed:
            print("  " + line)


if __name__ == "__main__":
    main()
