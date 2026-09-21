"""A real, stateful world behind the tools that have no real backend.

38 of the 60 tools in ``mcp_tool_bank.json`` come from Google SGD. They have no
service that can actually be called: there is no live ``Restaurants_2`` API. Answering
them with an LLM-invented ``tool_result`` means a READ after an ACTION never sees the
ACTION's effect, and chained dialogues only pretend to depend on each other.

This module gives those tools a real backend instead: a SQLite world seeded with
the *actual* entity records SGD's own dialogues returned (real addresses, phone
numbers, prices), plus a transaction log. Tools really read it and really write
it, so ``Alarm_1_AddAlarm`` followed by ``Alarm_1_GetAlarms`` genuinely returns
the alarm that was just set, and a reservation genuinely exists afterwards.

A query the world cannot satisfy returns ``None`` -> the caller speaks
``(no information found)``. That abstain is truthful, which is the whole point.

Build the DB first::

    python -m contextspan.duetaspan.runtime.mcp.build_world --sgd-dir /path/to/sgd-clone

Public API::

    from contextspan.duetaspan.runtime.mcp.world import World
    world = World()                       # opens the DB read/write
    world.call("Alarm_1_AddAlarm", {"new_alarm_time": "07:30"})
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from typing import Any, Optional
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)
DEFAULT_DB = os.environ.get(
    "MOSHICP_MCP_WORLD", os.path.join(DATA, "moshicp", "mcp_world.sqlite")
)

# Slot values that mean "the user does not care" — never used to filter.
_WILDCARD = {"", "dontcare", "any", "none", "null", "unknown", "n/a"}

# Actions whose effect is a durable user-visible record that a later READ must
# see. Everything else records a transaction and echoes the confirmation.
ACTION_WRITES_ENTITY = {"Alarm_1_AddAlarm"}

# Slots a READ result carries that describe the *request*, not the entity, so
# they must not filter the catalog (a 4-seat table request must not exclude a
# restaurant whose seeded row happened to say 2).
_REQUEST_SLOTS = {
    "number_of_tickets", "number_of_seats", "number_of_adults", "number_of_rooms",
    "num_passengers", "additional_luggage", "add_insurance", "trip_protection",
    "private_visibility", "subtitle_language", "device", "stay_length",
}

# Preferred head slot, so a spoken clause leads with the entity's name.
_HEAD_SLOT = (
    "restaurant_name", "place_name", "hotel_name", "attraction_name", "event_name",
    "movie_name", "movie_title", "title", "track", "car_name", "property_name",
    "address", "therapist_name", "stylist_name", "alarm_name", "receiver",
)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _is_time_slot(key: str) -> bool:
    return key == "time" or key.endswith("_time")


def _matches(doc: dict, args: dict) -> Optional[int]:
    """Score ``doc`` against ``args``; None if any shared slot contradicts.

    Only slots the entity actually carries are compared — an argument the
    catalog has no column for (a date, a passenger count) cannot falsify a row.
    """
    score = 0
    for key, want in args.items():
        if _norm(want) in _WILDCARD or key in _REQUEST_SLOTS:
            continue
        if key not in doc:
            continue
        if _norm(doc[key]) != _norm(want):
            return None
        score += 1
    return score


class WorldError(Exception):
    """A tool was called in a way the world refuses — missing required slots."""


class World:
    """SQLite-backed execution of the SGD tools."""

    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"no MCP world at {db_path} — run "
                f"`python -m contextspan.duetaspan.runtime.mcp.build_world` first"
            )
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._meta = {
            row["function"]: dict(row)
            for row in self._conn.execute("SELECT * FROM meta")
        }

    # ----- introspection ------------------------------------------------------
    def functions(self) -> list[str]:
        return sorted(self._meta)


    def required(self, function: str) -> list[str]:
        """The slots this world actually enforces — SGD's own ``required_slots``."""
        entry = self._meta.get(function)
        return json.loads(entry["required"]) if entry else []

    # ----- execution ----------------------------------------------------------
    def call(self, function: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Execute one tool. Returns a record dict, or None when nothing matches.

        Raises WorldError when a required slot is missing — a real service would
        reject the call too, and the caller should not paper over it.
        """
        entry = self._meta.get(function)
        if entry is None:
            return None
        args = {k: v for k, v in (args or {}).items() if v is not None}

        required = json.loads(entry["required"] or "[]")
        missing = [slot for slot in required if _norm(args.get(slot, "")) in _WILDCARD]
        if missing:
            raise WorldError(f"{function} requires {', '.join(missing)}")

        if entry["kind"] == "read":
            return self._read(entry["service"], args)
        return self._act(function, entry, args)


    def _scored(self, service: str, args: dict) -> list[tuple]:
        rows = self._conn.execute(
            "SELECT rowid, doc FROM entity WHERE service = ?", (service,)
        ).fetchall()
        scored = []
        for row in rows:
            doc = json.loads(row["doc"])
            score = _matches(doc, args)
            if score is not None:
                scored.append((score, row["rowid"], doc))
        # Most slots satisfied wins. Ties go to the most recently written row, so
        # a READ right after an ACTION sees the ACTION's effect rather than a
        # seed that happens to match just as well.
        scored.sort(key=lambda triple: (-triple[0], -triple[1]))
        return scored

    def _read(self, service: str, args: dict) -> Optional[dict]:
        scored = self._scored(service, args)
        relaxed: list[str] = []
        if not scored:
            # A time of day is a preference, not an identity: SGD's own service
            # answers a 16:00 pickup request with the 15:30 car. Treating it as
            # a hard filter is what turns a good answer into "(no information)".
            softened = {k: v for k, v in args.items() if not _is_time_slot(k)}
            if len(softened) < len(args):
                relaxed = [k for k in args if _is_time_slot(k)]
                scored = self._scored(service, softened)
        if not scored:
            return None
        return {"record": scored[0][2], "match_count": len(scored), "relaxed": relaxed}

    def _act(self, function: str, entry: dict, args: dict) -> dict:
        service = entry["service"]
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO txn(service, method, args) VALUES (?, ?, ?)",
                (service, entry["method"], json.dumps(args, ensure_ascii=False)),
            )
            txn_id = cursor.lastrowid
            record = dict(
                args,
                confirmation="%s-%06d" % (_abbrev(service), txn_id),
                status="confirmed",
            )
            self._conn.execute(
                "UPDATE txn SET record = ? WHERE txn_id = ?",
                (json.dumps(record, ensure_ascii=False), txn_id),
            )
            if function in ACTION_WRITES_ENTITY:
                self._upsert_entity(service, _alarm_doc(args))
            self._conn.commit()
        return {"record": record, "match_count": 1}

    def _upsert_entity(self, service: str, doc: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO entity(service, ekey, doc) VALUES (?, ?, ?)",
            (service, _ekey(doc), json.dumps(doc, ensure_ascii=False)),
        )


def _abbrev(service: str) -> str:
    head, _, num = service.partition("_")
    return (head[:3] + num).upper()


def _alarm_doc(args: dict) -> dict:
    """AddAlarm's arguments, renamed to the slots GetAlarms reads back."""
    return {
        "alarm_time": args["new_alarm_time"],
        "alarm_name": args.get("new_alarm_name", "New Alarm"),
    }


def _ekey(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, ensure_ascii=False)


def clause(result: dict, limit: int = 6) -> str:
    """Render a record as one RAW machine-form clause, digits and symbols intact.

    Matches the span contract the training data already uses: the reference keeps
    ``4.00`` and ``408-247-8880``; spelling those out is the model's job.
    """
    doc = result["record"]
    # The confirmation is the whole point of an action; it must never be the slot
    # that falls off the end of the budget.
    pinned = [str(doc[k]) for k in ("confirmation", "status") if k in doc]
    keys = [k for k in _HEAD_SLOT if k in doc]
    keys += [k for k in doc if k not in keys and k not in _REQUEST_SLOTS]
    keys = [k for k in keys if k not in ("confirmation", "status")]

    parts = []
    for key in keys:
        if len(parts) + len(pinned) >= limit:
            break
        value = str(doc[key]).strip()
        if not value:
            continue
        # A bare "True"/"False" says nothing on its own — speak the slot it
        # belongs to, and say nothing at all when the answer is no.
        if value.lower() in ("true", "false"):
            if value.lower() == "true":
                parts.append(key.replace("_", " "))
            continue
        parts.append(value)
    return ", ".join(parts + pinned)
