"""Disk cache for the free HTTP sources behind the map, media and search tools.

The tools are slow for two reasons, and only one of them is the network. Nominatim's
usage policy is one request per second, and Overpass hands out query slots and answers
429 when they run out — so the adapters sleep on purpose, 1.05 s and 1.5 s per call.
A voice turn cannot afford that twice, and `directions` pays it twice: once per endpoint.

Almost nothing these endpoints return actually changes. Seolleung Station's coordinates are the
same today as last year; so is the road geometry between two points. The answer is to
ask once. On a hit the caller skips the sleep *and* the
round trip — which is why `get` has to be consulted before the throttle, not after.

What genuinely changes gets a short TTL instead of a long one:

  ttl=FOREVER   geocoding, POI, routes   (OSM edits; nobody notices)
  ttl=HOUR      weather forecasts, web search
  ttl=MINUTE    transit plans

Values are JSON. A cached ``null`` would be indistinguishable from a miss, so `put`
refuses to store one.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)
DEFAULT_DB = os.environ.get(
    "MOSHICP_MCP_CACHE", os.path.join(DATA, "moshicp", "mcp_http_cache.sqlite")
)

MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
FOREVER = 30 * DAY

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DEFAULT_DB), exist_ok=True)
        _conn = sqlite3.connect(DEFAULT_DB, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS http (k TEXT PRIMARY KEY, v TEXT, expires REAL)"
        )
        _conn.commit()
    return _conn


def key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get(k: str) -> Any | None:
    with _lock:
        row = _db().execute("SELECT v, expires FROM http WHERE k = ?", (k,)).fetchone()
    if row is None or row[1] < time.time():
        return None
    return json.loads(row[0])


def put(k: str, value: Any, ttl: float) -> None:
    if value is None:
        return
    with _lock:
        _db().execute(
            "INSERT OR REPLACE INTO http VALUES (?, ?, ?)",
            (k, json.dumps(value, ensure_ascii=False), time.time() + ttl),
        )
        _db().commit()


if __name__ == "__main__":
    import tempfile

    DEFAULT_DB = os.path.join(tempfile.mkdtemp(), "c.sqlite")
    _conn = None
    k = key("GET", "https://example.com", {"q": "선릉역"})
    assert get(k) is None, "empty cache must miss"
    put(k, {"lat": 37.5}, FOREVER)
    assert get(k) == {"lat": 37.5}, "value must survive a round trip"
    assert key("GET", "u", {"a": 1, "b": 2}) == key("GET", "u", {"b": 2, "a": 1}), \
        "param order must not change the key"
    expired = key("expired")
    put(expired, {"x": 1}, -1.0)
    assert get(expired) is None, "an expired row must read as a miss"
    put(key("none"), None, FOREVER)
    assert get(key("none")) is None, "null must not be storable"
    print("cache ok")
