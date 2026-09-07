"""One dispatch table for all 125 tools in the bank.

Each bank name resolves to exactly one of:

  world   — SGD + PayPal tools, executed against the stateful SQLite world
  maps    — geocoding, routing, elevation, over keyless open services
  fs      — the official filesystem tools, sandboxed
  live    — the pre-existing servers (finance, websearch)
  UNSUPPORTED — no backend exists and none can be faked honestly

``Unsupported`` is not a failure to hide. A browser tool with no browser
installed, or a Tavily search with no API key, must say so; the alternative is a
tool that returns a plausible sentence it never obtained, which is exactly the
behaviour this package replaces.
"""
from __future__ import annotations

import json
import importlib.util
import os
from typing import Any, Callable

from contextspan.duetaspan.runtime.mcp import adapters_fs, adapters_maps, cache
from contextspan.duetaspan.runtime.mcp.world import World, WorldError, clause
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)
TOOL_BANK = os.path.join(DATA, "moshicp", "mcp_tool_bank.json")

_NEEDS_BROWSER = "no browser installed (playwright/puppeteer MCP server not provisioned)"
# The one thing left with no keyless source. Congestion is measured by cameras and
# loop detectors owned by agencies that gate the feed; there is no OpenStreetMap of
# live traffic. A free key from 서울 열린데이터광장 or TMAP closes it — free, not paid.
_NEEDS_TRAFFIC = ("real-time congestion needs a free agency key "
                  "(서울 열린데이터광장 / TMAP); no keyless feed exists")

_UNSUPPORTED = {"map_road_traffic": _NEEDS_TRAFFIC}


class Unsupported(Exception):
    """The tool exists in the bank but has no honest backend on this machine."""


def _load_bank() -> list[dict]:
    with open(TOOL_BANK, encoding="utf-8") as handle:
        return json.load(handle)


# BFCL states parameter types in Python's vocabulary, so 38 of the bank's schemas
# say ``"type": "dict"`` where JSON Schema says ``"object"``. Any MCP client (and
# any function-calling API) rejects those outright, so translate once, here.
_JSON_TYPE = {"dict": "object", "float": "number", "tuple": "array", "any": None}


def json_schema(schema: Any) -> Any:
    """Return ``schema`` with BFCL's Python type names rewritten to JSON Schema."""
    if isinstance(schema, list):
        return [json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str) and value in _JSON_TYPE:
            mapped = _JSON_TYPE[value]
            if mapped is None:          # "any" — express it by saying nothing
                continue
            out[key] = mapped
        else:
            out[key] = json_schema(value)
    return out


# BFCL renamed SGD slots that collide with Python keywords: `from` -> `_from`,
# `class` -> `_class`. The executor kept SGD's names, so a model that reads the
# advertised schema and sends `_from` gets "requires from" and never recovers.
_RESERVED = {"_from": "from", "_class": "class"}

# The bank inherited Amap's weather schema, which takes a city and nothing else.
# Our adapter is backed by open-meteo, which forecasts — so advertise the slots the
# adapter really implements. A schema that hides them forces the router to drop the
# "주말" in "주말에 부산 날씨 어때?" and answer about right now instead.
_DATE = {"type": "string", "description": "Date in YYYY-MM-DD."}
_END = {"type": "string", "description": "Last date of a range, YYYY-MM-DD."}

# SGD's payment slots describe `amount` as "specified in dollars", so "3만원 보내줘"
# becomes 30000 dollars and confirms. The executor already echoes every argument into
# the confirmation record, so naming the currency is all that was ever missing.
_CURRENCY = {"type": "string", "description": "ISO 4217 currency of `amount`.",
             "enum": ["KRW", "USD", "EUR", "JPY", "GBP", "CNY"]}
_EXTRA_PARAMS = {
    "maps_weather": {"date": _DATE, "end_date": _END},
    "map_weather": {"date": _DATE, "end_date": _END},
    "Payment_1_MakePayment": {"currency": _CURRENCY},
    "Payment_1_RequestPayment": {"currency": _CURRENCY},
}


def _unreserve(name: str) -> str:
    return _RESERVED.get(name, name)


def _browser_available() -> bool:
    """A Chromium that cannot launch is worse than an honest `unsupported`.

    The 32 browser tools stay unsupported on a machine without playwright and its
    bundled browser, rather than failing one call at a time deep inside a dialogue.
    """
    return importlib.util.find_spec("playwright.sync_api") is not None


def _world_override() -> dict[str, Callable]:
    """SGD read tools a keyless live source answers better than the seeded catalog.

    The catalog knows US cities on 2019 dates and a US music library, so a Korean asking
    any of these could only ever be told about a restaurant that does not exist. Every
    tool below now reads a real source: OpenStreetMap, open-meteo, iTunes.

    What stays in the world, and why:
      * Actions — reserving, booking, paying. Executing those for real would take a real
        seat and move real money. A stateful simulation is the honest answer; a mock is not.

    Reads with no keyless structured source — KTX timetables, cinema showtimes, bus/rental/
    event listings — used to return a seeded (fake-but-Korean) row. There is no free API for
    most of them (코레일 gives no public seat feed; cinema times are per-chain), so instead of
    a made-up row they now fall back to **web search** (DuckDuckGo instant answer), per the
    user's request. It is the real answer when the web has one and `(no information found)`
    when it does not — never a fake row, and never a loosely-matched encyclopedia page. Web
    search is worldwide, so "파리 기차" works as well as "서울 부산"; the intent word is English
    so a foreign place term does not fight a Korean one.
    Only the *reads* move; buying still records into the world. `_web_read` joins the args into
    a query — robust to BFCL's slot renaming (`from` -> `_from`), no per-slot template to drift.
    """
    # 생성 env(gsh)에는 mcp/fastmcp 패키지가 없어 server 모듈 import가 죽는다. 그 경우
    # override 없이 world(SQLite 시드)로 degrade — 데이터 생성용 dispatch에는 그게 오히려
    # 결정론적이고 네트워크 무의존이라 적합하다 (user 2026-07-21: backend/mcp를 참고해 생성).
    try:
        from contextspan.duetaspan.runtime.mcp.adapters_media import lookup_music
        from contextspan.duetaspan.runtime.mcp.servers.websearch_server import instant_answer
    except ImportError:
        return {}

    # DuckDuckGo's instant answer only (not full web_search): a clear factual answer or
    # "(no information found)" — never a loosely-matched Wikipedia page passed off as the
    # answer, which is the "정보 애매하면 no-info" the user asked for. One request, no
    # Wikipedia round trip, so it stays well under a second (~120-450 ms warm). Memoised.
    web = _memo("webread", cache.HOUR, instant_answer)

    def _web_read(intent: str) -> Callable:
        def run(**kw: Any) -> str:
            terms = " ".join(str(v) for v in kw.values() if isinstance(v, str) and v.strip())
            query = f"{terms} {intent}".strip()
            return web(query) if terms else "(no information found)"
        return run

    return {
        "Weather_1_GetWeather": adapters_maps.weather_on,
        "Travel_1_FindAttractions": adapters_maps.attractions,
        "Music_3_LookupMusic": lookup_music,
        "Restaurants_2_FindRestaurants": adapters_maps.find_restaurants,
        "Hotels_4_SearchHotel": adapters_maps.search_hotel,
        "Hotels_2_SearchHouse": adapters_maps.search_house,
        "Services_1_FindProvider": adapters_maps.find_salon,
        "Trains_1_FindTrains": _web_read("train schedule"),
        "Buses_3_FindBus": _web_read("intercity bus schedule"),
        "RentalCars_3_GetCarsAvailable": _web_read("car rental"),
        "Events_3_FindEvents": _web_read("events tickets"),
        "Movies_1_FindMovies": _web_read("movies now showing"),
        "Movies_1_GetTimesForMovie": _web_read("movie showtimes"),
        # seeded flights cover only 15 SGD cities on 2019-03 dates — a Seoul→Tokyo search can
        # never answer from the world, so the read goes to the web like trains/buses above.
        "Flights_4_SearchOnewayFlight": _web_read("flight schedule"),
        "Flights_4_SearchRoundtripFlights": _web_read("round trip flights"),
    }


def _memo(name: str, ttl: float, produce: Callable[[str], str]) -> Callable[[str], str]:
    """Answer a repeated lookup from disk. The two live servers reach the network on
    every call; a quote is worth a minute and "capital of South Korea" is worth an hour."""
    def wrapped(term: str) -> str:
        hot = cache.key(name, term)
        stored = cache.get(hot)
        if stored is not None:
            return stored
        answer = produce(term)
        cache.put(hot, answer, ttl)
        return answer
    return wrapped


def _live_tools() -> dict[str, Callable]:
    """Reuse the servers already in this package rather than reimplement them."""
    try:
        from contextspan.duetaspan.runtime.mcp.servers.finance_server import get_stock_price
        from contextspan.duetaspan.runtime.mcp.servers.websearch_server import web_search
    except ImportError:                      # mcp/fastmcp 미설치 env: live 없이 world만
        return {}

    quote = _memo("stock", cache.MINUTE, get_stock_price)
    lookup = _memo("websearch", cache.HOUR, web_search)
    return {
        "get_stock_price_global_market": lambda **kw: quote(
            kw.get("company") or kw.get("symbol") or kw.get("query", "")
        ),
        "search": lambda **kw: lookup(kw.get("query") or kw.get("q", "")),
    }


class Registry:
    """Resolve a bank tool name to a real, executing callable."""

    def __init__(self) -> None:
        self.bank = {tool["function"]: tool for tool in _load_bank()}
        self.world = World()
        override = _world_override()
        self._world_fns = set(self.world.functions()) - set(override)
        self._impl: dict[str, Callable] = {}
        self._impl.update(adapters_fs.TOOLS)
        self._impl.update(adapters_maps.TOOLS)
        self._impl.update(_live_tools())
        self._impl.update(override)
        from contextspan.duetaspan.runtime.mcp import adapters_search
        self._impl.update(adapters_search.TOOLS)
        if _browser_available():
            from contextspan.duetaspan.runtime.mcp import adapters_browser
            self._impl.update(adapters_browser.TOOLS)
            self._browser_fns = set(adapters_browser.TOOLS)
        else:
            self._browser_fns = set()
        # Handshakes to the European map hosts cost ~800 ms each. Start them now, in the
        # background, so the user's first question does not wait for them.
        adapters_maps.warm_connections()
        adapters_search.warm_connections()
        try:
            from contextspan.duetaspan.runtime.mcp.servers import websearch_server
            websearch_server.warm_connections()
        except ImportError:
            pass                             # mcp 패키지 없는 생성 env: world-only 모드

    # ----- classification -----------------------------------------------------
    def backend_of(self, function: str) -> str:
        if function not in self.bank:
            return "unknown"
        if function in _UNSUPPORTED:
            return "unsupported"
        if function in self._world_fns:
            return "world"
        if function in adapters_fs.TOOLS:
            return "fs"
        if function in adapters_maps.TOOLS:
            return "maps"
        if function in self._browser_fns:
            return "browser"
        if function in self._impl:
            return "live"
        return "unsupported"

    def unsupported_reason(self, function: str) -> str:
        if function in _UNSUPPORTED:
            return _UNSUPPORTED[function]
        if self.bank.get(function, {}).get("domain") == "browser":
            return _NEEDS_BROWSER
        return "no backend"

    def supported(self) -> list[str]:
        return [f for f in self.bank if self.backend_of(f) != "unsupported"]

    def input_schema(self, function: str) -> dict:
        """The tool's JSON Schema, valid for MCP and for function-calling APIs.

        For world tools the advertised ``required`` is the world's own — SGD's
        ``required_slots`` — not BFCL's. BFCL marks ``Alarm_1_GetAlarms`` as
        needing a ``user_id`` the service never had, and a schema that demands
        an argument the executor ignores rejects calls that would have worked.
        """
        schema = json_schema(self.bank[function].get("parameters") or {})
        if not schema:
            return {"type": "object", "properties": {}}
        if function in _EXTRA_PARAMS:
            schema.setdefault("properties", {}).update(_EXTRA_PARAMS[function])
        if self.backend_of(function) == "world":
            properties = schema.get("properties") or {}
            for name in [k for k in properties if _unreserve(k) != k]:
                properties[_unreserve(name)] = properties.pop(name)
            required = self.world.required(function)
            if required:
                schema["required"] = required
            else:
                schema.pop("required", None)
        return schema

    # ----- execution ----------------------------------------------------------
    def dispatch(self, function: str, args: dict[str, Any] | None = None) -> str:
        """Run one tool for real. Returns a RAW machine-form clause.

        Raises Unsupported / WorldError / the adapter's own error. Returns the
        literal ``(no information found)`` when a real lookup found nothing —
        that is an answer, not an error.
        """
        args = args or {}
        backend = self.backend_of(function)
        if backend == "unknown":
            raise Unsupported(f"{function} is not in the tool bank")
        if backend == "unsupported":
            raise Unsupported(f"{function}: {self.unsupported_reason(function)}")

        if backend == "world":
            args = {_unreserve(k): v for k, v in args.items()}
            result = self.world.call(function, args)
            return clause(result) if result else "(no information found)"
        return str(self._impl[function](**args))

    def close(self) -> None:
        self.world.close()


_REGISTRY: Registry | None = None


def get_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY


__all__ = ["Registry", "Unsupported", "WorldError", "get_registry"]
