"""The tool bank: the values a router cannot know, fetched and rendered as one span sentence.

`ROUTER_SYSTEM` in `backend.py` is a tool-routing prompt, and it is emphatic that the current time,
the current weather, a live price and an explicit web search MUST be answered by calling a tool
rather than from the model's own knowledge. With an empty catalog the LLM obeyed and produced a
tool call that nothing could execute, so exactly those requests returned `(no information found)`
and no span was ever injected for them.

This module is the catalog. Every tool declares a JSON schema, `backend.py` puts the schemas in
front of the router, and a tool call is EXECUTED here; the result comes back as one declarative
sentence in the same shape a reference sentence has, because that is what is spliced into the
stream and read aloud.

Three rules the tools follow, all of them for the frame clock rather than for tidiness:

  * one sentence, subject first, 8 to 40 words, third person. It is the agent's own line.
  * a hard timeout. The span has to land inside the natural response delay, so a tool that has not
    answered in `CS_TOOL_TIMEOUT` seconds (default 6) is a failure, not something to wait for.
  * never raise. A dead endpoint returns None and the conversation continues without a span.

Every provider is an environment variable, and a tool whose provider is set to an empty string is
dropped from the catalog rather than offered and then failing. That is the point of `available()`:
the router is only ever shown tools that can actually run here, so it cannot pick one that is
guaranteed to come back empty.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable

import requests

log = logging.getLogger(__name__)

TIMEOUT = float(os.environ.get("CS_TOOL_TIMEOUT", "6"))
UA = {"User-Agent": os.environ.get("CS_TOOL_USER_AGENT", "contextspan/1.0 (+tool bank)")}

WEATHER_URL = os.environ.get("CS_WEATHER_URL", "https://wttr.in")
PLACES_URL = os.environ.get("CS_PLACES_URL", "https://nominatim.openstreetmap.org/search")
SEARCH_URL = os.environ.get("CS_SEARCH_URL", "https://en.wikipedia.org")
PLACES_LANG = os.environ.get("CS_PLACES_LANG", "en")

# The released model is English-only, so a span it cannot pronounce is worse than no span at all:
# OpenStreetMap returns local-script names wherever an entry has no `name:en`, and reading
# "금강산오리" out of an English text channel produces noise, not a place name.
_NON_LATIN = re.compile("[^\\x00-\\u02ff\\u1e00-\\u1eff]")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                       # JSON schema, as the router is shown it
    run: Callable[..., str | None]
    enabled: Callable[[], bool] = field(default=lambda: True)

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _str(desc: str) -> dict:
    return {"type": "string", "description": desc}


# ── the tools ───────────────────────────────────────────────────────────────────────────────────
def get_time(timezone: str = "", **_) -> str | None:
    """Wall clock in an IANA zone. Local, no network, so it is the one tool that cannot be down."""
    tz = (timezone or "").strip() or "UTC"
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo(tz))
    except Exception:
        return None
    clock = f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    return f"It is {clock} on {now.strftime('%A')}, {now.day} {now.strftime('%B %Y')} in {tz}."


def get_weather(city: str = "", **_) -> str | None:
    """Current conditions from wttr.in, which needs no key. `CS_WEATHER_URL` swaps the provider."""
    place = (city or "").strip()
    if not place or not WEATHER_URL:
        return None
    try:
        r = requests.get(f"{WEATHER_URL.rstrip('/')}/{urllib.parse.quote(place)}",
                         params={"format": "j1"}, headers=UA, timeout=TIMEOUT)
        cur = r.json()["current_condition"][0]
    except Exception as e:
        log.warning("get_weather(%s) failed: %s", place, e)
        return None
    desc = (cur.get("weatherDesc") or [{}])[0].get("value", "").strip().lower()
    c, f = cur.get("temp_C"), cur.get("temp_F")
    feels, hum = cur.get("FeelsLikeC"), cur.get("humidity")
    if not (desc and c):
        return None
    return (f"It is {desc} in {place} right now, {c} degrees Celsius ({f} Fahrenheit), "
            f"feels like {feels}, humidity {hum} percent.")


def find_places(query: str = "", location: str = "", **_) -> str | None:
    """Named places near a location, from OpenStreetMap's Nominatim (no key).

    This is the tool the restaurant question needs. The router is told never to pass "nearby" or
    "near me", so `location` arrives as a real city; a blank one is a routing failure, not a search
    for everywhere, and returns None rather than a random result from the other side of the world.
    """
    what, where = (query or "").strip(), (location or "").strip()
    if not what or not where or not PLACES_URL:
        return None
    try:
        r = requests.get(PLACES_URL, headers={**UA, "Accept-Language": PLACES_LANG}, timeout=TIMEOUT,
                         params={"q": f"{what} in {where}", "format": "json", "limit": 25,
                                 "namedetails": 1, "accept-language": PLACES_LANG})
        hits = r.json()
    except Exception as e:
        log.warning("find_places(%s, %s) failed: %s", what, where, e)
        return None
    names = []
    for h in hits:
        nd = h.get("namedetails") or {}
        n = (nd.get(f"name:{PLACES_LANG}") or nd.get("name:en") or h.get("name")
             or (h.get("display_name") or "").split(",")[0]).strip()
        if not n or _NON_LATIN.search(n) or n in names:
            continue                          # unpronounceable here: drop it rather than inject it
        names.append(n)
    if not names:
        log.info("find_places(%s, %s): no name the model can read", what, where)
        return None
    top = names[:3]
    listed = ", ".join(top[:-1]) + f" and {top[-1]}" if len(top) > 1 else top[0]
    return f"Places matching {what} in {where} include {listed}."


def web_search(query: str = "", **_) -> str | None:
    """One encyclopedic sentence for an explicit search, from the Wikipedia API (no key).

    The router only routes here when the user asked to search, so a lead paragraph's first sentence
    is the right shape: it is source text, which is what a span is supposed to be.
    """
    q = (query or "").strip()
    if not q or not SEARCH_URL:
        return None
    base = SEARCH_URL.rstrip("/")
    try:
        s = requests.get(f"{base}/w/api.php", headers=UA, timeout=TIMEOUT,
                         params={"action": "query", "list": "search", "srsearch": q,
                                 "format": "json", "srlimit": 1}).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        d = requests.get(f"{base}/api/rest_v1/page/summary/{urllib.parse.quote(title, safe='')}",
                         headers=UA, timeout=TIMEOUT).json()
        extract = (d.get("extract") or "").strip()
    except Exception as e:
        log.warning("web_search(%s) failed: %s", q, e)
        return None
    if not extract:
        return None
    first = extract.split(". ")[0].strip().rstrip(".") + "."
    return first if len(first.split()) >= 6 else extract[:300].strip()


BANK: list[Tool] = [
    Tool("get_time",
         "Current wall-clock time and date in an IANA timezone. Use for 'what time is it'.",
         _obj({"timezone": _str("IANA timezone, e.g. Asia/Seoul. Fill from the user profile "
                                "when the user did not name a place.")}, ["timezone"]),
         get_time),
    Tool("get_weather",
         "Current weather conditions in a named city. Use for 'what's the weather like'.",
         _obj({"city": _str("City name, e.g. Seoul. Fill from the user profile when the user did "
                            "not name one. Never pass 'here' or 'nearby'.")}, ["city"]),
         get_weather,
         lambda: bool(WEATHER_URL)),
    Tool("find_places",
         "Find named places of a kind near a city: restaurants, cafes, museums, shops. Use for "
         "'recommend a restaurant', 'somewhere to eat', 'a cafe near me'.",
         _obj({"query": _str("What to look for, e.g. restaurant, sushi, coffee."),
               "location": _str("City or district to search in. Fill from the user profile when "
                                "the user said 'near me'. Never pass 'nearby' or 'here'.")},
              ["query", "location"]),
         find_places,
         lambda: bool(PLACES_URL)),
    Tool("web_search",
         "Search the web for a topic the model cannot already know, and return one sentence. Only "
         "when the user explicitly asks to search.",
         _obj({"query": _str("Search terms.")}, ["query"]),
         web_search,
         lambda: bool(SEARCH_URL)),
]


def _allowed() -> set[str] | None:
    v = os.environ.get("CS_TOOLS")
    return {x.strip() for x in v.split(",") if x.strip()} if v else None


def available() -> list[Tool]:
    """Tools that can actually run here. The router is shown these and nothing else, so it cannot
    pick one whose provider is switched off and strand the request with no span."""
    allow = _allowed()
    return [t for t in BANK if t.enabled() and (allow is None or t.name in allow)]


def schemas() -> list[dict]:
    return [t.schema() for t in available()]


def run(name: str, arguments: dict | None, profile: dict | None = None) -> str | None:
    """Execute a routed tool call and return its span sentence, or None.

    `profile` is the user context the page supplies (city, timezone). The router is told to fill a
    missing location or timezone from it, but it does not always do so, and a missing argument here
    means no span at all — so the profile is applied as a last resort before giving up.
    """
    tool = next((t for t in available() if t.name == name), None)
    if tool is None:
        log.warning("router picked an unavailable tool: %s", name)
        return None
    args = dict(arguments or {})
    p = profile or {}
    if tool.name == "get_time" and not args.get("timezone"):
        args["timezone"] = p.get("timezone") or ""
    if tool.name == "get_weather" and not args.get("city"):
        args["city"] = p.get("city") or ""
    if tool.name == "find_places" and not args.get("location"):
        args["location"] = p.get("city") or ""
    try:
        out = tool.run(**args)
    except Exception as e:                      # a tool must never end the conversation
        log.warning("tool %s raised: %s", name, e)
        return None
    if out:
        log.info("tool %s%s -> %s", name, tuple(args.values()), out)
    return out or None
