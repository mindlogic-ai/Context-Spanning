"""Execute 250 real tool calls across every supported tool and score them.

The point is not "did the call return a string". It is "did it return what the
real world holds". So the cases are not invented:

* SGD tools — replayed from the corpus. Each case sends an argument set some
  real dialogue actually sent, and passes only when the world's answer quotes
  the entity that service actually returned in ``service_results``.
* actions — pass when the effect is observable afterwards: a transaction row
  exists, or a later READ returns what the ACTION wrote.
* maps / fs / live — pass on an invariant only a real backend satisfies (Seoul
  geocodes inside its bounding box, a file read returns what was written, a
  stock quote contains a number).

Unsupported tools are reported with their reason, never as a pass.

    python -m contextspan.duetaspan.eval.mcp_toolbank --sgd-dir <clone> [--n 250] [--offline]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sqlite3
import urllib.parse

from contextspan.duetaspan.runtime.mcp import adapters_fs
from contextspan.duetaspan.runtime.mcp.registry import Unsupported, get_registry
from contextspan.duetaspan.runtime.mcp.world import WorldError, _norm, _REQUEST_SLOTS
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)

# 고성현, 대한민국 — the default context for tools that need a "where".
CTX = {"city": "Seoul", "country": "KR", "timezone": "Asia/Seoul",
       "lat": 37.5665, "lon": 126.9780}
_SEOUL_BOX = (37.4, 37.7, 126.7, 127.2)          # lat_min, lat_max, lon_min, lon_max

CALL, SELF, RAISES = "call", "self", "raises"    # how the runner drives a case


class Case:
    """``mode`` decides who calls the tool: the runner, the check, or nobody."""

    __slots__ = ("function", "args", "check", "note", "mode")

    def __init__(self, function, args, check, note="", mode=CALL):
        self.function, self.args = function, args
        self.check, self.note, self.mode = check, note, mode


# ----- SGD-replayed cases -----------------------------------------------------
def _sgd_calls(sgd_dir: str) -> dict[str, list[tuple[dict, list]]]:
    """function -> [(args, service_results)] over every recorded real call."""
    out: dict[str, list] = collections.defaultdict(list)
    for split in ("train", "dev", "test"):
        for path in sorted(glob.glob(os.path.join(sgd_dir, split, "dialogues_*.json"))):
            for dialogue in json.load(open(path, encoding="utf-8")):
                for turn in dialogue["turns"]:
                    for frame in turn.get("frames", []):
                        call = frame.get("service_call")
                        if not call:
                            continue
                        function = f"{frame['service']}_{call['method']}"
                        out[function].append(
                            (dict(call["parameters"]), frame.get("service_results") or [])
                        )
    return out


def _args_key(args: dict) -> str:
    return json.dumps({k: _norm(v) for k, v in sorted(args.items())}, sort_keys=True)


def _agrees(record: dict, expected: list[dict]) -> bool:
    """The record must BE one of the entities SGD ever returned for these args.

    Compared slot-by-slot on the record rather than on the spoken clause, so a
    clause that merely ran out of room is not scored as a wrong answer. Request
    slots are ignored: they describe the booking, not the entity.

    ``expected`` is the union over every recorded call with identical arguments,
    because an under-determined query ("attractions in Toronto") legitimately has
    dozens of right answers and SGD's service returned only the one it ranked
    first. Returning a row outside that union still fails.
    """
    for row in expected:
        shared = [k for k in row if k in record and k not in _REQUEST_SLOTS]
        if shared and all(_norm(row[k]) == _norm(record[k]) for k in shared):
            return True
    return False


def _read_check(function, args, expected, tally):
    """Judge a read on its record, at the strictest standard the query allows.

    A query that pins exactly one catalog entity must return the row SGD's own
    service returned. A query that pins many ("attractions in Toronto" matches
    55) has many right answers — SGD returned whichever its ranker put first —
    so the world only has to return one of them. Returning anything outside the
    match set means it invented a row, and fails either way.
    """
    def check(reg, _got):
        result = reg.world.call(function, args)
        candidates = reg.world.match_all(function, args)

        if not expected:
            return None if result is None else f"invented a result: {result['record']}"
        if result is None:
            return "(no information found) but SGD's service returned rows"

        record = result["record"]
        if len(candidates) <= 1:
            tally["strict"] += 1
            if not _agrees(record, expected):
                return f"unique-match query returned a row SGD never did: {record}"
            return None

        tally["underdetermined"] += 1
        if not any(record == other for other in candidates):
            return f"row is not in the catalog match set: {record}"
        if _agrees(record, expected):
            tally["also_exact"] += 1
        return None
    return check


def _action_check(function):
    def check(reg, got):
        if "confirmed" not in got:
            return f"no confirmation in {got[:60]!r}"
        service = reg.world._meta[function]["service"]
        conn = sqlite3.connect(reg.world.db_path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM txn WHERE service = ? AND record IS NOT NULL",
            (service,),
        ).fetchone()[0]
        conn.close()
        return None if rows else "no transaction row was written"
    return check


# Alarm_1 holds the user's own alarms, so it is deliberately not seeded from the
# corpus. Replaying it against SGD's ground truth would be asking our world to
# return a stranger's alarms; the add -> get roundtrip covers it instead.
_NO_REPLAY = {"Alarm_1_GetAlarms"}


def sgd_cases(sgd_dir: str, reg, rng, budget: int, tally: dict) -> list[Case]:
    calls = _sgd_calls(sgd_dir)

    # Every row SGD ever returned for one exact argument set, across all dialogues.
    union: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for function, recorded in calls.items():
        for args, results in recorded:
            union[(function, _args_key(args))].extend(results)

    functions = [f for f in reg.bank
                 if reg.backend_of(f) == "world" and f in calls and f not in _NO_REPLAY]

    def _pool(function):
        pool = calls[function]
        if reg.world.kind(function) == "read":
            return [c for c in pool if c[1]]          # only calls that returned rows
        return pool

    def _case(function, args):
        if reg.world.kind(function) == "read":
            expected = union[(function, _args_key(args))]
            return Case(function, args, _read_check(function, args, expected, tally),
                        "read", SELF)
        return Case(function, args, _action_check(function), "action")

    # Spread the budget evenly, then top up round-robin so the run hits exactly
    # ``budget`` cases instead of however many integer division left behind.
    per = max(1, budget // max(1, len(functions)))
    cases, leftovers = [], []
    for function in sorted(functions):
        pool = _pool(function)
        if not pool:
            continue
        chosen = rng.sample(pool, min(per, len(pool)))
        cases.extend(_case(function, args) for args, _ in chosen)
        leftovers.extend((function, args) for args, _ in pool[len(chosen):])
    rng.shuffle(leftovers)
    for function, args in leftovers[: max(0, budget - len(cases))]:
        cases.append(_case(function, args))
    return cases


# ----- live-backend cases -----------------------------------------------------
def _in_seoul(got: str) -> bool:
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", got)]
    lat_min, lat_max, lon_min, lon_max = _SEOUL_BOX
    return (any(lat_min <= n <= lat_max for n in nums)
            and any(lon_min <= n <= lon_max for n in nums))


def _expect(predicate, why):
    return lambda reg, got: None if predicate(got) else f"{why}: {got[:70]!r}"


# (lat_min, lat_max, lon_min, lon_max)
_MANHATTAN = (40.70, 40.85, -74.05, -73.90)
_PARIS = (48.80, 48.92, 2.20, 2.45)


def _in_box(box):
    def check(got: str) -> bool:
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", got)]
        lat_min, lat_max, lon_min, lon_max = box
        return (any(lat_min <= n <= lat_max for n in nums)
                and any(lon_min <= n <= lon_max for n in nums))
    return check


def abroad_cases() -> list[Case]:
    """The country bias that keeps "Hongdae" in Seoul must not drag "Times Square"
    there too. Only Seoul was ever asserted, so a geocoder that answered every
    English landmark with its Korean namesake still scored 250/250.
    """
    in_nyc = _expect(_in_box(_MANHATTAN), "Times Square is not in Manhattan")
    return [
        Case("maps_geocode", {"address": "Times Square"}, in_nyc),
        Case("maps_geocode", {"address": "Central Park"}, in_nyc),
        Case("maps_geo", {"address": "Eiffel Tower"},
             _expect(_in_box(_PARIS), "Eiffel Tower is not in Paris")),
        # A Manhattan crosstown hop, not a 33 km drive to a Korean namesake.
        Case("maps_directions", {"origin": "Times Square", "destination": "Central Park"},
             _expect(lambda g: float(g.split(" km")[0]) < 10, "route is not in Manhattan")),
        # And the home bias must still rescue an unnotable local nickname.
        Case("maps_geocode", {"address": "Hongdae"},
             _expect(_in_seoul, "Hongdae left Seoul")),
        # Seoul files 성수동 as 성수동1가, so Nominatim's only exact hit is a road in
        # Jeju. This has regressed twice — once when ranking changed, once when a
        # timeout was tightened — and the display name says "성수동1가" either way.
        # Only the coordinates tell the truth, so assert on those.
        Case("maps_geocode", {"address": "성수동"},
             _expect(_in_seoul, "성수동 resolved outside Seoul")),
        # A walk down Teheran-ro takes about ten minutes. The public OSRM demo
        # answered "1 min" for every profile it was asked about, car route and all.
        Case("maps_direction_walking", {"origin": "선릉역", "destination": "선정릉역"},
             _expect(lambda g: int(re.search(r"(\d+) min", g).group(1)) >= 5,
                     "walking is as fast as driving")),
        # OSRM has no transit profile. Transitous does, keyless, on published GTFS.
        Case("maps_direction_transit_integrated",
             {"origin": "선릉역", "destination": "선정릉역"},
             _expect(lambda g: "min" in g and "transfer" in g, "no transit itinerary")),
    ]


_FIXTURE_HTML = (
    "<h1 id=t>hello</h1><input id=i>"
    "<select id=s><option value=a>a</option><option value=b>b</option></select>"
    "<button id=b onclick=\"console.log('hi')\">go</button>"
    "<a id=newtab href='https://example.com' target='_blank'>tab</a>"
    "<div id=src>drag</div><div id=dst>drop</div>"
    "<iframe id=f srcdoc=\"<button id=ib>ok</button><input id=ii>\"></iframe>"
)
# charset must be declared, or Chromium reads the bytes as Latin-1.
FIXTURE = "data:text/html;charset=utf-8," + urllib.parse.quote(_FIXTURE_HTML)


def browser_cases() -> list[Case]:
    """Drive a real Chromium. Order matters: a click needs a page to click on.

    Everything is asserted against a `data:` fixture and example.com, so the check does
    not depend on some third party's markup staying put.
    """
    http = _expect(lambda g: "HTTP" in g, "no HTTP status")
    return [
        Case("start_codegen_session", {"options": {}},
             _expect(lambda g: g.startswith("codegen-"), "no session id")),
        Case("playwright_navigate", {"url": "https://example.com"},
             _expect(lambda g: "(200)" in g, "example.com did not load")),
        Case("playwright_expect_response", {"id": "r1", "url": "example.com"},
             _expect(lambda g: "watching" in g, "expectation not registered")),
        Case("playwright_assert_response", {"id": "r1"}, http),
        Case("playwright_navigate", {"url": FIXTURE},
             _expect(lambda g: "data:" in g, "fixture did not load")),
        Case("playwright_evaluate", {"script": "document.getElementById('t').innerText"},
             _expect(lambda g: g == "hello", "wrong text")),
        Case("playwright_click", {"selector": "#b"},
             _expect(lambda g: "clicked" in g, "click failed")),
        Case("playwright_console_logs", {"search": "hi"},
             _expect(lambda g: "hi" in g, "console log not captured")),
        Case("playwright_fill", {"selector": "#i", "value": "안녕"},
             _expect(lambda g: "filled" in g, "fill failed")),
        Case("playwright_select", {"selector": "#s", "value": "b"},
             _expect(lambda g: "selected" in g, "select failed")),
        Case("playwright_hover", {"selector": "#t"},
             _expect(lambda g: "hovered" in g, "hover failed")),
        Case("playwright_press_key", {"key": "Enter", "selector": "#i"},
             _expect(lambda g: "pressed" in g, "key press failed")),
        Case("playwright_drag", {"sourceSelector": "#src", "targetSelector": "#dst"},
             _expect(lambda g: "dragged" in g, "drag failed")),
        Case("playwright_screenshot", {"name": "eval_shot"},
             _expect(lambda g: "bytes" in g, "no screenshot")),
        Case("playwright_iframe_fill",
             {"iframeSelector": "#f", "selector": "#ii", "value": "x"},
             _expect(lambda g: "filled" in g, "iframe fill failed")),
        Case("playwright_iframe_click", {"iframeSelector": "#f", "selector": "#ib"},
             _expect(lambda g: "clicked" in g, "iframe click failed")),
        Case("playwright_get", {"url": "https://example.com"}, http),
        Case("playwright_post", {"url": "https://example.com", "value": "{}"}, http),
        Case("playwright_put", {"url": "https://example.com", "value": "{}"}, http),
        Case("playwright_patch", {"url": "https://example.com", "value": "{}"}, http),
        Case("playwright_delete", {"url": "https://example.com"}, http),
        Case("playwright_custom_user_agent", {"userAgent": "Mozilla/5.0 (iPhone)"},
             _expect(lambda g: "user agent" in g, "user agent not set")),
        Case("playwright_click_and_switch_tab", {"selector": "#newtab"},
             _expect(lambda g: "example.com" in g, "did not switch tab")),
        Case("puppeteer_navigate", {"url": FIXTURE},
             _expect(lambda g: "data:" in g, "fixture did not reload")),
        Case("puppeteer_evaluate", {"script": "1+1"},
             _expect(lambda g: g == "2", "evaluate failed")),
        Case("puppeteer_click", {"selector": "#b"},
             _expect(lambda g: "clicked" in g, "click failed")),
        Case("puppeteer_fill", {"selector": "#i", "value": "z"},
             _expect(lambda g: "filled" in g, "fill failed")),
        Case("puppeteer_select", {"selector": "#s", "value": "a"},
             _expect(lambda g: "selected" in g, "select failed")),
        Case("puppeteer_hover", {"selector": "#t"},
             _expect(lambda g: "hovered" in g, "hover failed")),
        Case("puppeteer_screenshot", {"name": "eval_shot2"},
             _expect(lambda g: "bytes" in g, "no screenshot")),
        Case("get_codegen_session", {"sessionId": "codegen-001"},
             _expect(lambda g: "step" in g, "no recorded steps")),
        Case("clear_codegen_session", {"sessionId": "codegen-001"},
             _expect(lambda g: "cleared" in g, "clear failed")),
        Case("end_codegen_session", {"sessionId": "codegen-001"},
             _expect(lambda g: "codegen-001" in g, "end failed")),
    ]


def search_cases() -> list[Case]:
    """Tavily's four operations, served without Tavily."""
    return [
        Case("tavily-search", {"query": "capital of South Korea", "max_results": 3},
             _expect(lambda g: "seoul" in g.lower(), "search found no Seoul")),
        Case("tavily-extract", {"urls": ["https://example.com"]},
             _expect(lambda g: "Example Domain" in g, "extraction found no text")),
        Case("tavily-crawl", {"url": "https://example.com", "limit": 3},
             _expect(lambda g: "page(s) from" in g, "crawl returned nothing")),
        Case("tavily-map", {"url": "https://www.iana.org/", "limit": 10},
             _expect(lambda g: "link(s)" in g or "sitemap" in g, "map returned nothing")),
    ]


def maps_cases() -> list[Case]:
    seoul = "Seoul, South Korea"
    here = {"latitude": CTX["lat"], "longitude": CTX["lon"]}
    in_seoul = _expect(_in_seoul, "coords outside Seoul")
    # Nominatim answers in the local language, so the city is "서울특별시" here.
    names_seoul = _expect(lambda g: "seoul" in g.lower() or "서울" in g,
                          "answer never mentions Seoul")
    return [
        Case("maps_geocode", {"address": seoul}, in_seoul),
        Case("map_geocode", {"address": seoul}, in_seoul),
        Case("maps_geo", {"address": seoul}, in_seoul),
        Case("maps_reverse_geocode", here, names_seoul),
        Case("map_reverse_geocode", here, names_seoul),
        Case("maps_regeocode", {"location": f"{CTX['lon']},{CTX['lat']}"}, names_seoul),
        Case("maps_search_places", {"query": "coffee", "city": seoul}, in_seoul),
        Case("map_search_places", {"keywords": "library", "city": seoul}, in_seoul),
        Case("maps_text_search", {"query": "Gyeongbokgung Palace"}, in_seoul),
        Case("maps_around_search", {"query": "park", "location": seoul}, in_seoul),
        Case("maps_place_details", {"place_id": "Namsan Tower Seoul"}, names_seoul),
        Case("map_place_details", {"id": "Seoul Station"}, names_seoul),
        Case("maps_search_detail", {"place_id": "Incheon Airport"},
             _expect(lambda g: len(g) > 10, "empty details")),
        Case("maps_directions", {"origin": "Seoul Station", "destination": "Incheon Airport"},
             _expect(lambda g: "km" in g and "min" in g, "no route")),
        Case("map_directions", {"origin": "Seoul Station", "destination": "Namsan Tower"},
             _expect(lambda g: "km" in g, "no route")),
        Case("maps_direction_driving", {"origin": "Seoul Station", "destination": "Suwon Station"},
             _expect(lambda g: "driving" in g, "not a driving route")),
        Case("maps_direction_walking", {"origin": "Seoul Station", "destination": "Namdaemun"},
             _expect(lambda g: "walking" in g, "not a walking route")),
        Case("maps_direction_bicycling", {"origin": "Seoul Station", "destination": "Yeouido"},
             _expect(lambda g: "bicycling" in g, "not a cycling route")),
        Case("maps_distance_matrix", {"origins": ["Seoul Station"],
                                      "destinations": ["Busan Station"]},
             _expect(lambda g: "km" in g, "no distance")),
        # Two destinations on purpose: OSRM wants ';' between indices, and a
        # single-destination matrix has no separator to get wrong.
        Case("map_directions_matrix", {"origins": ["Seoul Station"],
                                       "destinations": ["Daejeon Station", "Dongdaegu Station"]},
             _expect(lambda g: g.count("km") == 2, "matrix lost a destination")),
        Case("maps_distance", {"origins": ["Seoul"], "destinations": ["Incheon"]},
             _expect(lambda g: "km" in g, "no distance")),
        Case("maps_elevation", {"locations": [f"{CTX['lat']},{CTX['lon']}"]},
             _expect(lambda g: g.endswith("m"), "no elevation")),
        Case("map_weather", {"city": seoul}, _expect(lambda g: "°C" in g, "no temperature")),
        Case("maps_weather", {"city": "Busan"}, _expect(lambda g: "°C" in g, "no temperature")),
        Case("map_ip_location", {}, _expect(lambda g: "lat" in g, "no ip location")),
        Case("maps_ip_location", {}, _expect(lambda g: "lat" in g, "no ip location")),
        Case("maps_schema_navi", {"destination": "Gangnam Station"},
             _expect(lambda g: g.startswith("androidamap://navi"), "bad deep link")),
        Case("maps_schema_take_taxi", {"destination": "Hongdae"},
             _expect(lambda g: "OnRideNavi" in g, "bad deep link")),
        Case("map_mark", {"location": seoul, "name": "home"},
             _expect(lambda g: "viewMap" in g, "bad deep link")),
        Case("maps_schema_personal_map", {},
             _expect(lambda g: g.startswith("androidamap://myMap"), "bad deep link")),
    ]


def fs_cases() -> list[Case]:
    body = "line one\nline two\nline three\n"
    return [
        Case("list_allowed_directories", {},
             _expect(lambda g: g == adapters_fs.ROOT, "wrong sandbox root")),
        Case("create_directory", {"path": "proj/src"},
             _expect(lambda g: "created" in g, "mkdir failed")),
        Case("write_file", {"path": "proj/src/a.txt", "content": body},
             _expect(lambda g: "wrote" in g, "write failed")),
        Case("read_file", {"path": "proj/src/a.txt"},
             _expect(lambda g: g == body, "read did not return what was written")),
        Case("read_file", {"path": "proj/src/a.txt", "head": 1},
             _expect(lambda g: g == "line one\n", "head ignored")),
        Case("read_file", {"path": "proj/src/a.txt", "tail": 1},
             _expect(lambda g: g == "line three\n", "tail ignored")),
        Case("write_file", {"path": "proj/src/b.txt", "content": "bee"},
             _expect(lambda g: "wrote" in g, "write failed")),
        Case("read_multiple_files", {"paths": ["proj/src/a.txt", "proj/src/b.txt"]},
             _expect(lambda g: "bee" in g and "line one" in g, "did not read both")),
        Case("edit_file", {"path": "proj/src/b.txt",
                           "edits": [{"oldText": "bee", "newText": "cee"}]},
             _expect(lambda g: "applied" in g, "edit failed")),
        Case("read_file", {"path": "proj/src/b.txt"},
             _expect(lambda g: g == "cee", "edit did not persist")),
        Case("list_directory", {"path": "proj/src"},
             _expect(lambda g: "a.txt" in g and "b.txt" in g, "listing incomplete")),
        Case("list_directory_with_sizes", {"path": "proj/src"},
             _expect(lambda g: "bytes" in g, "no sizes")),
        Case("directory_tree", {"path": "proj"},
             _expect(lambda g: "src" in g, "tree missing subdir")),
        Case("search_files", {"path": ".", "pattern": "*.txt"},
             _expect(lambda g: "a.txt" in g, "search found nothing")),
        Case("get_file_info", {"path": "proj/src/a.txt"},
             _expect(lambda g: "type: file" in g, "bad file info")),
        # Climbing out of the sandbox must raise, not serve /etc/passwd.
        Case("read_file", {"path": "../../../etc/passwd"}, None,
             "guard: sandbox escape", RAISES),
    ]


def live_cases() -> list[Case]:
    return [
        Case("get_stock_price_global_market", {"company": "Apple"},
             _expect(lambda g: bool(re.search(r"\d", g)), "no price")),
        Case("get_stock_price_global_market", {"company": "Samsung Electronics"},
             _expect(lambda g: bool(re.search(r"\d", g)), "no price")),
        Case("search", {"query": "who wrote Dune"},
             _expect(lambda g: len(g) > 15, "empty search")),
        Case("search", {"query": "capital of South Korea"},
             _expect(lambda g: "seoul" in g.lower(), "wrong answer")),
        # Three SGD tools whose catalog knows only US cities on 2019 dates. They are
        # answered by keyless live sources now, so they belong here, not in the world.
        Case("Weather_1_GetWeather", {"city": "Seoul", "date": "2026-07-09"},
             _expect(lambda g: "°C" in g, "no temperature")),
        Case("Music_3_LookupMusic", {"artist": "아이유"},
             _expect(lambda g: "IU" in g or "아이유" in g, "no Korean artist")),
        Case("Travel_1_FindAttractions", {"location": "경주"},
             _expect(lambda g: "m)" in g, "no attractions near Gyeongju")),
        # Four more SGD reads moved off the seeded catalog onto OpenStreetMap. Each must
        # come back with a distance, which only real POI data can produce. Without these
        # cases `coverage` would call them with empty args — they are no longer `world`
        # tools, so nothing else synthesizes arguments for them.
        Case("Restaurants_2_FindRestaurants", {"category": "Korean", "location": "선릉역"},
             _expect(lambda g: "m)" in g and _in_seoul(g), "no Korean restaurants nearby")),
        Case("Hotels_4_SearchHotel", {"location": "제주"},
             _expect(lambda g: "m)" in g, "no hotels on Jeju")),
        Case("Hotels_2_SearchHouse", {"where_to": "부산"},
             _expect(lambda g: "m)" in g, "no guesthouses in Busan")),
        Case("Services_1_FindProvider", {"city": "선릉역"},
             _expect(lambda g: "m)" in g and _in_seoul(g), "no salons near Seolleung")),
    ]


def chained_cases() -> list[Case]:
    """The claim this package exists to make true: an ACTION is visible later."""

    def alarm_roundtrip(reg, _got):
        reg.dispatch("Alarm_1_AddAlarm",
                     {"new_alarm_time": "06:15", "new_alarm_name": "Roundtrip"})
        back = reg.dispatch("Alarm_1_GetAlarms", {})
        if "Roundtrip" not in back or "06:15" not in back:
            return f"GetAlarms did not see the alarm just added: {back!r}"
        return None

    def reservation_persists(reg, _got):
        got = reg.dispatch("Restaurants_2_ReserveRestaurant",
                           {"restaurant_name": "Sino", "location": "San Jose",
                            "time": "19:00", "date": "2019-03-01"})
        ids = [p for p in got.split(", ") if p.startswith("RES2-")]
        if not ids:
            return f"no confirmation id: {got!r}"
        conn = sqlite3.connect(reg.world.db_path)
        row = conn.execute("SELECT record FROM txn WHERE record LIKE ?",
                           (f"%{ids[0]}%",)).fetchone()
        conn.close()
        return None if row else "reservation is not in the transaction log"

    def missing_slot_rejected(reg, _got):
        try:
            reg.dispatch("Restaurants_2_ReserveRestaurant", {"location": "San Jose"})
        except WorldError:
            return None
        return "a reservation without a restaurant name was accepted"

    def unknown_city_abstains(reg, _got):
        got = reg.dispatch("Restaurants_2_FindRestaurants",
                           {"category": "Asian", "location": "Atlantis"})
        return None if got == "(no information found)" else f"invented a result: {got!r}"

    return [
        Case("Alarm_1_AddAlarm", {}, alarm_roundtrip, "chained: add -> get", SELF),
        Case("Restaurants_2_ReserveRestaurant", {}, reservation_persists,
             "chained: reserve -> log", SELF),
        Case("Restaurants_2_ReserveRestaurant", {}, missing_slot_rejected,
             "guard: missing slot", SELF),
        Case("Restaurants_2_FindRestaurants", {}, unknown_city_abstains,
             "guard: abstain", SELF),
    ]


# ----- runner -----------------------------------------------------------------
def run(cases, reg):
    passed, failures = 0, []
    for case in cases:
        try:
            if case.mode == RAISES:
                reg.dispatch(case.function, case.args)
                failures.append((case.function, case.note, "call was allowed to succeed"))
                continue
            problem = (case.check(reg, None) if case.mode == SELF
                       else case.check(reg, reg.dispatch(case.function, case.args)))
        except Exception as exc:
            if case.mode == RAISES:
                passed += 1
                continue
            label = "UNSUPPORTED " if isinstance(exc, Unsupported) else ""
            failures.append((case.function, case.note,
                             f"{label}{type(exc).__name__}: {str(exc)[:70]}"))
            continue
        if case.mode == RAISES:
            passed += 1
        elif problem:
            failures.append((case.function, case.note, problem))
        else:
            passed += 1
    return passed, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgd-dir", default=os.path.join(DATA, "moshicp", "_sgd_src"))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offline", action="store_true", help="skip maps/live network cases")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    adapters_fs.reset_sandbox()
    reg = get_registry()

    tally = collections.Counter()
    net = [] if args.offline else (
        maps_cases() + abroad_cases() + live_cases() + search_cases())
    # A machine without playwright keeps the 32 browser tools honestly unsupported,
    # so the cases that drive them only exist where a browser does.
    if net and reg.backend_of("playwright_navigate") == "browser":
        net += browser_cases()
    fixed = fs_cases() + chained_cases() + net
    budget = max(0, args.n - len(fixed))
    cases = sgd_cases(args.sgd_dir, reg, rng, budget, tally) + fixed
    cases = cases[: args.n]

    print(f"[eval] {len(cases)} cases over {len(reg.supported())} supported tools "
          f"({len(reg.bank)} in bank)")
    print(f"cases by backend: "
          f"{dict(collections.Counter(reg.backend_of(c.function) for c in cases))}\n")

    passed, failures = run(cases, reg)
    print(f"PASS {passed}   FAIL {len(failures)}   "
          f"({100 * passed / max(1, len(cases)):.1f}%)")
    if tally:
        print(f"\nSGD reads: {tally['strict']} judged strictly (query pins one entity), "
              f"{tally['underdetermined']} under-determined "
              f"(of which {tally['also_exact']} still matched SGD's exact row)")

    if failures:
        print("\n--- failures ---")
        grouped = collections.Counter(f[0] for f in failures)
        for function, count in grouped.most_common(25):
            sample = next(f for f in failures if f[0] == function)
            print(f"  {function:34s} x{count:<3d} {sample[2][:78]}")

    skipped = [f for f in reg.bank if reg.backend_of(f) == "unsupported"]
    print(f"\nskipped (no honest backend): {len(skipped)}")
    for reason, count in collections.Counter(
        reg.unsupported_reason(f) for f in skipped
    ).most_common():
        print(f"  {count:3d}  {reason}")


if __name__ == "__main__":
    main()
