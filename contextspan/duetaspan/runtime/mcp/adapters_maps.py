"""Real map tools over keyless open services.

The bank carries 32 map names because it merged two vendors: Google Maps
(``maps_geocode``) and Amap (``map_geocode``, ``maps_regeocode``). Both want API
keys we do not have, and both are thin wrappers over the same handful of
operations. So there are ~13 real implementations here, backed by open services
that need no key, and the bank's names are aliased onto them.

  geocode / reverse / search / details  -> OpenStreetMap Nominatim
  directions / distance matrix          -> OSRM public router
  elevation                             -> Open-Elevation
  weather                               -> open-meteo
  ip location                           -> ipapi.co

``map_road_traffic`` and ``maps_direction_transit_integrated`` have no keyless
source and are deliberately absent; the registry reports them unsupported rather
than inventing a congestion level or a bus timetable.

Nominatim's usage policy requires an identifying User-Agent and at most one
request per second — both are enforced here.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
import urllib.parse
from typing import Any

import requests

from contextspan.duetaspan.runtime.mcp import cache, geo_index

_UA = {"User-Agent": "DuetaSpan-MCP/1.0 (research)"}

# Every one of these hosts is in Europe. A fresh TCP+TLS handshake per call costs about
# 800 ms of the ~1030 ms an open-meteo request takes from here; on a reused connection the
# same request is 239 ms. The server is long-lived, so keep the connections.
_http = requests.Session()
_http.headers.update(_UA)
_NOMINATIM = "https://nominatim.openstreetmap.org"
_ELEV = "https://api.open-elevation.com/api/v1/lookup"
_METEO = "https://api.open-meteo.com/v1/forecast"
_IPAPI = "https://ipapi.co/json/"
# overpass.kumi.systems used to sit here as a fallback. It is unreachable from this
# network — every call read-timed out at 60 s — so it turned each failover on the primary
# into a minute of waiting for nothing. A fallback that cannot answer is not a fallback.
_OVERPASS = ("https://overpass-api.de/api/interpreter",)
_TRANSITOUS = "https://api.transitous.org/api/v1/plan"


def _now_utc() -> str:
    """Floored to the minute so that two calls a few seconds apart share a cache key.
    A transit plan departing 14:23:00 is the plan departing 14:23:40."""
    return time.strftime("%Y-%m-%dT%H:%M:00Z", time.gmtime())


def warm_connections(block: float = 0.0) -> None:
    """Open the TLS connections now so the first real question does not pay for them.

    The handshake is ~800 ms of a ~1050 ms first request; every later request on the same
    connection is ~240 ms. Open-Elevation is worse — its first call took 9.7 s and its
    second 226 ms, which is what made `maps_elevation` look like a broken tool.

    `block` waits that many seconds for the handshakes to land. A server should wait, since
    startup happens before anyone asks anything; a test importing the module should not.
    Silent on failure: a warm-up that cannot reach the network must not stop the server.
    """
    def _open(url: str, params: dict | None) -> None:
        try:
            _http.get(url, params=params, timeout=10)
        except requests.RequestException:
            pass

    # A real GET, not a HEAD. Several of these hosts answer HEAD with 405 and
    # `Connection: close`, which tears down the connection the warm-up just opened —
    # `maps_elevation` still paid 3.3 s after a "successful" HEAD warm-up.
    #
    # One entry per distinct host: FOSSGIS serves each routing profile from its own path
    # but a single host, and the pool is keyed by host.
    probes: list[tuple[str, dict | None]] = [
        (_NOMINATIM + "/status.php", None),
        (_METEO, {"latitude": 37.5, "longitude": 127.0, "current": "temperature_2m"}),
        (_ELEV, {"locations": "37.5,127.0"}),
        (_IPAPI, None),
        (f"{_FOSSGIS}/routed-car/route/v1/driving/126.98,37.56;126.99,37.57",
         {"overview": "false"}),
        # Transitous wants a real plan to hold the connection open: warming its root left
        # the first journey at 693 ms, warming a plan leaves it at 498 ms.
        (_TRANSITOUS, {"fromPlace": "37.50448,127.04904",
                       "toPlace": "37.51025,127.04386", "time": _now_utc()}),
    ]
    # Kakao is the first geocoder now, so its cold handshake (~525 ms) is on the critical
    # path. Warm it too. No auth header — a 401 still completes the TLS handshake that the
    # first real, authenticated call would otherwise pay for.
    if _KAKAO_KEY:
        probes.append((_KAKAO_ADDR, {"query": "서울시청"}))
    threads = [threading.Thread(target=_open, args=probe, daemon=True) for probe in probes]
    for thread in threads:
        thread.start()
    if block <= 0:
        return
    deadline = time.monotonic() + block
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))

# Optional. Without it, radius search still works from a station or a landmark;
# with it, bare administrative-dong names ("성수동") resolve too. Free tier, 100k calls/day.
# Two endpoints: address.json geocodes lot-number/road-name addresses ("역삼동 640-1",
# "테헤란로 152"), keyword.json resolves place names and POIs ("성수동", "남산서울타워",
# "강남역 스타벅스").
_KAKAO_SEARCH = "https://dapi.kakao.com/v2/local/search/keyword.json"
_KAKAO_ADDR = "https://dapi.kakao.com/v2/local/search/address.json"
_KAKAO_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
_HANGUL = re.compile(r"[가-힣]")

# What a Korean actually says -> what OpenStreetMap calls it. Order matters:
# the first entry whose word appears in the query wins, so put "한식" before
# the generic "식당" or every Korean-restaurant query collapses to plain restaurants.
_POI_TAGS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("주유소", "fuel", "gas station", "petrol"), '["amenity"="fuel"]'),
    (("충전소", "ev charg"), '["amenity"="charging_station"]'),
    (("카페", "커피", "cafe", "coffee"), '["amenity"="cafe"]'),
    (("편의점", "convenience"), '["shop"="convenience"]'),
    (("약국", "pharmacy"), '["amenity"="pharmacy"]'),
    (("병원", "hospital"), '["amenity"="hospital"]'),
    (("의원", "clinic"), '["amenity"="clinic"]'),
    (("은행", "bank"), '["amenity"="bank"]'),
    (("현금", "atm"), '["amenity"="atm"]'),
    (("한식", "korean"), '["amenity"="restaurant"]["cuisine"~"korean",i]'),
    (("식당", "음식점", "맛집", "restaurant"), '["amenity"="restaurant"]'),
    (("술집", "호프", "pub", "bar"), '["amenity"="bar"]'),
    (("공원", "park"), '["leisure"="park"]'),
    (("도서관", "library"), '["amenity"="library"]'),
    (("주차장", "parking"), '["amenity"="parking"]'),
    (("화장실", "toilet"), '["amenity"="toilets"]'),
    (("미용실", "헤어", "hairdresser", "salon"), '["shop"="hairdresser"]'),
    (("지하철", "전철", "subway", "metro"), '["railway"="station"]'),
    (("버스정류장", "bus stop"), '["highway"="bus_stop"]'),
    (("호텔", "hotel"), '["tourism"="hotel"]'),
    (("숙소", "게스트하우스", "guesthouse", "hostel"), '["tourism"~"hostel|guest_house"]'),
    (("관광", "명소", "attraction", "sightsee"), '["tourism"~"attraction|museum|viewpoint"]'),
    (("학교", "school"), '["amenity"="school"]'),
    (("마트", "슈퍼", "supermarket"), '["shop"="supermarket"]'),
    (("영화관", "극장", "cinema"), '["amenity"="cinema"]'),
    (("우체국", "post office"), '["amenity"="post_office"]'),
    (("경찰", "police"), '["amenity"="police"]'),
)

# OSRM's own demo host (router.project-osrm.org) serves the car profile and nothing
# else: ask it for `foot` and it answers with the car route under a walking label —
# 0.69 km in "1 min" on a walk that takes nine. FOSSGIS runs one OSRM instance per
# profile, so ask the right host instead of the right path. Same 0.69 km comes back as
# foot 9 min, bike 3 min. The demo host is also unreliable (2 of 3 calls read-timed out
# at 20 s where FOSSGIS answered 3 of 3), so the car profile lives here too.
_FOSSGIS = "https://routing.openstreetmap.de"
_ROUTERS = {
    "driving": (f"{_FOSSGIS}/routed-car", "driving"),
    "walking": (f"{_FOSSGIS}/routed-foot", "foot"),
    "bicycling": (f"{_FOSSGIS}/routed-bike", "bike"),
}
_PROFILE = {mode: profile for mode, (_host, profile) in _ROUTERS.items()}

# Soft geocoding bias: the user lives in South Korea (lon,lat,lon,lat corners).
_HOME_VIEWBOX = os.environ.get("MOSHICP_MCP_VIEWBOX", "124.5,33.0,131.0,38.7")

_throttle = threading.Lock()
_last_call = [0.0]

_overpass_gate = threading.Lock()
_overpass_last = [0.0]


class MapError(Exception):
    pass


_RETRY_STATUS = {429, 500, 502, 503, 504}

# Realtime router path: cap every remote map call to this many seconds and a single
# attempt, so a cold foreign query degrades to an honest "(no information found)"
# within the voice-assistant delay budget instead of retrying for a minute. Unset
# (the default) leaves the quality-first datagen behaviour untouched.
_FAST_BUDGET = float(os.environ.get("MCP_MAPS_BUDGET_S", "0") or 0)


def _get(url: str, params: dict | None = None, throttle: bool = False,
         attempts: int = 3, ttl: float = cache.FOREVER) -> Any:
    """GET with backoff, answered from disk when the answer cannot have changed.

    These are free public endpoints: they rate-limit and they blip. One 503 from
    open-meteo must not read as "the tool is broken".

    The cache is read before the throttle, not after. Nominatim's 1.05 s gate is a
    promise to Nominatim about how often this process will *call* it — a cache hit
    calls nobody, so sleeping through the gate would burn a second to honour a promise
    that is not being tested.
    """
    hot = cache.key("GET", url, params)
    stored = cache.get(hot)
    if stored is not None:
        return stored

    request_timeout = 20.0
    if _FAST_BUDGET:
        attempts, request_timeout = 1, _FAST_BUDGET
    last = ""
    for attempt in range(attempts):
        if throttle:                   # Nominatim: >= 1s between requests
            with _throttle:
                wait = 1.05 - (time.monotonic() - _last_call[0])
                if wait > 0:
                    time.sleep(wait)
                _last_call[0] = time.monotonic()
        try:
            resp = _http.get(url, params=params, timeout=request_timeout)
        except requests.RequestException as exc:
            last = str(exc)
        else:
            if resp.status_code == 200:
                payload = resp.json()
                cache.put(hot, payload, ttl)
                return payload
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in _RETRY_STATUS:
                break
        if attempt < attempts - 1:
            time.sleep(0.6 * (attempt + 1))
    raise MapError(f"{url} -> {last}")


def _search(address: str, limit: int, biased: bool) -> list[dict]:
    params: dict[str, Any] = {"q": address, "format": "jsonv2", "limit": limit}
    if biased:
        params.update(viewbox=_HOME_VIEWBOX, bounded=0)
    return _get(_NOMINATIM + "/search", params, throttle=True)


def _place(address: str) -> dict:
    """Geocode to the best hit, or raise. Shared by every route/detail tool."""
    return _places(address, limit=1)[0]


# ----- implementations --------------------------------------------------------
def geocode(address: str = "", **_: Any) -> str:
    hit = _place(address)
    return f"{hit['display_name']}, lat {float(hit['lat']):.5f}, lon {float(hit['lon']):.5f}"


def reverse_geocode(latitude: float = None, longitude: float = None,
                    location: str = "", **_: Any) -> str:
    if location and latitude is None:          # amap passes "lon,lat"
        lon, _, lat = location.partition(",")
        latitude, longitude = float(lat), float(lon)
    if latitude is None or longitude is None:
        raise MapError("reverse geocode needs latitude and longitude")
    hit = _get(_NOMINATIM + "/reverse",
               {"lat": latitude, "lon": longitude, "format": "jsonv2"}, throttle=True)
    if "display_name" not in hit:
        raise MapError("no address at those coordinates")
    return hit["display_name"]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle metres. Good enough to rank POIs inside one neighbourhood."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * 6371000 * math.asin(math.sqrt(a))


def _overpass(clauses: list[str], lat: float, lon: float, radius: int) -> list[dict]:
    """POI-by-tag inside a radius."""
    body = "".join(
        f"node{c}(around:{radius},{lat},{lon});way{c}(around:{radius},{lat},{lon});"
        for c in clauses
    )
    return _overpass_raw(f"[out:json][timeout:25];({body});out center 40;")


def _overpass_raw(query: str, attempts: int = 3, timeout: int = 12) -> list[dict]:
    """Run one Overpass QL query. Overpass hands out query slots and answers 429 when
    they run out, so serialise calls and back off rather than fail the tool.

    Checked against the cache first: the 1.5 s gate exists to protect Overpass from this
    process, and a hit never reaches Overpass."""
    hot = cache.key("overpass", query)
    stored = cache.get(hot)
    if stored is not None:
        return stored

    if _FAST_BUDGET:
        attempts, timeout = 1, _FAST_BUDGET
    last = ""
    for attempt in range(attempts):
        for endpoint in _OVERPASS:
            with _overpass_gate:
                wait = 1.5 - (time.monotonic() - _overpass_last[0])
                if wait > 0:
                    time.sleep(wait)
                _overpass_last[0] = time.monotonic()
            try:
                # 12 s by default, not 60. The old fallback endpoint was unreachable
                # from some networks, so every failover paid a minute for nothing.
                # Callers whose query carries a longer server-side [timeout:N] must
                # raise this too, or the client hangs up before the server answers.
                resp = _http.post(endpoint, data={"data": query},
                                  timeout=timeout)
            except requests.RequestException as exc:
                last = type(exc).__name__
                continue
            if resp.status_code == 200:
                elements = resp.json().get("elements", [])
                cache.put(hot, elements, cache.FOREVER)
                return elements
            last = f"HTTP {resp.status_code}"
        if attempt < attempts - 1:
            time.sleep(2.0 * (attempt + 1))
    raise MapError(f"overpass unavailable: {last}")


def _clauses(keyword: str) -> list[str]:
    """OSM tag filters for a search word, else a name substring match.

    Nominatim cannot answer "petrol stations near Seolleung": it geocodes names,
    it does not index what a place *is*. Overpass does, which is the question
    people actually ask ("근처 주유소 찾아줘").
    """
    word = keyword.strip().lower()
    for names, tag in _POI_TAGS:
        if any(name in word for name in names):
            return [tag]
    escaped = keyword.replace('"', '\\"')
    return [f'["name"~"{escaped}",i]']


def _kakao(query: str, x: str = "", y: str = "", radius: int = 0) -> list[dict]:
    """Kakao Local keyword search. Only reachable when a REST key is configured.

    OSM has no node called "성수동" — Seoul's neighbourhood is filed as 성수동1가 /
    성수동2가, so Nominatim resolves the bare name to a road in Jeju. Kakao carries
    the administrative-dong gazetteer, which is the one thing the keyless stack cannot fake.
    """
    params: dict[str, Any] = {"query": query, "size": 10}
    if x and y:
        params.update(x=x, y=y, radius=min(radius, 20000), sort="distance")
    hot = cache.key("kakao", params)          # keyed on params, never on the REST key
    stored = cache.get(hot)
    if stored is not None:
        return stored
    resp = _http.get(_KAKAO_SEARCH, params=params, timeout=20,
                     headers={"Authorization": f"KakaoAK {_KAKAO_KEY}"})
    if resp.status_code != 200:
        raise MapError(f"kakao -> HTTP {resp.status_code}")
    documents = resp.json().get("documents") or []
    cache.put(hot, documents, cache.DAY)      # shops open and close; streets do not
    return documents


def _kakao_nearby(centre: str, keyword: str, radius: int) -> tuple[float, float, list]:
    # Anchor via the address-first geocoder, NOT a bare keyword search. `_kakao("일산")`
    # keyword-matches a *business literally named 일산* 335 km away, so a radius search around
    # it found no Costco and the tool spent 13 s in Overpass returning nothing. `_kakao_geocode`
    # resolves 일산 to the district (address.json) first, so the anchor is the real 일산.
    anchor = _kakao_geocode(centre)
    if not anchor:
        raise MapError(f"no such place: {centre}")
    lat, lon = float(anchor["lat"]), float(anchor["lon"])
    hits = _kakao(keyword, x=anchor["lon"], y=anchor["lat"], radius=radius)
    if not hits:
        # A sparse brand ("코스트코 near 일산") sits a few km past a tight radius. Ask Kakao
        # directly, ranked by its own relevance, then rank by real distance from the anchor.
        hits = _kakao(f"{centre} {keyword}")
    found = sorted((_haversine(lat, lon, float(h["y"]), float(h["x"])), h["place_name"])
                   for h in hits)
    return lat, lon, found


def _kakao_addr(query: str) -> list[dict]:
    """Kakao's address geocoder — lot-number and road-name, what a name gazetteer cannot do.

    "역삼동 640-1" and "테헤란로 152" carry a building number; the local OSM gazetteer and
    Nominatim both stall or miss on those, while Kakao answers in ~160 ms nationwide."""
    params = {"query": query}
    hot = cache.key("kakao_addr", params)      # keyed on the query, never on the REST key
    stored = cache.get(hot)
    if stored is not None:
        return stored
    resp = _http.get(_KAKAO_ADDR, params=params, timeout=20,
                     headers={"Authorization": f"KakaoAK {_KAKAO_KEY}"})
    if resp.status_code != 200:
        raise MapError(f"kakao addr -> HTTP {resp.status_code}")
    documents = resp.json().get("documents") or []
    cache.put(hot, documents, cache.DAY)
    return documents


def _kakao_geocode(address: str) -> dict | None:
    """Kakao as the primary Korean geocoder: ~160 ms warm, nationwide, and it answers the
    things the keyless stack cannot — lot-number, road-name and bare dong names alike.

    Hangul only, because Kakao is Korea-only: letting it field "Times Square" would pull a
    Seoul cafe of that name over Manhattan. A miss or any error returns None so the caller
    falls through to the local index and Nominatim — Kakao speeds this path, never gates it.
    Address search first (precise for lot-number/road-name addresses), then keyword search
    (place names/POIs).
    """
    if not (_KAKAO_KEY and _HANGUL.search(address)):
        return None
    try:
        docs = _kakao_addr(address) or _kakao(address)
    except (MapError, requests.RequestException):
        return None
    for doc in docs:
        return {"lat": doc["y"], "lon": doc["x"],
                "display_name": doc.get("address_name") or doc.get("place_name") or address,
                "addresstype": "address", "importance": 0.0}
    return None


# A road named after a neighbourhood is a bad centre for "what is nearby".
# `borough` belongs here: Nominatim files 강남구 as one. Leaving it out told `_places`
# that a perfectly good hit was not place-like, which sent 강남구 down the Overpass
# name-scan path — 41 s of work to rediscover an answer already in hand.
_PLACE_TYPES = ("suburb", "quarter", "neighbourhood", "city_district", "borough",
                "city", "town", "village", "station", "railway", "amenity", "tourism")

# Nominatim's importance is a Wikipedia-derived notability score. Manhattan's
# Times Square scores 0.599; a colloquial name it has never heard of scores 0.000.
_IMPORTANCE_FLOOR = 0.35


# Only an administrative name is worth a second Overpass round-trip: "성수동" yes,
# "선릉역" no (Nominatim has the station, it just files it as a highway).
_ADMIN_NAME = re.compile(r"[가-힣]+(동|가|읍|면|리|구|시|군)$")
# Names Seoul files in numbered pieces — 성수동 is stored as 성수동1가 / 성수동2가 — so an
# exact-name lookup misses and only a prefix scan finds them. Districts and cities are
# not split, so a miss on those means the place simply is not there.
_SPLIT_NAME = re.compile(r"[가-힣]+(동|가|읍|면|리)$")

# Overpass returns tags, not Nominatim's shape. Normalise so both can be ranked together.
_HOME_BBOX = "33.0,124.5,38.7,131.0"          # south,west,north,east


def _named_query(name: str, match: str, seconds: int, attempts: int = 3) -> list[dict]:
    """One Overpass lookup for a place called `name`, matched by `match`."""
    query = (f'[out:json][timeout:{seconds}];'
             f'(node["name"{match}]["place"]({_HOME_BBOX});'
             f' way["name"{match}]["place"]({_HOME_BBOX});'
             f' relation["name"{match}]["boundary"="administrative"]({_HOME_BBOX}););'
             f'out center 5;')
    try:
        elements = _overpass_raw(query, attempts=attempts, timeout=seconds + 6)
    except MapError:
        return []
    found = []
    for element in elements:
        point = element.get("center") or element
        tags = element.get("tags") or {}
        if "lat" not in point:
            continue
        found.append({"lat": str(point["lat"]), "lon": str(point["lon"]),
                      "display_name": tags.get("name", name),
                      "addresstype": tags.get("place", "suburb"),
                      "importance": 0.0})
    return found


def _named_places(name: str) -> list[dict]:
    """Korean place names Nominatim does not carry, looked up in OSM.

    Exact match first, because it reads the name index and answers in milliseconds.
    A prefix regex cannot use that index: `["name"~"^강남구"]` makes Overpass scan every
    named node in the country, which took 90 s and then timed out — the whole cost of
    `map_search_places`, for a lookup whose exact form returns instantly.

    The regex is still needed (Seoul files 성수동 as 성수동1가 / 성수동2가), so it stays as
    a fallback with a server-side timeout that bounds the damage.
    """
    escaped = name.replace('"', '\\"')
    if _SPLIT_NAME.search(name):
        # 성수동 is stored as 성수동1가 / 성수동2가, so its exact name matches nothing —
        # asking anyway cost 18 s to be told zero. Go straight to the prefix scan.
        #
        # 25 s, and one attempt. A prefix regex cannot use the name index, so Overpass
        # scans; capping it at 10 s made the server abort before it found 성수동1가 and
        # sent 성수동 back to a road in Jeju. Correctness first — the answer is then
        # cached for a month, so the wait is paid once, ever.
        return _named_query(name, f'~"^{escaped}"', 25, attempts=1)
    return _named_query(name, f'="{escaped}"', 10)


def _at_home(hit: dict) -> bool:
    lon_w, lat_s, lon_e, lat_n = (float(x) for x in _HOME_VIEWBOX.split(","))
    return (lat_s <= float(hit["lat"]) <= lat_n
            and lon_w <= float(hit["lon"]) <= lon_e)


def _places(address: str, limit: int = 10) -> list[dict]:
    """Every plausible geocode for a name, best candidate first.

    Searching the whole planet is the default, because a country bias silently
    drags "Times Square" from Manhattan to a Yeongdeungpo shopping mall — same
    name, importance 0.29 against Manhattan's 0.60.

    But Nominatim scores an unqualified local nickname at importance 0.000 and
    then hands back whatever shares the string: "Hongdae" resolves to a building
    in Buenos Aires. So when the planet-wide winner carries no notability at all,
    ask again with the home bias and keep whichever hit is actually notable.
    """
    # Kakao first when a key is configured: ~160 ms nationwide, and the only source here
    # that geocodes lot-number/road-name addresses with a building number ("역삼동 640-1",
    # "테헤란로 152"). It returns None for non-Korean names and on any error, so the keyless
    # path still runs.
    kakao = _kakao_geocode(address)
    if kakao:
        return [kakao]

    # The local gazetteer answers Korean names in microseconds and never touches the
    # network, which is the only way `directions` avoids paying Nominatim's 1 req/s gap
    # twice. It holds Hangul names only, so 성수동 stops here and Times Square does not.
    local = geo_index.place(address)
    if local:
        return [local]

    hits = _search(address, limit, biased=False)
    best = max((h.get("importance") or 0.0) for h in hits) if hits else 0.0
    # An unnotable hit that already landed at home needs no second opinion —
    # "선릉역" scores 0.000 too, and paying Nominatim's 1 req/s twice for it is waste.
    if best < _IMPORTANCE_FLOOR and not any(_at_home(h) for h in hits[:1]):
        hits += _search(address, limit, biased=True)
    # OSM files Seoul's 성수동 as 성수동1가 / 성수동2가, so Nominatim answers the bare name
    # with a *road* of that name in Jeju — at home, so no re-ranking rescues it. Overpass
    # matches the name as a prefix and knows the quarter. Only for Hangul names, and only
    # when nothing place-like turned up: "Times Square" must never enter this branch.
    if _ADMIN_NAME.search(address) and not any(
            h.get("addresstype") in _PLACE_TYPES for h in hits[:3]):
        hits += _named_places(address)
    if not hits:
        raise MapError(f"no such place: {address}")
    hits.sort(key=_rank)
    return hits


def _rank(hit: dict) -> tuple:
    """Notable hits rank by notability; unnotable ones rank by being a place at all.

    Jeju's 성수동 *road* scores importance 5.9e-05 — not fame, just float residue. Sorting
    on it put a country lane ahead of Seoul's 성수동1가 (importance exactly 0.0). Below the
    floor, notability carries no signal, so let place-likeness decide instead.
    """
    importance = hit.get("importance") or 0.0
    notable = importance if importance >= _IMPORTANCE_FLOOR else 0.0
    return (-notable, hit.get("addresstype") not in _PLACE_TYPES, -importance)


# One Overpass clause, read back as the tag it stands for. Keeping a second table of
# keyword -> (key, value) would let the two drift; this reads the table already there.
#   ["amenity"="fuel"]                             -> amenity=fuel
#   ["tourism"~"hostel|guest_house"]               -> tourism=hostel, tourism=guest_house
#   ["amenity"="restaurant"]["cuisine"~"korean",i] -> amenity=restaurant, cuisine korean
_TAG_PAIR = re.compile(r'\["(\w+)"[=~]"([^"]+)"(?:,i)?\]')


def _local_tag(keyword: str) -> tuple[str, list[str], str] | None:
    """The (key, values, cuisine) a keyword stands for, or None when the local index
    cannot express it — a bare name match has no tag to look up."""
    pairs = _TAG_PAIR.findall(_clauses(keyword)[0])
    if not pairs or pairs[0][0] == "name":
        return None
    key, value = pairs[0]
    cuisine = next((v for k, v in pairs[1:] if k == "cuisine"), "")
    return key, value.split("|"), cuisine


def _nearby(lat: float, lon: float, keyword: str, radius: int) -> list[tuple]:
    tag = _local_tag(keyword)
    if tag and geo_index.covers(lat, lon):
        key, values, cuisine = tag
        local: list[tuple] = []
        for value in values:
            local += geo_index.nearby(lat, lon, radius, key, value, cuisine)
        if local:
            return sorted(local)
        # An empty local answer is not proof of absence: the extract covers Seoul only and
        # the rarer tags are thin. Ask Overpass before reporting nothing.

    found = []
    for element in _overpass(_clauses(keyword), lat, lon, radius):
        tags = element.get("tags") or {}
        name = tags.get("name") or tags.get("brand") or tags.get("operator")
        point = element.get("center") or element
        if not name or "lat" not in point:
            continue
        found.append((_haversine(lat, lon, point["lat"], point["lon"]), name))
    return sorted(found)


def search_places(query: str = "", keywords: str = "", location: str = "",
                  city: str = "", radius: int = 1500, **_: Any) -> str:
    keyword = (query or keywords).strip()
    centre = (city or location).strip()
    if not keyword and not centre:
        raise MapError("search needs a query")
    if not centre:                     # "남산서울타워 어디야" — the word IS the place
        centre, keyword = keyword, ""

    # Kakao is Korea-only. Firing it on a foreign centre finds a spurious Korean namesake —
    # "Nikko" matches a 부산 restaurant of that name, so "things to see in Nikko" answered with
    # Busan. Gate on Hangul, exactly as _kakao_geocode does: foreign names go to Nominatim.
    if keyword and _KAKAO_KEY and _HANGUL.search(centre):
        lat, lon, found = _kakao_nearby(centre, keyword, radius)
        if found:
            return _render(centre, lat, lon, found)

    candidates = _places(centre)
    if not keyword:
        hit = candidates[0]
        return (f"{hit['display_name'].split(',')[0]} "
                f"({float(hit['lat']):.5f}, {float(hit['lon']):.5f})")

    # A bare dong/gu (neighbourhood/district) name is ambiguous nationwide — Nominatim
    # reads "성수동" as the one in Jeju. Walk the candidates until one actually has the
    # thing asked for, rather than hardcoding Seoul.
    for hit in candidates:
        lat, lon = float(hit["lat"]), float(hit["lon"])
        found = _nearby(lat, lon, keyword, radius)
        if found:
            return _render(centre, lat, lon, found)
    return "(no information found)"


def _render(centre: str, lat: float, lon: float, found: list[tuple]) -> str:
    """The search centre with its coordinates, then the nearest five hits.

    No language word joins the two halves: this tool answers "카페 near 선릉역" and
    "cafes near Times Square" from the same code path, and a Korean particle
    welded into an English answer would be a bug, not a nicety.

    The coordinates are not decoration either — they are the only way a caller
    can tell which "성수동", or which Springfield, actually answered.
    """
    seen, best = set(), []
    for metres, name in found:
        if name not in seen:
            seen.add(name)
            best.append(f"{name} ({metres:.0f} m)")
    return f"{centre} ({lat:.5f}, {lon:.5f}): " + ", ".join(best[:5])


def transit_directions(origin: str = "", destination: str = "", start: str = "",
                       **_: Any) -> str:
    """Public-transit routing over Transitous, a keyless public MOTIS instance.

    I called this impossible because OSRM has no transit profile. That was true and
    beside the point: OSRM is not the only router. Transitous aggregates published
    GTFS feeds, Seoul's among them, and answers 선릉역 -> 선정릉역 with the bus that
    actually runs it. No key, no self-hosted OpenTripPlanner.
    """
    if not origin or not destination:
        raise MapError("transit directions need an origin and a destination")
    src, dst = _place(origin), _place(destination)
    plan = _get(_TRANSITOUS, {
        "fromPlace": f"{src['lat']},{src['lon']}",
        "toPlace": f"{dst['lat']},{dst['lon']}",
        "time": start or _now_utc(),
    }, ttl=cache.MINUTE)
    trips = plan.get("itineraries") or []
    if not trips:
        return "(no information found)"

    trip = trips[0]
    steps = []
    for leg in trip.get("legs") or []:
        mode = (leg.get("mode") or "").lower()
        if mode == "walk":
            metres = round(leg.get("distance") or 0)
            steps.append(f"walk {metres} m")
            continue
        line = leg.get("routeShortName") or leg.get("routeLongName") or mode
        board = (leg.get("from") or {}).get("name", "?")
        alight = (leg.get("to") or {}).get("name", "?")
        steps.append(f"{mode} {line}: {board} -> {alight}")
    minutes = round((trip.get("duration") or 0) / 60)
    changes = max(0, sum(1 for s in steps if not s.startswith("walk")) - 1)
    return f"{minutes} min, {changes} transfer(s): " + "; ".join(steps)


def attractions(location: str = "", **_: Any) -> str:
    """SGD's Travel_1_FindAttractions, answered from OSM instead of a US catalog."""
    if not location:
        raise MapError("attractions need a location")
    return search_places(query="관광지", city=location, radius=6000)


# ---- SGD read tools, answered from real OSM data rather than a seeded catalog --------
# The world's rows for these came from SGD's US catalog and from seed_korea.py: invented
# names at invented prices. OpenStreetMap has the real restaurants, hotels and salons, and
# the local index carries them nationwide — no seeding, no key.
#
# Only the *reads* move. Reserving a table still records into the world, because doing
# that for real would book a real table.
_SGD_RADIUS = 3000

# SGD's category enum against the values OSM actually writes into `cuisine`.
_CUISINE = {"korean": "korean", "asian": "asian", "chinese": "chinese",
            "japanese": "japanese", "italian": "italian", "mexican": "mexican",
            "indian": "indian", "american": "american", "thai": "thai",
            "vietnamese": "vietnamese", "french": "french", "spanish": "spanish"}


def _osm_search(where: str, key: str, values: tuple[str, ...], cuisine: str = "",
                radius: int = _SGD_RADIUS) -> str:
    """Places carrying an OSM tag near `where`, nearest first: local index, else Overpass."""
    hit = _place(where)
    lat, lon = float(hit["lat"]), float(hit["lon"])

    found: list[tuple] = []
    if geo_index.covers(lat, lon):
        for value in values:
            found += geo_index.nearby(lat, lon, radius, key, value, cuisine)

    if not found:              # outside the extract, or a tag the extract does not carry
        suffix = f'["cuisine"~"{cuisine}",i]' if cuisine else ""
        clauses = [f'["{key}"="{value}"]{suffix}' for value in values]
        for element in _overpass(clauses, lat, lon, radius):
            tags = element.get("tags") or {}
            name = tags.get("name") or tags.get("brand")
            point = element.get("center") or element
            if name and "lat" in point:
                found.append((_haversine(lat, lon, point["lat"], point["lon"]), name))

    if not found:
        return "(no information found)"
    return _render(where, lat, lon, sorted(found))


def find_restaurants(category: str = "", location: str = "", **_: Any) -> str:
    if not location:
        raise MapError("restaurant search needs a location")
    return _osm_search(location, "amenity", ("restaurant",),
                       _CUISINE.get(category.strip().lower(), ""))


def search_hotel(location: str = "", **_: Any) -> str:
    if not location:
        raise MapError("hotel search needs a location")
    return _osm_search(location, "tourism", ("hotel",))


def search_house(where_to: str = "", **_: Any) -> str:
    if not where_to:
        raise MapError("house search needs a destination")
    return _osm_search(where_to, "tourism", ("guest_house", "hostel"))


def find_salon(city: str = "", **_: Any) -> str:
    if not city:
        raise MapError("salon search needs a city")
    return _osm_search(city, "shop", ("hairdresser",))


def place_details(place_id: str = "", id: str = "", **_: Any) -> str:
    target = place_id or id
    if not target:
        raise MapError("place details needs a place_id")
    hits = _get(_NOMINATIM + "/search",
                {"q": target, "format": "jsonv2", "limit": 1,
                 "extratags": 1, "addressdetails": 1}, throttle=True)
    if not hits:
        raise MapError(f"no such place: {target}")
    hit = hits[0]
    tags = hit.get("extratags") or {}
    bits = [hit["display_name"], f"type {hit.get('type', 'place')}"]
    for key in ("phone", "website", "opening_hours"):
        if tags.get(key):
            bits.append(f"{key} {tags[key]}")
    return ", ".join(bits)


def directions(origin: str = "", destination: str = "", mode: str = "driving",
               **_: Any) -> str:
    if not origin or not destination:
        raise MapError("directions need an origin and a destination")
    if mode not in _ROUTERS:
        raise MapError(f"unsupported travel mode: {mode}")
    host, profile = _ROUTERS[mode]
    start, end = _place(origin), _place(destination)
    route = _get(
        f"{host}/route/v1/{profile}/"
        f"{start['lon']},{start['lat']};{end['lon']},{end['lat']}",
        {"overview": "false"},
    )
    if route.get("code") != "Ok" or not route.get("routes"):
        raise MapError("no route found")
    leg = route["routes"][0]
    return f"{leg['distance'] / 1000:.1f} km, {round(leg['duration'] / 60)} min by {mode}"


def distance_matrix(origins: Any = None, destinations: Any = None,
                    mode: str = "driving", **_: Any) -> str:
    origins, destinations = _as_list(origins), _as_list(destinations)
    if not origins or not destinations:
        raise MapError("distance matrix needs origins and destinations")
    host, profile = _ROUTERS.get(mode, _ROUTERS["driving"])
    points = [_place(p) for p in origins + destinations]
    pairs = len(origins) * len(destinations)

    # OSRM's `table` service is quadratic and, on FOSSGIS, slow over long distances: a
    # 1x2 matrix from 서울역 to 대전역 and 동대구역 took 10.0 s, while the same two legs
    # asked of `route` took 0.22 s each. Below a handful of pairs, ask for routes.
    if pairs <= 6:
        rows = []
        for i, origin in enumerate(origins):
            for j, destination in enumerate(destinations):
                src, dst = points[i], points[len(origins) + j]
                leg = _get(f"{host}/route/v1/{profile}/"
                           f"{src['lon']},{src['lat']};{dst['lon']},{dst['lat']}",
                           {"overview": "false"})
                if leg.get("code") != "Ok" or not leg.get("routes"):
                    raise MapError(f"no route from {origin} to {destination}")
                route = leg["routes"][0]
                rows.append(f"{origin} to {destination}: "
                            f"{route['distance'] / 1000:.1f} km, "
                            f"{round(route['duration'] / 60)} min")
        return "; ".join(rows)

    coords = ";".join(f"{p['lon']},{p['lat']}" for p in points)
    # OSRM separates source/destination indices with ';', not ','. A comma parses
    # as a malformed query — invisible with a single destination, fatal with two.
    sources = ";".join(str(i) for i in range(len(origins)))
    targets = ";".join(str(i + len(origins)) for i in range(len(destinations)))
    table = _get(f"{host}/table/v1/{profile}/{coords}",
                 {"sources": sources, "destinations": targets,
                  "annotations": "distance,duration"})
    if table.get("code") != "Ok":
        raise MapError("distance matrix unavailable")
    rows = []
    for i, origin in enumerate(origins):
        for j, destination in enumerate(destinations):
            metres = table["distances"][i][j]
            seconds = table["durations"][i][j]
            rows.append(f"{origin} to {destination}: {metres / 1000:.1f} km, "
                        f"{round(seconds / 60)} min")
    return "; ".join(rows)


def elevation(locations: Any = None, **_: Any) -> str:
    points = []
    for item in _as_list(locations):
        if isinstance(item, dict):
            points.append((item["latitude"], item["longitude"]))
        else:
            lat, _, lon = str(item).partition(",")
            points.append((float(lat), float(lon)))
    if not points:
        raise MapError("elevation needs locations")
    payload = _get(_ELEV, {"locations": "|".join(f"{a},{b}" for a, b in points)})
    return "; ".join(f"{r['elevation']} m" for r in payload["results"])


_DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"


def weather(city: str = "", location: str = "", date: str = "", end_date: str = "",
            **_: Any) -> str:
    """Current conditions, or the daily forecast over [date, end_date].

    "주말에 부산 날씨 어때?" carries a date range, and dropping the range to answer
    with right-now conditions is a wrong answer wearing a right answer's clothes.
    open-meteo forecasts ~16 days ahead and archives the recent past, so the range
    the question actually asked about is the range this returns.
    """
    where = city or location or "Seoul"
    hit = _place(where)
    if not date:
        payload = _get(_METEO, {
            "latitude": hit["lat"], "longitude": hit["lon"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"},
            ttl=10 * cache.MINUTE)
        now = payload["current"]
        return (f"{now['temperature_2m']}°C, humidity {now['relative_humidity_2m']}%, "
                f"wind {now['wind_speed_10m']} km/h")

    payload = _get(_METEO, {
        "latitude": hit["lat"], "longitude": hit["lon"], "timezone": "auto",
        "start_date": date, "end_date": end_date or date, "daily": _DAILY},
        ttl=cache.HOUR)
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        return "(no information found)"
    days = [
        f"{day}: high {high}°C, low {low}°C, precipitation {rain} mm, wind {gust} km/h"
        for day, high, low, rain, gust in zip(
            daily["time"], daily["temperature_2m_max"], daily["temperature_2m_min"],
            daily["precipitation_sum"], daily["wind_speed_10m_max"])
    ]
    return f"{where} " + "; ".join(days)


# SGD's Weather_1_GetWeather asks the same question with the same slots.
weather_on = weather


def ip_location(**_: Any) -> str:
    payload = _get(_IPAPI)
    if payload.get("error"):
        raise MapError(payload.get("reason", "ip lookup failed"))
    return (f"{payload['city']}, {payload['region']}, {payload['country_name']}, "
            f"lat {payload['latitude']}, lon {payload['longitude']}")


def _uri(scheme: str, **params: Any) -> str:
    """Amap's schema_* tools return a deep link, not data — build it exactly."""
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
    return f"androidamap://{scheme}?{query}"


def navi(destination: str = "", **_: Any) -> str:
    hit = _place(destination)
    return _uri("navi", lat=hit["lat"], lon=hit["lon"], style=2)


def take_taxi(destination: str = "", **_: Any) -> str:
    hit = _place(destination)
    return _uri("openFeature", featureName="OnRideNavi",
                dlat=hit["lat"], dlon=hit["lon"], dname=destination)


def mark(location: str = "", name: str = "", **_: Any) -> str:
    hit = _place(location or name)
    return _uri("viewMap", poiname=name or location, lat=hit["lat"], lon=hit["lon"])


def personal_map(**_: Any) -> str:
    return _uri("myMap")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in str(value).split("|") if part.strip()]


# ----- bank names -> implementations ------------------------------------------
TOOLS: dict[str, Any] = {
    "maps_geocode": geocode, "map_geocode": geocode, "maps_geo": geocode,
    "maps_reverse_geocode": reverse_geocode, "map_reverse_geocode": reverse_geocode,
    "maps_regeocode": reverse_geocode,
    "maps_search_places": search_places, "map_search_places": search_places,
    "maps_text_search": search_places, "maps_around_search": search_places,
    "maps_place_details": place_details, "map_place_details": place_details,
    "maps_search_detail": place_details,
    "maps_directions": directions, "map_directions": directions,
    "maps_direction_driving": lambda **kw: directions(**dict(kw, mode="driving")),
    "maps_direction_walking": lambda **kw: directions(**dict(kw, mode="walking")),
    "maps_direction_bicycling": lambda **kw: directions(**dict(kw, mode="bicycling")),
    "maps_distance_matrix": distance_matrix, "map_directions_matrix": distance_matrix,
    "maps_distance": distance_matrix,
    "maps_elevation": elevation,
    "map_weather": weather, "maps_weather": weather,
    "map_ip_location": ip_location, "maps_ip_location": ip_location,
    "maps_schema_navi": navi, "maps_schema_take_taxi": take_taxi,
    "map_mark": mark, "maps_schema_personal_map": personal_map,
    "maps_direction_transit_integrated": transit_directions,
}
