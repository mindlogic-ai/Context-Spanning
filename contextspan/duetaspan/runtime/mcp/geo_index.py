"""A local gazetteer and POI index, so the common questions never leave this machine.

The remote free sources cannot answer in a second, and no amount of caching fixes the
*first* call. Measured from this box: Nominatim 300-550 ms plus a mandatory 1.05 s gap
between requests, Overpass 2-12 s for a radius query, Photon 1.1 s. Most of that is the
round trip to Europe. `directions` pays the Nominatim gap twice, once per endpoint.

So bring the data here. Two sources, chosen because both download in seconds where
Geofabrik's Korea extract crawled at 17 kB/s:

  GeoNames KR.txt   144k Korean places, nationwide — 선릉역, 강남구, 성수동, 경주, 부산.
                    Its `name` column is romanised; the Korean is in `alternatenames`.
  BBBike Seoul.pbf  Seoul's OSM extract — every POI kind the tool bank asks for, plus the
                    stations and parks GeoNames omits (선정릉역, 반포한강공원).

Geocoding is therefore nationwide; POI search is Seoul-only. `covers()` reports the
measured bounding box of the POI table, and outside it the adapters must still ask
Overpass — there is simply no local data for 부산's cafes, and a slow answer beats none.

    python -m contextspan.duetaspan.runtime.mcp.geo_index --build
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
from typing import Any
from contextspan.duetaspan.common import paths

DATA = str(paths.DATA)
GEO_DIR = os.path.join(DATA, "moshicp", "geo")
DEFAULT_DB = os.environ.get("MOSHICP_MCP_GEO", os.path.join(DATA, "moshicp", "mcp_geo.sqlite"))

# GeoNames feature codes worth geocoding, best first. A question about "부산" means the
# province or the city, never the reservoir that shares the name.
_RANK = {
    "PPLC": 0, "ADM1": 1, "PPLA": 1, "ADM2": 2, "PPLA2": 2, "ADM3": 3, "PPLA3": 3,
    "PPL": 4, "RSTN": 4, "MTRO": 4, "ADM4": 5, "PPLX": 5, "AIRP": 5,
}
# OSM tags to index from the Seoul extract. Mirrors _POI_TAGS in adapters_maps.
_POI_KEYS = {
    "amenity": {"fuel", "charging_station", "cafe", "pharmacy", "hospital", "clinic",
                "bank", "atm", "restaurant", "bar", "pub", "library", "parking",
                "toilets", "cinema", "post_office", "police", "school"},
    "shop": {"convenience", "hairdresser", "supermarket"},
    "leisure": {"park"},
    "tourism": {"hotel", "hostel", "guest_house", "attraction", "museum", "viewpoint"},
    "railway": {"station"},
    "highway": {"bus_stop"},
}
_PLACE_KINDS = {"city", "borough", "suburb", "quarter", "neighbourhood", "town",
                "village", "city_district"}

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_box: tuple[float, float, float, float] | None = None


def _db(write: bool = False) -> sqlite3.Connection | None:
    global _conn
    if _conn is None:
        if not write and not os.path.exists(DEFAULT_DB):
            return None
        _conn = sqlite3.connect(DEFAULT_DB, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def _hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


# ----- build ------------------------------------------------------------------
def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE place (name TEXT, lat REAL, lon REAL, kind TEXT, rank REAL);
        CREATE INDEX place_name ON place (name);
        CREATE TABLE poi (id INTEGER PRIMARY KEY, name TEXT, k TEXT, v TEXT,
                          cuisine TEXT, lat REAL, lon REAL);
        CREATE INDEX poi_tag ON poi (k, v);
        CREATE VIRTUAL TABLE poi_rtree USING rtree (id, minLat, maxLat, minLon, maxLon);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)


def _load_geonames(conn: sqlite3.Connection) -> int:
    rows = []
    with open(os.path.join(GEO_DIR, "KR.txt"), encoding="utf-8") as handle:
        for line in handle:
            field = line.rstrip("\n").split("\t")
            if len(field) < 15 or field[7] not in _RANK:
                continue
            lat, lon = float(field[4]), float(field[5])
            population = int(field[14]) if field[14].isdigit() else 0
            # Lower score wins. Feature code decides first; population breaks ties, so
            # 부산광역시 outranks the hamlet that shares its name.
            score = _RANK[field[7]] - min(population, 10_000_000) / 2e7
            names = {field[1], field[2], *(n for n in field[3].split(",") if n)}
            rows.extend((name, lat, lon, field[7], score) for name in names if name)
    conn.executemany("INSERT INTO place VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def _extract() -> str:
    """The widest OSM extract on disk. Geofabrik's whole-country file when it is there,
    otherwise BBBike's Seoul-only one — which limits POI search to Seoul, and `covers()`
    then says so honestly."""
    for name in ("south-korea.osm.pbf", "Seoul.osm.pbf"):
        path = os.path.join(GEO_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no .osm.pbf in {GEO_DIR}")


def _load_pbf(conn: sqlite3.Connection) -> tuple[int, int]:
    import osmium

    places: list[tuple] = []
    pois: list[tuple] = []

    class Collector(osmium.SimpleHandler):
        def _take(self, tags: dict, lat: float, lon: float) -> None:
            name = tags.get("name")
            if not name:
                return
            if tags.get("place") in _PLACE_KINDS or tags.get("railway") == "station":
                places.append((name, lat, lon, tags.get("place") or "RSTN", 3.5))
            for key, values in _POI_KEYS.items():
                if tags.get(key) in values:
                    pois.append((name, key, tags[key], tags.get("cuisine", ""), lat, lon))

        def node(self, node):
            if node.location.valid():
                self._take(dict(node.tags), node.location.lat, node.location.lon)

        def way(self, way):
            # Parks, hotels and hospitals are drawn as closed ways, not points. Indexing
            # nodes alone would answer "공원 near 성수동" with silence.
            points = [(n.location.lat, n.location.lon) for n in way.nodes
                      if n.location.valid()]
            if points:
                self._take(dict(way.tags),
                           sum(p[0] for p in points) / len(points),
                           sum(p[1] for p in points) / len(points))

    # `locations=True` makes osmium resolve each way's node coordinates; without it a
    # way carries only node ids and no geometry.
    Collector().apply_file(_extract(), locations=True)
    conn.executemany("INSERT INTO place VALUES (?, ?, ?, ?, ?)", places)
    conn.executemany(
        "INSERT INTO poi (name, k, v, cuisine, lat, lon) VALUES (?, ?, ?, ?, ?, ?)", pois)
    conn.execute("INSERT INTO poi_rtree SELECT id, lat, lat, lon, lon FROM poi")
    return len(places), len(pois)


def build() -> None:
    global _conn
    os.makedirs(os.path.dirname(DEFAULT_DB), exist_ok=True)
    if os.path.exists(DEFAULT_DB):
        os.remove(DEFAULT_DB)
    _conn = None
    conn = _db(write=True)
    _schema(conn)
    named = _load_geonames(conn)
    places, pois = _load_pbf(conn)
    box = conn.execute(
        "SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon) FROM poi").fetchone()
    conn.execute("INSERT INTO meta VALUES ('poi_box', ?)",
                 (",".join(str(x) for x in box),))
    conn.commit()
    print(f"geonames names {named}, osm places {places}, osm pois {pois}")
    print(f"poi box {tuple(box)} -> {DEFAULT_DB}")


# ----- query ------------------------------------------------------------------
def place(name: str) -> dict[str, Any] | None:
    """Best local match for a Korean place name, shaped like a Nominatim hit.

    None means "not indexed here", not "no such place" — the caller asks Nominatim.
    """
    conn = _db()
    if conn is None or not _hangul(name):
        return None
    target = name.strip()
    with _lock:
        row = conn.execute(
            "SELECT name, lat, lon, kind FROM place WHERE name = ? ORDER BY rank LIMIT 1",
            (target,)).fetchone()
        if row is None:                       # 성수동 is filed as 성수동1가 / 성수동2가
            row = conn.execute(
                "SELECT name, lat, lon, kind FROM place WHERE name LIKE ? "
                "ORDER BY rank, LENGTH(name) LIMIT 1", (target + "%",)).fetchone()
        if row is None and target.endswith("역") and len(target) > 2:
            # OSM names Seoul's stations without the suffix — 선정릉, 한성대입구 — while
            # every Korean says 선정릉역. Drop the 역 and look only among stations, so
            # 강남역 cannot land on the neighbourhood called 강남.
            row = conn.execute(
                "SELECT name, lat, lon, kind FROM place WHERE name = ? AND kind = 'RSTN' "
                "LIMIT 1", (target[:-1],)).fetchone()
    if row is None:
        return None
    return {"display_name": row["name"], "lat": str(row["lat"]), "lon": str(row["lon"]),
            "addresstype": row["kind"], "importance": 1.0}


def covers(lat: float, lon: float) -> bool:
    """True where the POI table has data. Outside, the caller must ask Overpass."""
    global _box
    conn = _db()
    if conn is None:
        return False
    if _box is None:
        with _lock:
            row = conn.execute("SELECT value FROM meta WHERE key = 'poi_box'").fetchone()
        if row is None:
            return False
        values = [float(x) for x in row["value"].split(",")]
        _box = (values[0], values[1], values[2], values[3])
    lat_min, lat_max, lon_min, lon_max = _box
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def nearby(lat: float, lon: float, radius: int, key: str, value: str,
           cuisine: str = "") -> list[tuple[float, str]]:
    """POIs within `radius` metres, nearest first, as (distance, name)."""
    conn = _db()
    if conn is None:
        return []
    # A degree of latitude is ~111 km everywhere; a degree of longitude shrinks with
    # latitude. Over-select a rectangle from the R-tree, then filter by true distance.
    dlat = radius / 111_000.0
    dlon = radius / (111_000.0 * max(math.cos(math.radians(lat)), 0.01))
    sql = ("SELECT p.name, p.lat, p.lon FROM poi_rtree r JOIN poi p ON p.id = r.id "
           "WHERE r.minLat >= ? AND r.maxLat <= ? AND r.minLon >= ? AND r.maxLon <= ? "
           "AND p.k = ? AND p.v = ?")
    args: list[Any] = [lat - dlat, lat + dlat, lon - dlon, lon + dlon, key, value]
    if cuisine:
        sql += " AND p.cuisine LIKE ?"
        args.append(f"%{cuisine}%")
    with _lock:
        rows = conn.execute(sql, args).fetchall()
    found = [(haversine(lat, lon, row["lat"], row["lon"]), row["name"]) for row in rows]
    return sorted(hit for hit in found if hit[0] <= radius)


if __name__ == "__main__":
    import sys
    import time

    if "--build" in sys.argv:
        build()

    for name in ("선릉역", "강남구", "성수동", "여의도", "경주", "부산", "선정릉역"):
        started = time.monotonic()
        hit = place(name)
        elapsed = (time.monotonic() - started) * 1000
        where = f"{hit['display_name']} ({hit['lat']}, {hit['lon']})" if hit else "— none"
        print(f"{name:<8} {elapsed:6.2f} ms  {where}")

    seolleung = place("선릉역")
    assert seolleung, "선릉역 must be in the local gazetteer"
    lat, lon = float(seolleung["lat"]), float(seolleung["lon"])
    assert covers(lat, lon), "선릉역 must sit inside the POI box"

    started = time.monotonic()
    fuel = nearby(lat, lon, 1500, "amenity", "fuel")
    print(f"fuel near 선릉역  {(time.monotonic() - started) * 1000:.2f} ms  {fuel[:3]}")
    assert fuel, "OSM has fuel stations within 1.5 km of 선릉역"
    assert fuel == sorted(fuel), "results must come back nearest first"

    seongsu = place("성수동")
    assert seongsu and 37.4 <= float(seongsu["lat"]) <= 37.7, \
        f"성수동 belongs in Seoul, got {seongsu}"

    # OSM drops the 역 suffix; a Korean speaker never does.
    seonjeongneung = place("선정릉역")
    assert seonjeongneung, "선정릉역 must resolve even though OSM calls it 선정릉"
    assert haversine(float(seonjeongneung["lat"]), float(seonjeongneung["lon"]),
                     37.51046, 127.04326) < 500, f"선정릉역 is in Gangnam, got {seonjeongneung}"
    assert place("Times Square") is None, "non-Korean names fall through to Nominatim"
    assert not covers(35.68, 139.76), "Tokyo lies outside any Korean extract"

    # Seoul Forest is a closed way, so this is empty unless ways were indexed too.
    parks = nearby(float(seongsu["lat"]), float(seongsu["lon"]), 2000, "leisure", "park")
    assert parks, "성수동 has parks within 2 km; ways must be indexed, not just nodes"
    print(f"parks near 성수동  {len(parks)} found, nearest {parks[0][1]}")

    # 경주 is the whole point of the country-wide extract: on the Seoul-only file this
    # box excluded it and `attractions` fell through to a 3.1 s Overpass call.
    gyeongju = place("경주")
    lat, lon = float(gyeongju["lat"]), float(gyeongju["lon"])
    if covers(lat, lon):
        started = time.monotonic()
        sights = nearby(lat, lon, 6000, "tourism", "attraction")
        print(f"경주 sights  {(time.monotonic() - started) * 1000:.2f} ms  {len(sights)} found")
        assert sights, "경주 has tourist attractions within 6 km"
    else:
        print("경주 is outside the indexed box — Seoul-only extract, Overpass will answer")
    print("geo index ok")
