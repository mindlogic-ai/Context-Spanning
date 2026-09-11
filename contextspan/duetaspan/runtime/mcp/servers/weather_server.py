"""FastMCP server exposing a get_weather tool over stdio (open-meteo, key-free)."""
from __future__ import annotations

from typing import Optional

import time

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

_http = requests.Session()   # TLS reuse: ~800ms saved per call (measured 1018 -> 239ms)

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_geo_cache: dict = {}          # city(lower) → geocoding results[0]; coordinates never change, no TTL
_wx_cache: dict = {}           # (lat,lon) → (ts, forecast json); 60s TTL
_FC_URL = "https://api.open-meteo.com/v1/forecast"

# open-meteo WMO weather interpretation codes -> spoken condition.
_WMO = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


def _condition(code: int) -> str:
    return _WMO.get(int(code), "unclear conditions")


@mcp.tool()
def get_weather(
    city: str = "", lat: Optional[float] = None, lon: Optional[float] = None
) -> str:
    """Get the current weather for a city OR explicit latitude/longitude.

    Args:
        city: City name to geocode (e.g. "Tokyo"). Ignored if lat/lon are given.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
    """
    place = city or "that location"
    try:
        if lat is None or lon is None:
            if not city:
                return "I need a city name or coordinates to check the weather."
            ck = city.strip().lower()
            top = _geo_cache.get(ck)          # city name -> coords never changes — drops ~1.2s geocode per call
            if top is None:
                geo = _http.get(
                    _GEO_URL,
                    params={"name": city, "count": 1, "language": "en", "format": "json"},
                    timeout=4,
                ).json()
                results = geo.get("results") or []
                if not results:
                    return f"I could not find a place called {city}."
                top = _geo_cache[ck] = results[0]
            lat = top["latitude"]
            lon = top["longitude"]
            place = top.get("name", city)
            country = top.get("country")
            if country:
                place = f"{place}, {country}"

        wk = (round(float(lat), 2), round(float(lon), 2))
        hit = _wx_cache.get(wk)                # current weather, 60s TTL — drops ~1.1s of forecast API
        if hit and time.time() - hit[0] < 60:
            fc = hit[1]
        else:
            fc = _http.get(
                _FC_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m",
                },
                timeout=4,
            ).json()
            _wx_cache[wk] = (time.time(), fc)
        cur = fc.get("current") or {}
        temp = cur.get("temperature_2m")
        code = cur.get("weather_code", -1)
        if temp is None:
            return f"I could not get the current weather for {place}."
        cond = _condition(code)
        # Surface-form parity (2026-08-03): the canonical weather form in the training spans is
        #   "(tool result) Busan: 26.1°C, humidity 68%, wind 3.8 km/h, clear skies."  (13,856 cases)
        # The earlier "It's N degrees Celsius and COND in CITY right now." appeared 0 times in
        # training — we were feeding the model a sentence it had never seen. The principle is
        # training surface form == inference surface form.
        hum = cur.get("relative_humidity_2m")
        wind = cur.get("wind_speed_10m")
        parts = [f"{place}: {float(temp):.1f}\u00b0C"]
        if hum is not None:
            parts.append(f"humidity {round(hum)}%")
        if wind is not None:
            parts.append(f"wind {float(wind):.1f} km/h")
        parts.append(cond)
        return "(tool result) " + ", ".join(parts) + "."
    except Exception:  # pragma: no cover - network failure path
        return ""   # a transport failure is not something to say aloud: no span (ContextSpanning #10)


if __name__ == "__main__":
    mcp.run(transport="stdio")
