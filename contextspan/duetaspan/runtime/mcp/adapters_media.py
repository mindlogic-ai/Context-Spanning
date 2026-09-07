"""Music lookup over the keyless iTunes Search API.

SGD's Music_3 catalog holds 594 seeded tracks, none of them Korean, so "아이유 노래
찾아줘" could only ever return nothing. iTunes Search needs no key and knows the
real catalog, which is a better answer than a seeded row pretending to be one.
"""
from __future__ import annotations

from typing import Any

import requests

from contextspan.duetaspan.runtime.mcp import cache

_SEARCH = "https://itunes.apple.com/search"
_UA = {"User-Agent": "DuetaSpan-MCP/1.0 (research)"}

# Reused TCP+TLS: the handshake alone was most of each request.
_http = requests.Session()
_http.headers.update(_UA)


class MediaError(Exception):
    pass


def lookup_music(artist: str = "", track: str = "", album: str = "",
                 genre: str = "", **_: Any) -> str:
    term = " ".join(x for x in (artist, track, album) if x).strip() or genre.strip()
    if not term:
        raise MediaError("music lookup needs an artist, track, album or genre")

    hot = cache.key("itunes", term)
    hits = cache.get(hot)
    if hits is None:
        try:
            resp = _http.get(
                _SEARCH,
                params={"term": term, "entity": "song", "limit": 5},
                timeout=20,
            )
        except requests.RequestException as exc:
            raise MediaError(f"itunes unavailable: {exc}") from exc
        if resp.status_code != 200:
            raise MediaError(f"itunes -> HTTP {resp.status_code}")
        hits = resp.json().get("results") or []
        cache.put(hot, hits, cache.DAY)
    if not hits:
        return "(no information found)"
    return "; ".join(
        f"{h.get('trackName', '?')} — {h.get('artistName', '?')}"
        f" ({h.get('collectionName', '?')}, {str(h.get('releaseDate', ''))[:4]})"
        for h in hits
    )


if __name__ == "__main__":  # smallest thing that fails if the response shape changes
    out = lookup_music(artist="IU")
    assert "—" in out and "(" in out, out
    assert lookup_music(artist="zzzzqqqxnotaband12345") == "(no information found)"
    print("ok:", out[:90])
