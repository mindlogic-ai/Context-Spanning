"""The bank's Tavily search tool, executed without Tavily.

Tavily needs a key; searching does not. The bank's tool name comes from one vendor, the way the map
bank's names come from Google and Amap, and the same approach applies: implement the operation and
alias the vendor's name onto it.

  tavily-search   -> DuckDuckGo (HTML endpoint + instant-answer abstract)

What is lost against real Tavily: LLM-ranked relevance, `search_depth`, `topic`/`days` filtering and
image results. Those slots are accepted and ignored rather than faked, and the caller can tell because
nothing about them appears in the answer.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from bs4 import BeautifulSoup

from contextspan.duetaspan.runtime.mcp import cache

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
_DDG_HTML = "https://html.duckduckgo.com/html/"
_DDG_API = "https://api.duckduckgo.com/"
_TIMEOUT = 25
_RETRY_STATUS = {202, 429, 500, 502, 503, 504}

# Reused TCP+TLS: the handshake alone was most of each request.
_http = requests.Session()
_http.headers.update(_UA)


class SearchError(Exception):
    pass


def warm_connections() -> None:
    """Open the TLS connections to DuckDuckGo before the first search needs them.

    A real GET, not a HEAD: DuckDuckGo closes the connection a HEAD opens, so a search
    right after a "successful" HEAD warm-up still cost 984 ms. Warmed with a GET it is
    777 ms. Daemon threads, silent on failure — a warm-up that cannot reach the network
    must not stop the server from starting.
    """
    def _open(url: str, params: dict) -> None:
        try:
            _http.get(url, params=params, timeout=10)
        except requests.RequestException:
            pass

    probes = [(_DDG_HTML, {"q": "warmup"}),
              (_DDG_API, {"q": "warmup", "format": "json"})]
    for url, params in probes:
        threading.Thread(target=_open, args=(url, params), daemon=True).start()


def _body(url: str, method: str = "get", ttl: float = cache.HOUR, **kwargs) -> str:
    """Fetch a page body, from disk when it was fetched recently: a repeated query is the
    difference between a two-second tool and a two-millisecond one."""
    hot = cache.key(method, url, kwargs.get("data"), kwargs.get("params"))
    stored = cache.get(hot)
    if stored is not None:
        return stored

    last = ""
    for attempt in range(3):
        try:
            resp = _http.request(method, url, timeout=_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last = type(exc).__name__
        else:
            if resp.status_code == 200:
                cache.put(hot, resp.text, ttl)
                return resp.text
            last = f"HTTP {resp.status_code}"
            # DuckDuckGo answers a burst of queries with 202, not 429: "accepted, slow
            # down". Failing on it turns a rate limit into a broken tool.
            if resp.status_code not in _RETRY_STATUS:
                break
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))
    raise SearchError(f"{url} -> {last}")


def _unwrap(href: str) -> str:
    """DuckDuckGo wraps results as /l/?uddg=<encoded>. Give back the real URL."""
    if "uddg=" not in href:
        return href
    target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
    return urllib.parse.unquote(target[0]) if target else href


def _is_ad(href: str) -> bool:
    """Sponsored rows point at DuckDuckGo's own click tracker, not at a page.

    Their href is /y.js?ad_provider=..., which `_unwrap` cannot resolve — it would be
    handed to the caller as the result URL. A search result nobody can fetch is worse
    than one fewer search result.
    """
    return "/y.js" in href or "ad_provider=" in href or "ad_domain=" in href


def _abstract(query: str) -> str:
    """DuckDuckGo's one-line answer, or '' when it has none."""
    try:
        answer = json.loads(_body(_DDG_API, params={"q": query, "format": "json",
                                                    "no_html": 1, "skip_disambig": 1}))
    except SearchError:
        return ""
    text = answer.get("AbstractText") or ""
    return f"answer: {text[:240]}" if text else ""


def search(query: str = "", max_results: int = 5, **_: Any) -> str:
    if not query:
        raise SearchError("search needs a query")

    # The abstract does not depend on the result list, so do not wait for one to ask for
    # the other. Serially, a 120 ms instant answer was bolted onto a ~900 ms page fetch.
    with ThreadPoolExecutor(max_workers=2) as pool:
        page = pool.submit(_body, _DDG_HTML, "post", data={"q": query})
        abstract = pool.submit(_abstract, query)
        soup = BeautifulSoup(page.result(), "html.parser")
        answer = abstract.result()

    hits = []
    for result in soup.select(".result"):
        if len(hits) >= int(max_results or 5):
            break
        anchor = result.select_one("a.result__a")
        if not anchor or _is_ad(anchor.get("href", "")):
            continue
        snippet = result.select_one(".result__snippet")
        hits.append({"title": anchor.get_text(strip=True),
                     "url": _unwrap(anchor.get("href", "")),
                     "snippet": snippet.get_text(" ", strip=True) if snippet else ""})
    if not hits:
        return "(no information found)"

    lines = [answer] if answer else []
    lines += [f"{h['title']} — {h['url']}: {h['snippet'][:120]}" for h in hits]
    # One line, " ; "-separated: a span is a single REF line in the voice pipeline, so the
    # answer + hit rows must not span multiple lines (the span is the backend output as-is).
    return " ; ".join(lines)


TOOLS: dict[str, Any] = {
    "tavily-search": search,
}


if __name__ == "__main__":  # one runnable check per operation
    found = search("hotels in seoul", max_results=4)   # ad-heavy query on purpose
    assert "duckduckgo.com" not in found, f"ad redirector leaked: {found}"
    found = search("capital of South Korea", max_results=3)
    assert "seoul" in found.lower(), found
    print("search adapters ok")
