"""FastMCP server exposing a web_search tool over stdio (DuckDuckGo + Wikipedia, key-free)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("websearch")

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_DDG_URL = "https://api.duckduckgo.com/"
_WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"

_NO_INFO = "(no information found)"

# Every call opened a fresh TLS connection to DuckDuckGo, and the handshake alone was
# ~1.3 s of a 1.5 s search. One reused session answers the same query in ~130 ms.
_http = requests.Session()
_http.headers.update(_UA)


def warm_connections() -> None:
    """Open the connections at startup so the first question does not pay the handshake.

    A real GET, not a HEAD — both hosts answer HEAD by closing the connection it opened.
    """
    import threading

    def _open(url: str, params: dict) -> None:
        try:
            _http.get(url, params=params, timeout=4)
        except requests.RequestException:
            pass

    probes = [(_DDG_URL, {"q": "warmup", "format": "json"}),
              (_WIKI_SEARCH, {"action": "query", "format": "json", "meta": "siteinfo"})]
    for url, params in probes:
        threading.Thread(target=_open, args=(url, params), daemon=True).start()


def _ddg_answer(query: str) -> str | None:
    """DuckDuckGo's instant answer, ~120 ms. None when it has none, or is rate-limiting.

    A burst of queries earns HTTP 202 with a non-JSON body, and the old bare `except`
    read that as "no answer" — so the tool silently degraded to a slower source and never
    said why. Narrow the catch: a network failure or a malformed body, nothing else.
    """
    try:
        data = _http.get(
            _DDG_URL,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=4,
        ).json()
    except (requests.RequestException, ValueError):
        return None
    for key in ("AbstractText", "Answer", "Definition"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for topic in data.get("RelatedTopics") or []:
        if isinstance(topic, dict):
            text = topic.get("Text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _wiki_summary(query: str) -> str | None:
    """The lead sentence of the best-matching article, in one request.

    `generator=search` feeds the search hit straight into `prop=extracts`, so the title
    never makes a round trip back to this process. Two calls were 572 ms; this is 492 ms.
    """
    try:
        page = _http.get(
            _WIKI_SEARCH,
            params={
                "action": "query", "format": "json",
                "generator": "search", "gsrsearch": query, "gsrlimit": 1,
                "prop": "extracts", "exintro": 1, "explaintext": 1,
            },
            timeout=4,
        ).json()
    except (requests.RequestException, ValueError):
        return None
    for entry in ((page.get("query") or {}).get("pages") or {}).values():
        extract = entry.get("extract")
        if isinstance(extract, str) and extract.strip():
            return _one_sentence(extract)
    return None


def _one_sentence(text: str) -> str:
    first = text.strip().split(". ")[0].strip()
    return first if first.endswith(".") else first + "."


def instant_answer(query: str) -> str:
    """A confident one-line factual answer, or "(no information found)".

    DuckDuckGo's instant answer only — deliberately without web_search()'s Wikipedia
    fallback. That fallback returns the lead of the best string-matching article even when
    it is only tangentially related ("서울 영화" -> an actor's biography), which is worse
    than silence for a tool result: a caller cannot tell a real answer from a plausible
    wrong one. Real-time listings — today's showtimes, seat availability — have no instant
    answer, so this honestly returns no-info instead of a made-up-looking encyclopedia page.
    One request, no Wikipedia round trip, so it is also faster than web_search().
    """
    if not query:
        return _NO_INFO
    answer = _ddg_answer(query)
    return _one_sentence(answer) if answer else _NO_INFO


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for a single factual answer to a general-knowledge question.

    Args:
        query: The natural-language question or search query.
    """
    if not query:
        return _NO_INFO
    # Neither source needs the other's answer, so ask both at once and keep DuckDuckGo's
    # when it has one. Serially this was 120 ms + 490 ms whenever DuckDuckGo came up empty.
    with ThreadPoolExecutor(max_workers=2) as pool:
        ddg = pool.submit(_ddg_answer, query)
        wiki = pool.submit(_wiki_summary, query)
        answer, fallback = ddg.result(), wiki.result()
    if answer:
        return _one_sentence(answer)
    return fallback or _NO_INFO


if __name__ == "__main__":
    import sys
    import time

    if "--check" in sys.argv:               # one runnable check, no framework
        warm_connections()
        time.sleep(2.0)                     # let the handshakes land
        started = time.monotonic()
        answer = web_search("capital of South Korea")
        elapsed = (time.monotonic() - started) * 1000
        print(f"{elapsed:.0f} ms  {answer}")
        assert "Seoul" in answer, answer
        assert web_search("") == _NO_INFO
        assert _one_sentence("Seoul is a city. It is large.") == "Seoul is a city."
        print("websearch ok")
        sys.exit(0)

    mcp.run(transport="stdio")
