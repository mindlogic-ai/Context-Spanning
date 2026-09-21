"""FastMCP server exposing a web_search tool over stdio (DuckDuckGo + Wikipedia, key-free)."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("websearch")

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_DDG_URL = "https://api.duckduckgo.com/"
_WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"

_NO_INFO = "(no information found)"
_TIMEOUT_S = 2   # per request: the span deadline is 2.5 s, so a slower answer is dropped anyway (as get_weather)

# A fresh TLS connection to DuckDuckGo per call makes the handshake alone ~1.3 s of a
# 1.5 s search. One reused session answers the same query in ~130 ms.
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

    A burst of queries earns HTTP 202 with a non-JSON body. The catch is kept narrow — a
    network failure or a malformed body, nothing else — so no other error is silently read
    as "no answer".
    """
    try:
        data = _http.get(
            _DDG_URL,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=_TIMEOUT_S,
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
    never makes a round trip back to this process. Two calls take 572 ms; this takes 492 ms.
    """
    try:
        page = _http.get(
            _WIKI_SEARCH,
            params={
                "action": "query", "format": "json",
                "generator": "search", "gsrsearch": query, "gsrlimit": 1,
                "prop": "extracts", "exintro": 1, "explaintext": 1,
            },
            timeout=_TIMEOUT_S,
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
    """Search the web when the user explicitly asks for a web search or needs live information the
    assistant cannot already know. Ordinary general-knowledge questions are answered directly.

    Args:
        query: The natural-language question or search query.
    """
    if not query:
        return _NO_INFO
    # Ask both at once and return as soon as either has an answer. Waiting for both means a
    # provider that cannot be reached (DuckDuckGo's TCP connect hangs from some hosts) holds
    # back the answer the other one already has until its own timeout: measured 4.06 s for a
    # Wikipedia sentence that was in hand at 0.62 s, i.e. past the span deadline, so the turn
    # ends as "(no information found)". DuckDuckGo's instant answer is preferred when both
    # arrive together.
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {pool.submit(_ddg_answer, query): 0, pool.submit(_wiki_summary, query): 1}
    try:
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=_TIMEOUT_S + 0.5, return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in sorted(done, key=futs.get):          # DuckDuckGo first when both are in
                try:
                    answer = fut.result()
                except Exception:
                    answer = None
                if answer:
                    # DuckDuckGo's instant answer is cut to its first sentence here; the Wikipedia
                    # lead was already cut to its first sentence by _wiki_summary.
                    return _one_sentence(answer) if futs[fut] == 0 else answer
        return _NO_INFO
    finally:
        pool.shutdown(wait=False)


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
