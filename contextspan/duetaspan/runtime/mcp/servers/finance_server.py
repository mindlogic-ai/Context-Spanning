"""FastMCP server exposing a get_stock_price tool over stdio (Yahoo chart API)."""
from __future__ import annotations

import re
import time

import requests

_http = requests.Session()   # TLS reuse: ~800ms saved per call
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("finance")

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

# Built-in map for common names. The Yahoo search endpoint 429s aggressively,
# so resolve from here first and only hit search on a miss.
_SYMBOL_MAP = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "samsung": "005930.KS",
    "samsung electronics": "005930.KS",
    "hyundai": "005380.KS",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    # Korean company names (the Yahoo search endpoint cannot resolve Hangul)
    "삼성전자": "005930.KS",
    "삼성": "005930.KS",
    "현대차": "005380.KS",
    "현대자동차": "005380.KS",
    "애플": "AAPL",
    "테슬라": "TSLA",
    "엔비디아": "NVDA",
    "비트코인": "BTC-USD",
    "이더리움": "ETH-USD",
}


_NAVER_AC = "https://ac.stock.naver.com/ac"
_HANGUL = re.compile(r"[가-힣]")
_naver_cache: dict[str, tuple[str, str] | None] = {}


def _naver_symbol(company: str) -> tuple[str, str] | None:
    """Korean name → Yahoo symbol via Naver stock autocomplete (keyless, Hangul-native,
    carries the KOSPI/KOSDAQ market split Yahoo search cannot resolve). ANY listed Korean
    ticker works — no per-company map entry needed."""
    if company in _naver_cache:
        return _naver_cache[company]
    out = None
    try:
        resp = _http.get(_NAVER_AC, params={"q": company, "target": "stock"},
                            headers=_UA, timeout=3)
        for item in resp.json().get("items") or []:
            code = item.get("code") or ""
            if item.get("nationCode") == "KOR" and code.isdigit():
                suffix = ".KQ" if "KOSDAQ" in (item.get("typeCode") or "").upper() else ".KS"
                out = (code + suffix, item.get("name") or company)
                break
            if code and not code.isdigit():          # foreign listing surfaced by Naver
                out = (code, item.get("name") or company)
                break
    except Exception:
        out = None
    _naver_cache[company] = out
    return out


def _search_symbol(company: str) -> tuple[str, str] | None:
    """Resolve a company name to (symbol, longname) via Yahoo search with backoff."""
    for attempt in range(2):
        try:
            resp = _http.get(
                _SEARCH_URL,
                params={"q": company, "quotesCount": 1, "newsCount": 0},
                headers=_UA,
                timeout=4,
            )
            if resp.status_code == 429:
                time.sleep(0.5)
                continue
            quotes = resp.json().get("quotes") or []
            if quotes:
                q = quotes[0]
                sym = q.get("symbol")
                if sym:
                    name = q.get("shortname") or q.get("longname") or company
                    return sym, name
            return None
        except Exception:
            time.sleep(0.5)
    return None


@mcp.tool()
def get_stock_price(company: str) -> str:
    """Get the live stock/crypto price for a company or ticker.

    Args:
        company: Company name (e.g. "apple", "samsung electronics") or a ticker symbol.
    """
    if not company:
        return "I need a company name to look up a stock price."

    key = company.strip().lower()
    name = company.strip()
    sym = _SYMBOL_MAP.get(key)
    if sym is None:
        # Hangul names resolve via Naver first (Yahoo search cannot index them at all);
        # everything else tries Yahoo first with Naver as the safety net.
        order = (_naver_symbol, _search_symbol) if _HANGUL.search(name) \
            else (_search_symbol, _naver_symbol)
        found = order[0](name) or order[1](name)
        if found is None:
            return f"I could not find a stock symbol for {company}."
        sym, name = found

    try:
        resp = _http.get(_CHART_URL.format(sym=sym), headers=_UA, timeout=4)
        meta = (resp.json().get("chart", {}).get("result") or [{}])[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        if price is None:
            return f"I could not get a live price for {name} ({sym})."
        currency = meta.get("currency", "")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        change_txt = ""
        if prev:
            try:
                pct = (price - prev) / prev * 100.0
                # Surface-form parity: the canonical change-rate form in the training spans is
                # ", up 1.2% from the previous close" (up/down, 70 cases). A parenthesized
                # "(-8.8% today)" form appears 0 times in training.
                change_txt = f", {'up' if pct >= 0 else 'down'} {abs(pct):.1f}% from the previous close"
            except Exception:
                change_txt = ""
        price_txt = f"{price:,.2f}".rstrip("0").rstrip(".") if isinstance(price, float) else str(price)
        # No "(tool result)" prefix here: the span-prefix is applied uniformly by the consuming
        # layer (realtime._as_tool_result), so baking it into this one server would make its
        # output inconsistent with every other adapter (plain record) and double-prefixed
        # downstream. A trailing "right now" also appears 0 times in the training stock spans —
        # the canonical form ends the sentence with a period: "... KRW."
        return (
            f"{name} ({sym}) is trading at "
            f"{price_txt} {currency}{change_txt}."
        ).replace("  ", " ")
    except Exception:  # pragma: no cover - network failure path
        return ""   # a transport failure is not something to say aloud: no span


if __name__ == "__main__":
    mcp.run(transport="stdio")
