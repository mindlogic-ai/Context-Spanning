"""The bank's 32 browser tools, executed by a real headless Chromium.

They were reported unsupported because no browser was installed. Installing one turned
out to need `pip install playwright`, the bundled Chromium, and exactly one system
library (`libgbm1`) — not a rewrite. The 32 names collapse to ~20 implementations:
`puppeteer_*` is the same operation under a second vendor's name, the way the map bank
carries both Google's and Amap's spelling of "geocode".

Playwright's sync API must be driven from the thread that created it, and the MCP server
dispatches from a thread pool. So every operation is funnelled through one worker thread.
The browser is created on first use and never on import — a data-gen run that never opens
a page should not pay for a Chromium.
"""
from __future__ import annotations

import base64
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from contextspan.duetaspan.common import paths

SHOTS = os.environ.get("MOSHICP_BROWSER_DIR",
                       f"{paths.WORK}/mcp_browser")
_HEADLESS = os.environ.get("MOSHICP_BROWSER_HEADLESS", "1") != "0"


class BrowserError(Exception):
    pass


class _Session:
    """Everything one browser run remembers. Touched only from the worker thread."""

    def __init__(self) -> None:
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.console: list[dict] = []
        self.responses: list[dict] = []
        self.expectations: dict[str, str] = {}
        self.codegen: dict[str, list[str]] = {}


_state = _Session()
_pump = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-browser")
_lock = threading.Lock()


def _in_worker(fn, *args, **kwargs):
    """Run `fn` on the one thread Playwright is allowed to be touched from."""
    with _lock:
        return _pump.submit(fn, *args, **kwargs).result()


def _page():
    if _state.page is not None:
        return _state.page
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:                        # pragma: no cover
        raise BrowserError("playwright is not installed") from exc

    _state.playwright = sync_playwright().start()
    _state.browser = _state.playwright.chromium.launch(
        headless=_HEADLESS, args=["--no-sandbox", "--disable-dev-shm-usage"])
    _state.context = _state.browser.new_context()
    _state.page = _state.context.new_page()
    _state.page.on("console", lambda msg: _state.console.append(
        {"type": msg.type, "text": msg.text}))
    _state.page.on("response", lambda resp: _state.responses.append(
        {"url": resp.url, "status": resp.status}))
    return _state.page


def _record(action: str) -> None:
    for steps in _state.codegen.values():
        steps.append(action)


def shutdown() -> None:
    """Close the browser. Tests call this; a long-lived server need not."""
    def _close():
        for attr in ("context", "browser"):
            handle = getattr(_state, attr)
            if handle is not None:
                handle.close()
        if _state.playwright is not None:
            _state.playwright.stop()
        _state.__init__()
    _in_worker(_close)


# ----- navigation & interaction ----------------------------------------------
def navigate(url: str = "", **_: Any) -> str:
    if not url:
        raise BrowserError("navigate needs a url")

    def _go():
        page = _page()
        response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        _record(f"page.goto({url!r})")
        status = response.status if response else "?"
        return f"{page.url[:80]} ({status}), title {page.title()!r}"
    return _in_worker(_go)


def click(selector: str = "", **_: Any) -> str:
    if not selector:
        raise BrowserError("click needs a selector")

    def _do():
        _page().click(selector, timeout=10000)
        _record(f"page.click({selector!r})")
        return f"clicked {selector}"
    return _in_worker(_do)


def fill(selector: str = "", value: str = "", **_: Any) -> str:
    if not selector:
        raise BrowserError("fill needs a selector")

    def _do():
        _page().fill(selector, value, timeout=10000)
        _record(f"page.fill({selector!r}, {value!r})")
        return f"filled {selector}"
    return _in_worker(_do)


def select(selector: str = "", value: str = "", **_: Any) -> str:
    if not selector:
        raise BrowserError("select needs a selector")

    def _do():
        chosen = _page().select_option(selector, value, timeout=10000)
        _record(f"page.select_option({selector!r}, {value!r})")
        return f"selected {chosen}"
    return _in_worker(_do)


def hover(selector: str = "", **_: Any) -> str:
    if not selector:
        raise BrowserError("hover needs a selector")

    def _do():
        _page().hover(selector, timeout=10000)
        _record(f"page.hover({selector!r})")
        return f"hovered {selector}"
    return _in_worker(_do)


def drag(sourceSelector: str = "", targetSelector: str = "", **_: Any) -> str:
    if not sourceSelector or not targetSelector:
        raise BrowserError("drag needs a source and a target selector")

    def _do():
        _page().drag_and_drop(sourceSelector, targetSelector, timeout=10000)
        _record(f"page.drag_and_drop({sourceSelector!r}, {targetSelector!r})")
        return f"dragged {sourceSelector} -> {targetSelector}"
    return _in_worker(_do)


def press_key(key: str = "", selector: str = "", **_: Any) -> str:
    if not key:
        raise BrowserError("press_key needs a key")

    def _do():
        page = _page()
        if selector:
            page.press(selector, key, timeout=10000)
        else:
            page.keyboard.press(key)
        _record(f"page.keyboard.press({key!r})")
        return f"pressed {key}"
    return _in_worker(_do)


def evaluate(script: str = "", **_: Any) -> str:
    if not script:
        raise BrowserError("evaluate needs a script")

    def _do():
        result = _page().evaluate(script)
        _record(f"page.evaluate({script!r})")
        return str(result)
    return _in_worker(_do)


def screenshot(name: str = "shot", selector: str = "", fullPage: bool = False,
               storeBase64: bool = False, **_: Any) -> str:
    def _do():
        page = _page()
        png = page.locator(selector).screenshot() if selector \
            else page.screenshot(full_page=bool(fullPage))
        _record(f"page.screenshot(path={name!r})")
        if storeBase64:
            head = base64.b64encode(png).decode()[:48]
            return f"data:image/png;base64,{head}… ({len(png)} bytes)"
        os.makedirs(SHOTS, exist_ok=True)
        path = os.path.join(SHOTS, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.png")
        with open(path, "wb") as handle:
            handle.write(png)
        return f"{path} ({len(png)} bytes)"
    return _in_worker(_do)


def custom_user_agent(userAgent: str = "", **_: Any) -> str:
    if not userAgent:
        raise BrowserError("custom_user_agent needs a userAgent")

    def _do():
        # A user agent is set on the context, and a context cannot be mutated — so the
        # page has to be rebuilt. Reopen whatever was on screen: asking to look like a
        # phone should not silently throw away the page you were looking at.
        was = _page().url
        _state.context.close()
        _state.context = _state.browser.new_context(user_agent=userAgent)
        _state.page = _state.context.new_page()
        if was and was != "about:blank":
            _state.page.goto(was, wait_until="domcontentloaded", timeout=30000)
        _record(f"browser.new_context(user_agent={userAgent!r})")
        return f"user agent set to {userAgent}"
    return _in_worker(_do)


def click_and_switch_tab(selector: str = "", **_: Any) -> str:
    if not selector:
        raise BrowserError("click_and_switch_tab needs a selector")

    def _do():
        page = _page()
        with _state.context.expect_page() as opened:
            page.click(selector, timeout=10000)
        _state.page = opened.value
        _state.page.wait_for_load_state("domcontentloaded")
        _record(f"page.click({selector!r})  # opens a tab")
        return f"switched to {_state.page.url[:80]}"
    return _in_worker(_do)


# ----- iframes ----------------------------------------------------------------
def iframe_click(iframeSelector: str = "", selector: str = "", **_: Any) -> str:
    if not iframeSelector or not selector:
        raise BrowserError("iframe_click needs an iframeSelector and a selector")

    def _do():
        _page().frame_locator(iframeSelector).locator(selector).click(timeout=10000)
        _record(f"frame_locator({iframeSelector!r}).click({selector!r})")
        return f"clicked {selector} inside {iframeSelector}"
    return _in_worker(_do)


def iframe_fill(iframeSelector: str = "", selector: str = "", value: str = "",
                **_: Any) -> str:
    if not iframeSelector or not selector:
        raise BrowserError("iframe_fill needs an iframeSelector and a selector")

    def _do():
        _page().frame_locator(iframeSelector).locator(selector).fill(value, timeout=10000)
        _record(f"frame_locator({iframeSelector!r}).fill({selector!r}, {value!r})")
        return f"filled {selector} inside {iframeSelector}"
    return _in_worker(_do)


# ----- console & responses ----------------------------------------------------
def console_logs(type: str = "", search: str = "", limit: int = 20,
                 clear: bool = False, **_: Any) -> str:
    def _do():
        _page()
        entries = _state.console
        if type:
            entries = [e for e in entries if e["type"] == type]
        if search:
            entries = [e for e in entries if search in e["text"]]
        picked = entries[-int(limit or 20):]
        if clear:
            _state.console.clear()
        if not picked:
            return "(no information found)"
        return "; ".join(f"[{e['type']}] {e['text']}" for e in picked)
    return _in_worker(_do)


def expect_response(id: str = "", url: str = "", **_: Any) -> str:
    if not id or not url:
        raise BrowserError("expect_response needs an id and a url")

    def _do():
        _page()
        _state.expectations[id] = url
        return f"watching {url} as {id}"
    return _in_worker(_do)


def assert_response(id: str = "", value: str = "", **_: Any) -> str:
    if not id:
        raise BrowserError("assert_response needs an id")

    def _do():
        pattern = _state.expectations.get(id)
        if pattern is None:
            raise BrowserError(f"no expectation registered as {id}")
        for entry in reversed(_state.responses):
            if pattern in entry["url"]:
                if value and value not in str(entry["status"]):
                    raise BrowserError(
                        f"{entry['url']} answered {entry['status']}, expected {value}")
                return f"{entry['url'][:70]} -> HTTP {entry['status']}"
        return "(no information found)"
    return _in_worker(_do)


# ----- HTTP verbs -------------------------------------------------------------
def _request(method: str, url: str, value: str = "", headers: dict | None = None) -> str:
    if not url:
        raise BrowserError(f"{method} needs a url")

    def _do():
        _page()
        api = _state.context.request
        kwargs: dict[str, Any] = {"headers": headers or {}}
        if value:
            kwargs["data"] = value
        response = getattr(api, method)(url, **kwargs)
        _record(f"context.request.{method}({url!r})")
        return f"HTTP {response.status} {response.status_text}: {response.text()[:110]}"
    return _in_worker(_do)


def http_get(url: str = "", **_: Any) -> str:
    return _request("get", url)


def http_post(url: str = "", value: str = "", token: str = "",
              headers: dict | None = None, **_: Any) -> str:
    merged = dict(headers or {})
    if token:
        merged["Authorization"] = f"Bearer {token}"
    return _request("post", url, value, merged)


def http_put(url: str = "", value: str = "", **_: Any) -> str:
    return _request("put", url, value)


def http_patch(url: str = "", value: str = "", **_: Any) -> str:
    return _request("patch", url, value)


def http_delete(url: str = "", **_: Any) -> str:
    return _request("delete", url)


# ----- codegen sessions -------------------------------------------------------
def start_codegen(options: dict | None = None, **_: Any) -> str:
    def _do():
        session_id = f"codegen-{len(_state.codegen) + 1:03d}"
        _state.codegen[session_id] = []
        return session_id
    return _in_worker(_do)


def get_codegen(sessionId: str = "", **_: Any) -> str:
    def _do():
        steps = _state.codegen.get(sessionId)
        if steps is None:
            raise BrowserError(f"no codegen session {sessionId}")
        body = "; " + "; ".join(steps) if steps else ""
        return f"{sessionId}: {len(steps)} step(s)" + body
    return _in_worker(_do)


def end_codegen(sessionId: str = "", **_: Any) -> str:
    def _do():
        steps = _state.codegen.pop(sessionId, None)
        if steps is None:
            raise BrowserError(f"no codegen session {sessionId}")
        if not steps:
            return f"{sessionId}: no steps recorded"
        return "\n".join([f"# generated from {sessionId}"] + steps)
    return _in_worker(_do)


def clear_codegen(sessionId: str = "", **_: Any) -> str:
    def _do():
        if sessionId not in _state.codegen:
            raise BrowserError(f"no codegen session {sessionId}")
        _state.codegen[sessionId] = []
        return f"{sessionId} cleared"
    return _in_worker(_do)


TOOLS: dict[str, Any] = {
    "playwright_navigate": navigate, "puppeteer_navigate": navigate,
    "playwright_click": click, "puppeteer_click": click,
    "playwright_fill": fill, "puppeteer_fill": fill,
    "playwright_select": select, "puppeteer_select": select,
    "playwright_hover": hover, "puppeteer_hover": hover,
    "playwright_evaluate": evaluate, "puppeteer_evaluate": evaluate,
    "playwright_screenshot": screenshot, "puppeteer_screenshot": screenshot,
    "playwright_drag": drag,
    "playwright_press_key": press_key,
    "playwright_custom_user_agent": custom_user_agent,
    "playwright_click_and_switch_tab": click_and_switch_tab,
    "playwright_iframe_click": iframe_click,
    "playwright_iframe_fill": iframe_fill,
    "playwright_console_logs": console_logs,
    "playwright_expect_response": expect_response,
    "playwright_assert_response": assert_response,
    "playwright_get": http_get,
    "playwright_post": http_post,
    "playwright_put": http_put,
    "playwright_patch": http_patch,
    "playwright_delete": http_delete,
    "start_codegen_session": start_codegen,
    "end_codegen_session": end_codegen,
    "get_codegen_session": get_codegen,
    "clear_codegen_session": clear_codegen,
}


if __name__ == "__main__":  # smallest check that fails if the driver breaks
    # charset must be declared: without it Chromium reads the UTF-8 bytes as Latin-1
    # and "안녕" arrives as "ì•ˆë…•".
    page = ("data:text/html;charset=utf-8,<h1 id=t>안녕</h1>"
            "<select id=s><option>가</option><option>나</option></select>"
            "<button id=b onclick=\"console.log('hi')\">눌러</button>")
    session = start_codegen()
    navigate(page)
    assert evaluate("document.getElementById('t').innerText") == "안녕"
    assert "clicked" in click("#b")
    assert "selected" in select("#s", "나")
    assert "hi" in console_logs()
    assert "bytes" in screenshot("selftest")
    assert "HTTP 200" in http_get("https://example.com")
    steps = end_codegen(session)
    assert "page.click" in steps, steps
    shutdown()
    print("browser adapters ok")
