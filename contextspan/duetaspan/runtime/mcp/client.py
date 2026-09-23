"""Real MCP client + LLM tool-router for the Context Spanning runtime.

Spawns the four local FastMCP servers (stdio), discovers their tools, keeps the
connections alive on a dedicated background event-loop thread, and exposes a
synchronous ``mcp_route(query, ctx)`` that asks an OpenAI-compatible chat model to
pick the single most appropriate tool (JSON reply) and executes it.

Public API:
    router = get_mcp_router()               # lazy singleton
    result = router.mcp_route(query, ctx)   # -> dict | None

A successful result is::

    {"reference": <spoken one-line str>, "tool": <name>,
     "server": <server name>, "args": <dict>}

A direct answer (general knowledge / abstain) comes back as ``{"answer": <str>, "tool": None, ...}``.
``None`` means no MCP tool applied (or it errored); the caller then injects nothing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from typing import Any, Optional

import requests

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("contextspan.duetaspan.mcp")


class _SkipMerge(Exception):
    """Internal sentinel: skip the registry bank merge (MOSHICP_TOOLPACK_ONLY)."""

_PYTHON = sys.executable
_SERVERS_DIR = os.path.join(os.path.dirname(__file__), "servers")

# server name -> module file
_SERVER_FILES = {
    "time": "time_server.py",
    "weather": "weather_server.py",
    "finance": "finance_server.py",
    "websearch": "websearch_server.py",
}

# The router LLM: any OpenAI-compatible /v1/chat/completions server (e.g. vLLM). The catalog is embedded
# in the prompt and the reply is JSON, because vLLM without --tool-call-parser rejects the native tools API.
_LLM_URL = os.environ.get("MCP_ROUTER_LLM_URL", "http://localhost:8004")
_LLM_MODEL = os.environ.get("MCP_ROUTER_LLM_MODEL", "google/gemma-4-26B-A4B-it")
# Decode is the router's cost (~38 ms/token on a 27B model): bound the reply and the wait, and ask
# the server for a JSON object so no preamble is generated. The reply contract is
# {"name","arguments"} or {"answer"}.
_MAX_TOKENS = int(os.environ.get("MCP_ROUTER_MAX_TOKENS", "120"))
_TIMEOUT_S = float(os.environ.get("MCP_ROUTER_TIMEOUT_S", "8"))
_JSON_MODE = os.environ.get("MCP_ROUTER_JSON_MODE", "1") not in ("0", "false", "off")


def _post_chat(payload, headers, timeout):
    """One chat call; JSON mode is requested first and dropped if the server rejects it."""
    if _JSON_MODE:
        r = requests.post(f"{_LLM_URL}/v1/chat/completions", json={**payload, "response_format": {"type": "json_object"}},
                          headers=headers, timeout=timeout)
        if r.status_code < 400:
            return r
    return requests.post(f"{_LLM_URL}/v1/chat/completions", json=payload, headers=headers, timeout=timeout)


# ── English-only speech model: tool results are anglicized before they become a span ──────────
# Hangul/CJK inside a span is tokenized into byte pieces the speech model never saw, and the model
# then invents or avoids the names.
_NONLATIN_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u3040-\u30ff\u4e00-\u9fff]")


def anglicize_reference(text: str) -> str:
    """Rewrite a tool result into spoken English before it becomes a Context Span.

    Numbers, distances and prices are kept verbatim; proper nouns are romanized (Revised
    Romanization) by the router LLM. No-op when the text is already Latin-script; on any failure the
    original text is returned. Exchange tickers ("(005930.KS)", "(AAPL)") are print-only notation the
    speech model reads as garbage, so the parenthetical is dropped."""
    text = re.sub(r"\s*\((?:[A-Z]{1,5}|\d{4,6})(?:\.[A-Z]{1,3})?\)", "", text or "")
    if not text or not _NONLATIN_RE.search(text):
        return text
    payload = {"model": _LLM_MODEL, "temperature": 0.0,
               "max_tokens": _MAX_TOKENS,
               "messages": [
                   {"role": "system", "content":
                    "Rewrite the given tool result as ONE plain English line for a voice assistant to read aloud. "
                    "Keep every number, unit, distance and price exactly. Write Korean/Japanese/Chinese proper nouns "
                    "in Latin letters (Korean: Revised Romanization, e.g. 스파게티가있는풍경 -> Spaghetti-ga-inneun Punggyeong). "
                    "Do not add, drop or reorder facts. Output only the line, no quotes."},
                   {"role": "user", "content": text}]}
    headers = {"Authorization": "Bearer " + _LLM_KEY} if _LLM_KEY else None
    try:
        resp = requests.post(f"{_LLM_URL}/v1/chat/completions", json=payload, headers=headers, timeout=20)
        out = (resp.json()["choices"][0]["message"]["content"] or "").strip().strip("`")
    except Exception as exc:
        logger.warning("anglicize_reference failed: %s", exc)
        return text
    if not out or _NONLATIN_RE.search(out):
        return text
    if text.startswith("(tool result)") and not out.startswith("(tool result)"):
        out = "(tool result) " + out
    return out
# gpt-5.x / o-series (e.g. gpt-5.6-luna) reject max_tokens + temperature!=1 and need a bearer key.
_rz = os.environ.get("MCP_ROUTER_LLM_REASONING", "").strip().lower()
if _rz in ("1", "true", "on"):
    _LLM_REASONING = True
elif _rz in ("0", "false", "off"):
    _LLM_REASONING = False
else:
    _LLM_REASONING = bool(re.search(r"(gpt-5|luna|\bo[1-9])", _LLM_MODEL, re.I))
_LLM_KEY = os.environ.get("MCP_ROUTER_LLM_KEY") or os.environ.get("OPENAI_API_KEY")

_SYSTEM_PROMPT = (
    "You are a tool-routing controller for a voice assistant. "
    "Given the user's request, decide whether exactly ONE of the available "
    "tools should be called to answer it. Call the single most appropriate "
    "tool with the best arguments you can infer. "
    "Only call get_time for current time/date, get_weather for current weather, "
    "and get_stock_price for live stock or crypto prices. "
    # Live values must never be answered directly. Clock/temperature/price are values the model
    # cannot know, so prose answers are fabrications (e.g. weather restated with no tool).
    "The current time, the current weather, and a live price are values you CANNOT know: "
    "never state them from your own knowledge and never restate them from earlier text. "
    "If such a request is unanswered, you MUST call its tool — answering directly is forbidden. "
    # The model can fabricate weather before the span arrives; the router then reads it as
    # 'already answered' and abstains from get_weather, so the fabrication kills the corrective
    # tool call (a self-reinforcing loop).
    "IGNORE any time/weather/price statement inside ASSISTANT_ALREADY_SAID unless the history "
    "shows the tool call that produced it — with no tool result behind it, that statement is a "
    "HALLUCINATION and the request is still UNANSWERED. For time/weather/price requests, calling "
    "the tool always takes precedence over abstaining. "
    "Only call web_search when the user explicitly asks to search the web or "
    "needs up-to-the-minute live information you cannot already know. "
    "Do NOT call web_search for ordinary general-knowledge or factual questions "
    "(history, science, who-wrote/who-is/what-is questions, definitions, opinions, "
    "or chit-chat) -- for those, call NO tool and instead ANSWER DIRECTLY with one "
    "concise factual spoken sentence (this answer is used verbatim, so make it "
    "complete and correct; resolve pronouns from any provided context). "
    # In long running speech the whole cumulative transcript can arrive at once; without this rule
    # routing picks the FIRST question (e.g. get_time) instead of the last one, and the time is
    # injected for a stock question. Even when the utterance-level window collapses and several
    # questions arrive together, the routing target is always the single most recent UNANSWERED
    # question.
    # Location/timezone arguments are filled by the router from the profile; _inject_ctx only
    # covers the ones it leaves empty.
    "The conversation context may include the user's PROFILE (home city, timezone). "
    "When a tool needs a location or timezone argument and the user did not name one, fill it "
    "from the profile yourself (profile says Lisbon + 'how's the weather?' -> city='Lisbon'; "
    "'what time is it?' -> timezone='Europe/Lisbon'). A place the user explicitly names ALWAYS "
    "wins over the profile ('weather in New York' -> city='New York'). "
    "The transcript may contain SEVERAL questions in a row (running speech). "
    "Route for the LAST question that has not been answered yet -- the most recent "
    "request at the END of the transcript. Earlier questions in the same transcript "
    "are either already answered or superseded; never route for them again. "
    "The request is live-ASR text: minor typos and fillers do NOT make it invalid -- "
    "read through them and route/answer normally ('what tmie is it' is a time "
    "question; 'search the web for X' / 'X 검색해줘' explicitly asks for web_search). "
    "Only when the text is mangled or truncated so badly that the actual request "
    "cannot be recovered, NEVER guess or invent a meaning (do not define garbled "
    "words, do not answer a question the user might have meant) -- answer exactly: "
    "(no information found). "
    "If no argument value is known, OMIT that argument entirely; never pass "
    "placeholder strings like 'null', '<null>', 'none', or 'unknown'. "
    "Never call more than one tool. "
    "The transcript is from live ASR and may contain SEVERAL user requests in a row; "
    "route the LATEST request. If a list of already-answered requests is provided, "
    "those are done — route the newest request that has NOT been answered yet, and "
    "never re-route an already-answered one. "
    # What the agent already SAID counts as answered too — absent from history (the tool ledger), present in speech.
    "An ASSISTANT_ALREADY_SAID block, when present, is the assistant's own spoken answer so "
    "far. Any request it already answers is DONE — never route that one again. Then look at "
    "ASR_TRANSCRIPT for the requests it does NOT yet answer, take the LAST such request, and "
    "CALL ITS TOOL. Do not describe what you would do and do not reply that the answer is "
    "already known: emit the tool call itself. Answer directly (no tool) only when EVERY "
    "request in the transcript is already answered. A greeting, a filler, or a sentence that "
    "has not yet stated the fact does NOT count as an answer. "
    # Scope limit: a reply covers only the single LAST unanswered request. No restating.
    # Without it the router produces, besides the tool call, a direct answer bundling
    # already-answered facts ("It is 11:32 AM ... and the weather is 32°C..."). Fed in as a span,
    # the model gets the same fact twice and either repeats it or wavers over which one to say.
    "SCOPE — whatever you return covers EXACTLY ONE request: the LAST one that is still "
    "unanswered. Work backwards from the END of ASR_TRANSCRIPT: take the last request, and "
    "if ASSISTANT_ALREADY_SAID already answers it, step back to the one before it, and so on. "
    "Never bundle several requests into one reply and never repeat a fact that "
    "ASSISTANT_ALREADY_SAID or an earlier tool result already provided. If every request is "
    "already answered, reply exactly: (no information found). "
    # Serial chains: one utterance may require several steps (several tool calls) — earlier call
    # RESULTS arrive in history, so when a step remains, pick the next tool using those results.
    "One utterance may require SEVERAL tool calls in sequence (e.g. search first, "
    "then book/add/update using the search result). Earlier tool calls are listed "
    "with their RESULTS: if the request still has an unfinished step, call the NEXT "
    "needed tool now — reuse values from those results as arguments (an id, address, "
    "price, or name returned earlier). Calling the SAME tool again with DIFFERENT "
    "arguments is a valid next step (e.g. two conversions, two order ids); only an "
    "identical tool+arguments repeat is forbidden. Answer directly only when every "
    "step of the request is already done. Route a next step ONLY when the transcript "
    "itself asks for it ('then...', 'also...', 'after that...'): NEVER invent a step, "
    "or argument values, that appear in neither the transcript nor the earlier "
    "results. "
    # Self-correction: only the latest intent is valid.
    "If the user corrects themselves mid-request ('no wait', 'actually', 'I mean', "
    "'scratch that', 'not X, Y'), ONLY the latest corrected intent and values are "
    "valid — the pre-correction tool choice and argument values are void; never use "
    "them. "
    # Block near-duplicate re-issues: no repackaged call for the same request.
    "A repeat of an ALREADY-MADE call whose arguments differ only in formatting, "
    "spelling, or a superseded pre-correction value is still a repeat — FORBIDDEN. "
    "Call the same tool again only for a genuinely NEW request or the user's FINAL "
    "corrected values not yet executed."
    # There is deliberately no rule "no tool if the request is incomplete": with it a 26B router
    # escapes into direct answers (a hallucinated id / transcript parroting) instead of a tool call
    # even after the ID is complete. Premature-duplicate calls are left to router model quality,
    # not to the prompt.
)


def _with_history(query: str, history, convo: Optional[str] = None) -> str:
    """Frame the ASR transcript with (a) the Context DB conversation snapshot and
    (b) already-answered requests, so the router model answers the LATEST unanswered
    request (fixed-window ASR re-captures old questions) and serial chains can reuse
    earlier tool results that the ASR window no longer contains."""
    parts = []
    if convo:
        parts.append("Conversation so far (Context DB, oldest→newest; 'tool:' lines are "
                     "calls already executed with their results — reuse their values for "
                     "next steps, never re-run them):\n" + convo)
    if history:
        done = "\n".join(f"- {h}" for h in history[-5:])
        parts.append(f"Tool calls already made in this conversation, with their results "
                     f"(do NOT repeat an identical call; DO use these results as arguments "
                     f"for the next step if the request is not finished):\n{done}")
    if not parts:
        return query
    return "\n\n".join(parts) + \
        f"\n\nTranscript (route the newest unfinished request/step):\n{query}"

# Placeholder / null-ish string values that small models emit for missing args.
_NULLISH = {"", "null", "none", "<null>", "<none>", "unknown", "n/a", "na", "<unknown>"}


# Per-tool intent gates — a small router model over-calls tools (e.g. picks get_weather for
# "capital of France"). A tool only fires when the query actually expresses that intent; otherwise
# no tool is called.
_WEATHER_INTENT = {"weather", "temperature", "forecast", "rain", "raining", "sunny", "cloudy",
                   "hot", "cold", "humid", "wind", "windy", "degrees", "climate", "snow", "snowing",
                   "날씨", "기온", "기상", "습도", "더워", "추워", "덥", "춥", "맑", "흐리"}
_TIME_INTENT = {"time", "clock", "o'clock", "date", "today", "hour", "now", "tonight", "currently",
                "몇 시", "몇시", "시간", "시각", "날짜", "오늘", "지금"}
_STOCK_INTENT = {"stock", "stocks", "share", "shares", "price", "priced", "trading", "ticker",
                 "nasdaq", "kospi", "nyse", "crypto", "bitcoin", "ethereum", "valuation", "worth",
                 "주가", "주식", "시세", "비트코인", "이더리움", "코스피", "나스닥"}


def _has_intent(query: str, words: set) -> bool:
    """Latin words match as tokens (typo-tolerant), Korean gate words by containment.

    An exact-match matcher lets ASR typos ('tmie') miss the gate words, and a hard gate then kills
    correct routing. difflib ratio >= 0.78 fuzzy matching gives the typo tolerance the hard gate needs."""
    import difflib as _dl
    toks = set(re.findall(r"[a-z']+", query.lower()))
    if toks & words:
        return True
    lat = [w for w in words if not any("가" <= c <= "힣" for c in w)]
    for t in toks:
        if len(t) >= 4 and any(len(w) >= 4 and _dl.SequenceMatcher(None, t, w).ratio() >= 0.78
                               for w in lat):
            return True
    return any(w in query for w in words if any("가" <= c <= "힣" for c in w))


_TOOL_INTENT_GATE = {
    "get_weather": _WEATHER_INTENT,
    "get_time": _TIME_INTENT,
    "get_stock_price": _STOCK_INTENT,
}

# ── Structural removal of wrong tool calls: deterministic name recovery + booking-verb guard ──
_TXN_RE = re.compile(r"buy|reserve|book|pay|order|transfer|schedule|cancel|delete", re.IGNORECASE)
# Explicit verb evidence that licenses a booking/payment tool (judged by fuzzy _has_intent)
_TXN_VERBS = {"book", "reserve", "reservation", "buy", "purchase", "order", "pay", "send",
              "transfer", "schedule", "cancel", "delete", "tickets", "예약", "예매", "끊어",
              "잡아", "결제", "취소", "보내"}


def _canonical_name(bank: dict, name: str):
    """Deterministic recovery of a tool name the LLM mangled (no LLM involved).
    exact → case/underscore-insensitive → sole candidate in the same domain → difflib(0.6).
    Guard: a search-intent name (no _TXN match) is never recovered into a booking/payment tool."""
    import difflib as _dl
    if not name:
        return None
    if name in bank:
        return name
    _n = lambda s: s.lower().replace("_", "").replace("-", "")
    norms = {_n(k): k for k in bank}
    hit = norms.get(_n(name))
    if hit:
        return hit
    txn = bool(_TXN_RE.search(name))
    ok = lambda c: txn or not _TXN_RE.search(c)
    parts = name.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        dom = f"{parts[0]}_{parts[1]}"
        same = [k for k in bank if k.startswith(dom + "_") and ok(k)]
        if txn:
            same = [k for k in same if _TXN_RE.search(k)] or same
        if len(same) == 1:
            return same[0]
        close = _dl.get_close_matches(_n(name), [_n(k) for k in same], n=1, cutoff=0.6)
        if close:
            return next(k for k in same if _n(k) == close[0])
    close = _dl.get_close_matches(_n(name), [_n(k) for k in bank if ok(k)], n=1, cutoff=0.75)
    return next((k for k in bank if _n(k) == close[0]), None) if close else None


def _coerce_types(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Deterministically coerce argument values to the schema-declared types (post-processing, the
    model is never touched). number/integer/boolean the LLM stringified as "1234"/"true" go back to
    their declared type; a type mismatch alone fails an otherwise correct call. The values
    themselves are never changed."""
    props = (schema or {}).get("properties") or {}
    out: dict[str, Any] = {}
    for k, v in args.items():
        p = props.get(k) if isinstance(props.get(k), dict) else None
        t = p.get("type") if p else None
        untyped = p is not None and "type" not in p   # in schema but no declared type (= Any)
        if isinstance(v, str):
            s = v.strip()
            try:
                if t == "integer":
                    v = int(float(s))
                elif t == "number":
                    f = float(s)
                    v = int(f) if f.is_integer() else f
                elif t == "boolean" and s.lower() in ("true", "false"):
                    v = s.lower() == "true"
                elif untyped:
                    # Any field -> natural JSON type: "1234"→1234, "true"→true, else keep str.
                    if s.lower() in ("true", "false"):
                        v = s.lower() == "true"
                    elif re.fullmatch(r"-?\d+", s):
                        v = int(s)
                    elif re.fullmatch(r"-?\d+\.\d+", s):
                        v = float(s)
            except Exception:
                pass
        out[k] = v
    return out


def _clean_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop placeholder/null-ish values the LLM emits for missing arguments."""
    cleaned: dict[str, Any] = {}
    for key, val in args.items():
        if val is None:
            continue
        if isinstance(val, str) and val.strip().lower() in _NULLISH:
            continue
        cleaned[key] = val
    return cleaned


class MCPRouter:
    """Holds live stdio connections to all local MCP servers + an LLM router."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-router-loop", daemon=True
        )
        self._thread.start()

        self._stack: Optional[AsyncExitStack] = None
        # tool name -> {"session": ClientSession, "server": str, "schema": dict}
        self._tools: dict[str, dict[str, Any]] = {}

        # build connections synchronously (blocks until all servers attempted)
        self._submit(self._async_setup()).result()

    # ----- background event loop plumbing -------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro) -> Future:
        """Schedule a coroutine on the background loop, return a concurrent Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ----- async setup: spawn + connect + discover ----------------------------
    async def _async_setup(self) -> None:
        self._stack = AsyncExitStack()
        # RUN-SCOPED isolation: when a toolpack is mounted with
        # MOSHICP_TOOLPACK_ONLY=1 the router's tool universe is EXACTLY the pack
        # (no live MCP servers, no registry bank) so a benchmark sees the same
        # tool set its official agent does — a clean, faithful baseline.
        toolpack_only = os.environ.get("MOSHICP_TOOLPACK_ONLY", "").strip().lower() \
            in ("1", "true", "on", "yes")
        for server_name, filename in ({} if toolpack_only else _SERVER_FILES).items():
            path = os.path.join(_SERVERS_DIR, filename)
            try:
                params = StdioServerParameters(
                    command=_PYTHON,
                    args=[path],
                    env=os.environ.copy(),
                )
                read, write = await self._stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                listed = await session.list_tools()
                for tool in listed.tools:
                    schema = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema
                            or {"type": "object", "properties": {}},
                        },
                    }
                    self._tools[tool.name] = {
                        "session": session,
                        "server": server_name,
                        "schema": schema,
                    }
                logger.info(
                    "MCP server '%s' connected with tools: %s",
                    server_name,
                    [t.name for t in listed.tools],
                )
            except Exception as exc:
                logger.warning("MCP server '%s' failed to start: %s", server_name, exc)
        # ── registry tool-bank merge: expose SGD actions (stateful simulation) and other support
        # tools in the router catalog. Schemas are slimmed (prompt budget: a 4096-token router
        # context) — name, short description and types only.
        try:
            if toolpack_only:
                raise _SkipMerge()
            from contextspan.duetaspan.runtime.mcp.registry import get_registry
            _reg = get_registry()
            n_merged = 0
            for fn in _reg.supported():
                if fn in self._tools:
                    continue
                try:
                    sch = _reg.input_schema(fn) or {}
                except Exception:
                    sch = {}
                props = sch.get("properties") or {}
                slim = {"type": "object",
                        "properties": {k: {"type": (v.get("type", "string")
                                                    if isinstance(v, dict) else "string")}
                                       for k, v in props.items()},
                        "required": sch.get("required") or []}
                desc = str(_reg.bank.get(fn, {}).get("description") or "")[:90]
                # Domain-first two-stage routing: stage0 exposes domain groups only (prompt
                # budget), stage1 picks that domain's tools + slim schemas — keep argschema/domain.
                dom = str(_reg.bank.get(fn, {}).get("domain") or fn.split("_")[0])
                self._tools[fn] = {"session": None, "server": "registry", "argschema": slim,
                                   "domain": dom,
                                   "schema": {"type": "function",
                                              "function": {"name": fn, "description": desc,
                                                           "parameters": {"type": "object",
                                                                          "properties": {}}}}}
                n_merged += 1
            logger.info("registry bank merged into router catalog: %d tools", n_merged)
        except _SkipMerge:
            logger.info("registry bank merge skipped (MOSHICP_TOOLPACK_ONLY)")
        except Exception as exc:
            logger.warning("registry merge failed: %s", exc)

        # ── run-scoped toolpack merge (MOSHICP_EXTRA_TOOLPACK): expose a benchmark's official
        # tools in the router catalog for this run only. The pack module exports
        #   TOOLS = {name: {"description":.., "parameters": <json schema>, "fn": callable}}
        # (+ optional DOMAIN). Same contract as the registry merge (argschema/domain/domain-first
        # two-stage) but dispatch calls the pack's fn directly. On a name clash the pack wins
        # (it is the bench's official backend).
        pack_path = os.environ.get("MOSHICP_EXTRA_TOOLPACK", "").strip()
        if pack_path:
            try:
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location("_duetaspan_extra_toolpack", pack_path)
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                pack = getattr(mod, "TOOLS", {}) or {}
                default_dom = str(getattr(mod, "DOMAIN", "") or "toolpack")
                dom_descs = getattr(mod, "DOMAINS", {}) or {}   # {domain: group description (stage0)}
                n_pack = 0
                for fn, spec_d in pack.items():
                    params = spec_d.get("parameters") or {"type": "object", "properties": {}}
                    # 300 chars keeps a description holding parameter hints + one typical query
                    # example (120 truncated the example and cut routing accuracy). Budget: about
                    # 3 tools per domain.
                    desc = str(spec_d.get("description") or "")[:300]
                    dom = str(spec_d.get("domain") or default_dom)
                    # Pack wins: overrides a same-named live/registry tool to reach the bench backend.
                    self._tools[fn] = {"session": None, "server": "toolpack",
                                       "fn": spec_d.get("fn"), "argschema": params,
                                       "domain": dom,
                                       "domain_desc": str(dom_descs.get(dom) or ""),
                                       "schema": {"type": "function",
                                                  "function": {"name": fn, "description": desc,
                                                               "parameters": {"type": "object",
                                                                              "properties": {}}}}}
                    n_pack += 1
                logger.info("toolpack '%s' merged into router catalog: %d tools", pack_path, n_pack)
            except Exception as exc:
                logger.warning("toolpack merge failed for '%s': %s", pack_path, exc)

    async def _async_call_tool(self, name: str, args: dict[str, Any]) -> Optional[str]:
        entry = self._tools.get(name)
        if entry is None:
            return None
        if entry.get("server") == "toolpack":
            # run-scoped toolpack tool: call the pack's fn directly and return its string result.
            fn = entry.get("fn")
            if fn is None:
                return None
            try:
                return str(fn(**(args or {})))
            except Exception as exc:
                logger.warning("toolpack tool '%s' failed: %s", name, exc)
                return None
        if entry.get("server") == "registry":
            # registry tool: reads = real API/web search, actions = stateful sim (registry.dispatch contract)
            try:
                from contextspan.duetaspan.runtime.mcp.registry import get_registry
                return str(get_registry().dispatch(name, args or {}))
            except Exception as exc:
                logger.warning("registry tool '%s' failed: %s", name, exc)
                return None
        try:
            result = await entry["session"].call_tool(name, args)
        except Exception as exc:
            logger.warning("MCP tool '%s' call failed: %s", name, exc)
            return None
        if getattr(result, "isError", False):
            logger.warning("MCP tool '%s' returned an error result", name)
            return None
        for block in result.content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return None

    # ----- LLM tool selection -------------------------------------------------
    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [entry["schema"] for entry in self._tools.values()]

    @staticmethod
    def _llm_pick_openai(query: str, tools: list[dict[str, Any]],
                         history: Optional[list[str]] = None,
                         convo: Optional[str] = None,
                         stage: str = "pick") -> Optional[dict[str, Any]]:
        import json
        import time as _time

        query = _with_history(query, history, convo)

        catalog = json.dumps([t.get("function", t) for t in tools], ensure_ascii=False)
        sys_msg = (
            _SYSTEM_PROMPT
            + "\nAvailable tools (JSON schemas): " + catalog
            + "\nReply with ONLY a JSON object {\"name\": \"<tool>\", \"arguments\": {...}}"
              " when a tool applies, or {\"answer\": \"<one spoken sentence>\"} when you"
              " answer directly (general knowledge / abstain). No prose, no code fences."
        )
        payload = {
            "model": _LLM_MODEL,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": query},
            ],
        }
        if _LLM_REASONING:
            payload["max_completion_tokens"] = 512   # budget hidden reasoning + JSON reply
        else:
            payload["temperature"] = 0.0
            payload["max_tokens"] = _MAX_TOKENS
        headers = {"Authorization": "Bearer " + _LLM_KEY} if _LLM_KEY else None
        _t0 = _time.time()
        try:
            resp = _post_chat(payload, headers, _TIMEOUT_S)
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            logger.warning("MCP router LLM call failed: %s", exc)
            return None
        finally:
            # Measured router latency (wall-time per call).
            print(f"[{_time.strftime('%H:%M:%S')}] [router-llm] stage={stage} "
                  f"dt={_time.time() - _t0:.2f}s", flush=True)
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`\n")
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        if not text or text.lower() in ("null", "none"):
            return None
        try:
            picked = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception:
            # Not JSON = the model answered in prose — keep it as a direct answer instead of dropping it.
            return {"answer": text} if len(text) > 2 else None
        if not isinstance(picked, dict):
            return None
        if picked.get("answer"):                      # single-call direct answer (general knowledge/abstain)
            return {"answer": str(picked["answer"]).strip()}
        # The contract key is "name", but the router model writes the same call as {"tool": ...},
        # {"call": ...} or {"function": ...} often enough to matter: on a WebQuestions + TriviaQA
        # run, 48 no-info spans were well-formed calls under another key.
        for alias in ("tool", "call", "function", "tool_name"):
            if not picked.get("name") and isinstance(picked.get(alias), str):
                picked["name"] = picked.pop(alias)
        for alias in ("args", "parameters", "params"):
            if "arguments" not in picked and isinstance(picked.get(alias), dict):
                picked["arguments"] = picked.pop(alias)
        if not picked.get("name"):
            return None
        return picked

    def _select_tool(self, query: str,
                     history: Optional[list[str]] = None, aux: Optional[str] = None,
                     convo: Optional[str] = None):
        """Ask the LLM to pick one tool. Returns (tool_name, args), {"answer": str}, or None.

        aux (the agent's inner monologue) is attached, labelled, to the LLM message only (for
        pronoun resolution) — the intent gates below judge the clean `query`, so aux words cannot
        open a gate by mistake."""
        tools = self._tool_schemas()
        if not tools:
            return None
        # ── Catch prompt length up front: when the utterance-level window collapses the whole
        # cumulative transcript comes in, routing slows down and is pulled to the first question.
        # Over the cap, log a WARN (window-collapse signal — trigger a VAD/AGC check) and keep only
        # the tail. Consistent with the 'last question' rule.
        self._last_convo = convo or ""     # for hallucinated-city checks (profile-sourced args are legit)
        _qcap = int(os.environ.get("MOSHICP_ROUTER_QUERY_CAP", "480") or 480)
        if len(query) > _qcap:
            print(f"[router] WARN: transcript {len(query)} chars > cap {_qcap} — "
                  f"tail-trim (utterance-window collapse signal)", flush=True)
            query = query[-_qcap:]
        if aux and len(aux) > _qcap:
            aux = aux[-_qcap:]
        # aux = the agent's own inner monologue (= the answer it already said aloud). A generic label
        # such as "Context (earlier conversation)" does not name the source, so the router cannot
        # read it as 'already answered' and picks get_time again from the same transcript after the
        # agent has already said the time. Naming the source and its implication makes it move on to
        # the unanswered request. The label states FACTS only: instructions here ("do not route it
        # again" etc.) make the router reply conversationally, emitting 'call:get_weather{...}' as
        # text instead of JSON. All rules belong to _SYSTEM_PROMPT; the user message carries
        # labelled material only.
        msg = (f"ASSISTANT_ALREADY_SAID (the assistant's own spoken answer so far):\n{aux}"
               f"\n\nASR_TRANSCRIPT:\n{query}") if aux else query
        # Domain-first two-stage (when the registry merge pushes the catalog over budget):
        # stage0 = live tools individually + registry as domain groups → stage1 = that domain's tools+schemas.
        regs = {n: e for n, e in self._tools.items()
                if e.get("server") in ("registry", "toolpack")}
        if regs:
            live = [e["schema"] for n, e in self._tools.items()
                    if e.get("server") not in ("registry", "toolpack")]
            doms: dict[str, list[str]] = {}
            dom_desc: dict[str, str] = {}
            for n, e in regs.items():
                d = e.get("domain") or "misc"
                doms.setdefault(d, []).append(n)
                if e.get("domain_desc") and d not in dom_desc:
                    dom_desc[d] = str(e["domain_desc"])
            stage0 = live + [
                {"type": "function",
                 "function": {"name": f"domain::{d}",
                              "description": ((dom_desc[d] + " — tools: ") if d in dom_desc
                                              else "tool group: ")
                                             + ", ".join(sorted(ns)[:12]),
                              "parameters": {"type": "object", "properties": {}}}}
                for d, ns in sorted(doms.items())]
            fn = self._llm_pick_openai(msg, stage0, history, convo, stage="stage0")
            if isinstance(fn, dict) and str(fn.get("name", "")).startswith("domain::"):
                d = str(fn["name"])[8:]
                sub = [{"type": "function",
                        "function": {"name": n,
                                     "description": (self._tools[n]["schema"]["function"]
                                                     .get("description") or ""),
                                     "parameters": (self._tools[n].get("argschema")
                                                    or {"type": "object", "properties": {}})}}
                       for n in doms.get(d, [])]
                fn = self._llm_pick_openai(msg, sub, history, convo,
                                           stage="stage1") if sub else None
        else:
            fn = self._llm_pick_openai(msg, tools, history, convo, stage="flat")
        if isinstance(fn, dict) and fn.get("answer"):
            return {"answer": fn["answer"]}
        if fn is None:
            return None
        name = fn.get("name")
        if name not in self._tools:
            # Deterministic name recovery — unrecoverable means a safe no-tool
            name = _canonical_name(self._tools, name)
            if name is None:
                return None
        # ── SEARCH vs BOOK enforced in code: a booking/payment tool is allowed only with explicit
        # booking-verb evidence in the utterance. Without it, demote to the domain's sole search
        # tool, and if there is none, no-tool — this structurally turns a 'confidently wrong call'
        # into an abstain (under heavy noise a wrong tool call is far more harmful than abstaining).
        if _TXN_RE.search(name) and not _has_intent(query, _TXN_VERBS):
            parts = name.split("_")
            dom = f"{parts[0]}_{parts[1]}_" if len(parts) >= 3 and parts[1].isdigit() else None
            finds = [k for k in self._tools if dom and k.startswith(dom)
                     and not _TXN_RE.search(k)]
            if len(finds) == 1:
                logger.info("txn-guard: %s -> %s (no booking verb, demoted to search)", name, finds[0])
                name = finds[0]
            else:
                logger.info("txn-guard: %s vetoed (no booking verb) -> no-tool", name)
                return None
        # per-tool intent gate (hard): a live tool fires only when the utterance carries its intent
        # words. The matcher is fuzzy, so an ASR typo ("tmie") does not veto correct routing.
        gate = _TOOL_INTENT_GATE.get(name)
        if gate is not None and not _has_intent(query, gate):
            logger.info("intent-gate veto: %s for %r -> no-tool", name, query[:60])
            return None
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            import json

            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        args = _clean_args(args)
        # The router LLM may fill the city from the profile, so a city absent from the query is not
        # by itself a hallucination: only a city found neither in the query nor in the conversation
        # context (profile included) is dropped.
        if name == "get_weather":
            city = str(args.get("city") or "")
            if city and city.lower() not in query.lower() \
                    and city.lower() not in (self._last_convo or "").lower():
                args.pop("city", None)
        return name, args

    def _llm_fill_args(self, name: str, query: str,
                       history: Optional[list[str]] = None,
                       convo: Optional[str] = None) -> dict[str, Any]:
        """Stage-2 routing: generate only the arguments, from the chosen registry tool's full schema
        (the stage-1 catalog carries no schemas). convo (Context DB snapshot) + history (earlier
        calls and their results) are passed together so serial-chain arguments ("the address/id
        from result A into B") get filled and a self-correcting utterance yields only the latest values."""
        import json as _json
        import time as _time
        entry = self._tools.get(name) or {}
        sch = entry.get("argschema") or {}
        prior = ""
        if convo:
            prior += ("\nConversation so far (Context DB; 'tool:' lines are executed calls "
                      "with results — take referenced values like an address, id, or price "
                      "from them verbatim):\n" + convo)
        if history:
            prior += ("\nEarlier tool calls and their results (use these values when the "
                      "request refers to them, e.g. 'the first one', 'that address', an id "
                      "returned earlier):\n" + "\n".join(f"- {h}" for h in history[-5:]))
        payload = {
            "model": _LLM_MODEL,
            "messages": [
                {"role": "system",
                 "content": ("Extract the arguments for ONE tool call from the user request. "
                             "Reply with ONLY a JSON object of arguments (no prose). "
                             "Use verbatim values from the request; omit unknown OPTIONAL fields. "
                             # Block the path "missing required -> backend TypeError -> the whole
                             # call is lost": required args are always filled from the context,
                             # earlier results, or a common-sense default.
                             "Every field in the schema's 'required' list MUST be present: if "
                             "the user did not state it, infer the most plausible value from the "
                             "request, the earlier results, or a common default (e.g. bedrooms: "
                             "1, mode: 'driving') — never omit a required field. "
                             "The request is live ASR and the user may self-correct ('no wait', "
                             "'actually', 'I mean', 'not X, Y'): ONLY the latest corrected value "
                             "is valid — never use a value the user replaced. "
                             # Spoken-value notation rules (ASR speech → API value): same meaning, notation only.
                             "Value formatting rules for spoken input: write dates as "
                             "'<Month> <number>' with NO ordinal suffix ('May 6', never "
                             "'May 6th'). When the schema description names canonical "
                             "values (e.g. 'passport', 'id_card'), use exactly that snake_case "
                             "token ('id_card', never \"ID card\"). For id-like "
                             "fields (order id, document number, confirmation code), join any "
                             "letters/digits the user spelled out into ONE compact uppercase "
                             "token ('a b 1 2' -> 'AB12'); if the id sounds incomplete or "
                             "garbled, prefer the fullest version heard in the conversation. "
                             "Numeric fields must be JSON numbers (1234, not \"1234\"); boolean "
                             "fields must be true/false literals.\n"
                             f"tool: {name}\nschema: {_json.dumps(sch, ensure_ascii=False)}"
                             + prior)},
                {"role": "user", "content": query},
            ],
        }
        if _LLM_REASONING:
            payload["max_completion_tokens"] = 512
        else:
            payload.update({"temperature": 0.0, "max_tokens": _MAX_TOKENS})
        headers = {"Authorization": "Bearer " + _LLM_KEY} if _LLM_KEY else None
        _t0 = _time.time()
        try:
            resp = _post_chat(payload, headers, _TIMEOUT_S)
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            m = text[text.index("{"): text.rindex("}") + 1]
            args = _json.loads(m)
            return args if isinstance(args, dict) else {}
        except Exception as exc:
            logger.warning("args fill failed for '%s': %s", name, exc)
            return {}
        finally:
            print(f"[router-llm] stage=fill dt={_time.time() - _t0:.2f}s", flush=True)

    # ----- ctx default injection ----------------------------------------------
    @staticmethod
    def _inject_ctx(name: str, args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        args = dict(args)
        if name == "get_time":
            if not args.get("timezone") and ctx.get("timezone"):
                args["timezone"] = ctx["timezone"]
        elif name == "get_weather":
            has_coords = args.get("lat") is not None and args.get("lon") is not None
            if not args.get("city") and not has_coords:
                if ctx.get("city"):
                    args["city"] = ctx["city"]
                elif ctx.get("lat") is not None and ctx.get("lon") is not None:
                    args["lat"] = ctx["lat"]
                    args["lon"] = ctx["lon"]
        return args

    # ----- public sync API ----------------------------------------------------
    def mcp_route(self, query: str, ctx: Optional[dict[str, Any]] = None,
                  history: Optional[list[str]] = None,
                  aux: Optional[str] = None,
                  convo: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Route a query to one MCP tool; return a spoken result dict, a direct
        {"answer": ...} (general knowledge / abstain — single-call design), or None.
        history: already-answered requests (spoken form) — the router must pick the
        newest UNanswered request when the ASR transcript re-captures old ones.
        convo: Context DB working_text() snapshot — injects the cumulative conversation state (user
        turns + tool results) as a first-class input to stage0/stage1/arg-filling, unifying the
        chain's 'result A as an argument to B'."""
        ctx = ctx or {}
        selection = self._select_tool(query, history, aux, convo)
        if selection is None:
            return None
        if isinstance(selection, dict) and selection.get("answer"):
            # Single-call direct answer: the router answers general knowledge/abstain itself — no extra LLM call.
            return {"answer": selection["answer"], "tool": None, "server": None, "args": {}}
        name, args = selection
        entry = self._tools.get(name, {})
        if entry.get("server") in ("registry", "toolpack"):
            if not args:
                args = self._llm_fill_args(name, query, history, convo)  # stage 2: generate args
            # ── final required-args guard right before dispatch ────────────────
            # A missing required arg means a backend TypeError → 'the call itself is lost' (not
            # recorded even when the pick was correct; e.g. add_to_cart without quantity). Validate →
            # retry fill once → otherwise a sensible per-type default. Being recorded always beats
            # dying on empty args.
            sch = entry.get("argschema") or {}
            req = sch.get("required") or []
            missing = [k for k in req if k not in args]
            if missing:
                filled = self._llm_fill_args(name, query, history, convo)
                args = {**filled, **args}
                missing = [k for k in req if k not in args]
            if missing:
                props = sch.get("properties") or {}
                _DEF = {"integer": 1, "number": 1, "boolean": True}
                for k in missing:
                    t = (props.get(k) or {}).get("type", "string")
                    args[k] = _DEF.get(t, "")
                logger.warning("required args defaulted for '%s': %s", name, missing)
            # Schema type coercion (post-processing): "1234"→1234, "true"→true — type only, value unchanged.
            args = _coerce_types(args, sch)
        # Profile arguments for the live tools without a second LLM call: a `get_time` with no timezone or a
        # `get_weather` with no place takes the user's profile value (an argument round trip costs as
        # much as the pick). Values the router did give are never overwritten.
        args = self._inject_ctx(name, args, ctx)

        try:
            reference = self._submit(self._async_call_tool(name, args)).result(
                timeout=30
            )
        except Exception as exc:
            logger.warning("MCP tool '%s' execution failed: %s", name, exc)
            return None
        if not reference:
            return None
        reference = anglicize_reference(reference)          # EN-only speech model
        return {
            "reference": reference,
            "tool": name,
            "server": self._tools[name]["server"],
            "args": args,
        }

    def list_tools(self) -> dict[str, str]:
        """Return {tool_name: server_name} for diagnostics."""
        return {n: e["server"] for n, e in self._tools.items()}


# ----- module-level singleton -------------------------------------------------
_ROUTER: Optional[MCPRouter] = None
_ROUTER_LOCK = threading.Lock()


def get_mcp_router() -> MCPRouter:
    """Lazily build and return the one shared MCP router for this process."""
    global _ROUTER
    if _ROUTER is None:
        with _ROUTER_LOCK:
            if _ROUTER is None:
                _ROUTER = MCPRouter()
    return _ROUTER


def mcp_route(query: str, ctx: Optional[dict[str, Any]] = None,
              history: Optional[list[str]] = None,
              aux: Optional[str] = None,
              convo: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Module-level convenience wrapper around the singleton router."""
    return get_mcp_router().mcp_route(query, ctx, history, aux, convo)
