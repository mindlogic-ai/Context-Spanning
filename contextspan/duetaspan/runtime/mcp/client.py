"""Real MCP client + LLM tool-router for MoshiCP.

Spawns the four local FastMCP servers (stdio), discovers their tools, keeps the
connections alive on a dedicated background event-loop thread, and exposes a
synchronous ``mcp_route(query, ctx)`` that uses ollama function-calling to pick
and execute the single most appropriate MCP tool.

Public API:
    router = get_mcp_router()               # lazy singleton
    result = router.mcp_route(query, ctx)   # -> dict | None

A successful result is::

    {"reference": <spoken one-line str>, "tool": <name>,
     "server": <server name>, "args": <dict>}

``None`` means no MCP tool applied (or it errored), so the caller falls back to
its own LLM-RAG path.
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

_LLM_URL = os.environ.get("MCP_ROUTER_LLM_URL", "http://localhost:11434")
_LLM_MODEL = os.environ.get("MCP_ROUTER_LLM_MODEL", "llama3.2:3b")
# "openai" -> route via an OpenAI-compatible /v1/chat/completions server (e.g. vLLM) using a
# prompt-embedded catalog + JSON reply, because vLLM without --tool-call-parser rejects the
# native tools API. Default stays ollama /api/chat.
_LLM_API = os.environ.get("MCP_ROUTER_LLM_API", "ollama").lower()
# Decode is the router's cost (~38 ms/token on a 27B model): bound the reply and the wait, and ask
# the server for a JSON object so no preamble is generated (ContextSpanning #12). The reply
# contract is unchanged: {"name","arguments"} or {"answer"}.
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
# DuetaSpan working tree (2026-08-27; adopted here 2026-09-07 with the owner's explicit approval as
# the one change to the vendored runtime): Hangul/CJK inside a span is tokenized into byte pieces the
# speech model never saw, and the model then invents or avoids the names (ContextSpanning #11).
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
               "max_tokens": int(os.environ.get("MCP_ROUTER_MAX_TOKENS", "120")),
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
import re as _re
_rz = os.environ.get("MCP_ROUTER_LLM_REASONING", "").strip().lower()
if _rz in ("1", "true", "on"):
    _LLM_REASONING = True
elif _rz in ("0", "false", "off"):
    _LLM_REASONING = False
else:
    _LLM_REASONING = bool(_re.search(r"(gpt-5|luna|\bo[1-9])", _LLM_MODEL, _re.I))
_LLM_KEY = (os.environ.get("MCP_ROUTER_LLM_KEY") or os.environ.get("API_BACKEND_KEY")
            or os.environ.get("OPENAI_API_KEY"))

_SYSTEM_PROMPT = (
    "You are a tool-routing controller for a voice assistant. "
    "Given the user's request, decide whether exactly ONE of the available "
    "tools should be called to answer it. Call the single most appropriate "
    "tool with the best arguments you can infer. "
    "Only call get_time for current time/date, get_weather for current weather, "
    "and get_stock_price for live stock or crypto prices. "
    # 라이브 값은 '직접 답변' 금지. 시계/기온/시세는 모델이 알 수 없는 값이라, 산문으로
    # 답하면 그 자체가 지어낸 값이다(실측: 시간 답한 뒤 날씨를 도구 없이 산문으로 재진술).
    "The current time, the current weather, and a live price are values you CANNOT know: "
    "never state them from your own knowledge and never restate them from earlier text. "
    "If such a request is unanswered, you MUST call its tool — answering directly is forbidden. "
    # (2026-08-25 실사고: 모델이 스팬 도착 전 날씨를 조작 발화 → 라우터가 '기답'으로 오인해
    # get_weather 기권 → 조작이 교정 툴콜을 죽이는 자기강화 루프)
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
    # 2026-08-25 (실측 실패: 40s 연속발화에서 누적 전사가 그대로 들어와 '마지막 질문'이
    # 아니라 첫 질문(get_time)으로 라우팅 → 주가 질문에 시각이 주입됨). 발화-단위 창이
    # 무너져 여러 질문이 함께 들어와도, 라우팅 대상은 항상 '아직 답하지 않은 가장 최근
    # 질문' 하나임을 규칙으로 명시한다 (유저 지시: "누적 전사 전체로 들어가도 가장 최근
    # 질문에 답할 수 있도록").
    # (2026-08-25 유저 설계) 위치/타임존 인자는 코드 폴백이 아니라 라우터가 프로필에서 채운다.
    "The conversation context may include the user's PROFILE (home city, timezone). "
    "When a tool needs a location or timezone argument and the user did not name one, fill it "
    "from the profile yourself (profile says Seoul + 'how's the weather?' -> city='Seoul'; "
    "'what time is it?' -> timezone='Asia/Seoul'). A place the user explicitly names ALWAYS "
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
    "(no information found). Also answer exactly that when you cannot answer "
    "reliably. A wrong confident answer is far worse than abstaining. "
    "If no argument value is known, OMIT that argument entirely; never pass "
    "placeholder strings like 'null', '<null>', 'none', or 'unknown'. "
    "Never call more than one tool. "
    "The transcript is from live ASR and may contain SEVERAL user requests in a row; "
    "route the LATEST request. If a list of already-answered requests is provided, "
    "those are done — route the newest request that has NOT been answered yet, and "
    "never re-route an already-answered one. "
    # 에이전트가 이미 '말한' 것도 answered 다. history(도구 장부)에는 안 실리지만 발화에는 있다.
    # 에이전트가 이미 '말한' 것도 answered 다. history(도구 장부)에는 안 실리지만 발화에는 있다.
    "An ASSISTANT_ALREADY_SAID block, when present, is the assistant's own spoken answer so "
    "far. Any request it already answers is DONE — never route that one again. Then look at "
    "ASR_TRANSCRIPT for the requests it does NOT yet answer, take the LAST such request, and "
    "CALL ITS TOOL. Do not describe what you would do and do not reply that the answer is "
    "already known: emit the tool call itself. Answer directly (no tool) only when EVERY "
    "request in the transcript is already answered. "
    # 범위 제한: 답은 '마지막 미응답 요청' 하나만 다룬다. 재진술 금지.
    # 실측(2026-08-03): 도구콜과 별개로 라우터가 "It is 11:32 AM ... and the weather is 32°C..."
    # 처럼 이미 답한 것까지 묶은 직접답변을 만들어냈다. 그게 span 으로 들어가면 모델은 같은
    # 사실을 두 번 받아 중복 발화하거나 어느 쪽을 말할지 흔들린다.
    "SCOPE — whatever you return covers EXACTLY ONE request: the LAST one that is still "
    "unanswered. Work backwards from the END of ASR_TRANSCRIPT: take the last request, and "
    "if ASSISTANT_ALREADY_SAID already answers it, step back to the one before it, and so on. "
    "Never bundle several requests into one reply and never repeat a fact that "
    "ASSISTANT_ALREADY_SAID or an earlier tool result already provided. If every request is "
    "already answered, reply exactly: (no information found). "
    # 직렬 체인: 한 발화가 여러 스텝(여러 도구콜)을 요구할 수 있다 — 이전 콜의 '결과'가
    # history에 실려 오므로, 남은 스텝이 있으면 그 결과를 인자로 다음 도구를 고른다.
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
    # 자기수정: 최신 의도만 유효.
    "If the user corrects themselves mid-request ('no wait', 'actually', 'I mean', "
    "'scratch that', 'not X, Y'), ONLY the latest corrected intent and values are "
    "valid — the pre-correction tool choice and argument values are void; never use "
    "them. "
    # 근사-중복 재발행 방지(r4 정밀도 실패 16건의 처방): 같은 요청의 재포장 콜 금지.
    "A repeat of an ALREADY-MADE call whose arguments differ only in formatting, "
    "spelling, or a superseded pre-correction value is still a repeat — FORBIDDEN. "
    "Call the same tool again only for a genuinely NEW request or the user's FINAL "
    "corrected values not yet executed."
    # (2026-08-03) '미완성 요청이면 no-tool' 규칙을 넣었다가 되돌림: 26B 라우터가 ID 가
    # 완성된 뒤에도 툴콜 대신 직접답변(환각 'ABC12'/전사 앵무새)으로 도망갔다(프로브 실측
    # calls=[] 2/2). 선발행-중복은 프롬프트가 아니라 라우터 모델 품질(벤치 표준 31B)로 잡는다.
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


# Phrases that signal the user explicitly wants a live web lookup. web_search
# is gated to these so ordinary general-knowledge questions fall back to the
# caller's LLM-RAG path instead of being answered from the open web.
_SEARCH_INTENT = (
    "search",
    "look up",
    "lookup",
    "google",
    "browse the web",
    "on the web",
    "on the internet",
    "latest news",
    "find online",
    "look it up",
    "검색",
    "찾아봐",
    "찾아 줘",
    "찾아줘",
    "알아봐",
)


def _has_search_intent(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _SEARCH_INTENT)


# Per-tool intent gates — a small router model over-calls tools (e.g. picks get_weather for
# "capital of France"). A tool only fires when the query actually expresses that intent; otherwise
# the turn falls back to the caller's LLM-RAG path.
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

    2026-08-25 (router_speed 실험 이식): 종전 정확일치 매처는 ASR 오타('tmie')가 게이트를
    빠뜨려 하드 게이팅을 못 켰다. difflib ratio >= 0.78 퍼지 매칭을 더해 오타 내성을 확보
    — 이것이 하드 게이트 복원의 전제다."""
    import re as _re
    import difflib as _dl
    toks = set(_re.findall(r"[a-z']+", query.lower()))
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

# ── 오툴콜 구조적 제거 (2026-08-25 유저: "잘못된 툴콜 하는 걸 아예 없애줘" — router_speed
# 실험(_canonical/_post_guard/safe-abstain)의 프로덕션 이식) ─────────────────────────────
_TXN_RE = __import__("re").compile(
    r"buy|reserve|book|pay|order|transfer|schedule|cancel|delete", __import__("re").IGNORECASE)
# 예약/결제류 툴을 부를 자격이 되는 명시적 동사 증거 (퍼지 매칭 _has_intent로 판정)
_TXN_VERBS = {"book", "reserve", "reservation", "buy", "purchase", "order", "pay", "send",
              "transfer", "schedule", "cancel", "delete", "tickets", "예약", "예매", "끊어",
              "잡아", "결제", "취소", "보내"}


def _canonical_name(bank: dict, name: str):
    """LLM이 변형한 툴이름의 결정적 복구 (실험 _canonical 이식, LLM 무개입).
    exact → 대소문자/언더스코어 무시 → 같은 도메인 유일 후보 → difflib(0.6).
    가드: 검색 의도 이름(_TXN 미포함)은 절대 예약/결제 툴로 복구되지 않는다."""
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
    """스키마 선언 타입으로 인자 값을 결정적으로 강제(모델 무접촉 후처리).
    LLM이 "1500"/"true"처럼 문자열화한 number/integer/boolean을 원타입으로 — r4 실측:
    타입 불일치만으로 판정 실패 3건(housing_03/08/24). 값 자체는 절대 바꾸지 않는다."""
    props = (schema or {}).get("properties") or {}
    out: dict[str, Any] = {}
    for k, v in args.items():
        p = props.get(k) if isinstance(props.get(k), dict) else None
        t = p.get("type") if p else None
        untyped = p is not None and "type" not in p   # 스키마에 있으나 타입 미선언(= Any)
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
                    # Any 필드는 자연 JSON 타입으로: "1500"→1500, "true"→true, 그 외 문자열 유지.
                    if s.lower() in ("true", "false"):
                        v = s.lower() == "true"
                    elif _re.fullmatch(r"-?\d+", s):
                        v = int(s)
                    elif _re.fullmatch(r"-?\d+\.\d+", s):
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
        self._ready = threading.Event()

        # build connections synchronously (blocks until all servers attempted)
        self._submit(self._async_setup()).result()
        self._ready.set()

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
        # RUN-SCOPED isolation (FDB-v3 etc.): when a toolpack is mounted with
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
        # ── registry 도구뱅크 병합 (2026-07-27): SGD 액션(상태유지 시뮬)·기타 지원 도구를
        # 라우터 카탈로그에 노출 — 벤치(cases100/FDB-v3)의 도구 우주와 백엔드 연결.
        # 스키마는 슬림화(프롬프트 예산: gemma len 4096) — 이름/짧은설명/타입만.
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
                # 도메인-우선 2단 라우팅: 0단은 도메인 그룹만 노출(프롬프트 예산), 1단에서
                # 해당 도메인 툴+슬림 스키마로 선택 — argschema/domain을 엔트리에 보관.
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

        # ── run-scoped toolpack 병합 (MOSHICP_EXTRA_TOOLPACK): 벤치(FDB-v3)의 공식
        # 도구를 이 실행에 한해 라우터 카탈로그에 노출. 팩 모듈은
        #   TOOLS = {name: {"description":.., "parameters": <json schema>, "fn": callable}}
        # (+ 선택 DOMAIN)을 export. registry 병합과 동일 규약(argschema/domain/도메인-우선
        # 2단) 이되 dispatch는 팩의 fn을 직접 호출. 이름 충돌 시 팩이 우선(벤치의 공식 백엔드).
        pack_path = os.environ.get("MOSHICP_EXTRA_TOOLPACK", "").strip()
        if pack_path:
            try:
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location("_duetaspan_extra_toolpack", pack_path)
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                pack = getattr(mod, "TOOLS", {}) or {}
                default_dom = str(getattr(mod, "DOMAIN", "") or "toolpack")
                dom_descs = getattr(mod, "DOMAINS", {}) or {}   # {domain: 그룹 설명(stage0)}
                n_pack = 0
                for fn, spec_d in pack.items():
                    params = spec_d.get("parameters") or {"type": "object", "properties": {}}
                    # 300자: 파라미터 힌트+전형 질의 예시 1개가 들어가는 설명을 살린다
                    # (120은 예시를 잘라 라우팅 정확도를 깎았다). 예산: 도메인당 툴 3개 수준.
                    desc = str(spec_d.get("description") or "")[:300]
                    dom = str(spec_d.get("domain") or default_dom)
                    # 팩이 우선: 라이브/registry 동명 도구를 덮어써 벤치 백엔드로 연결.
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
            # run-scoped toolpack 도구: 팩의 fn을 직접 호출하고 문자열 결과 반환.
            fn = entry.get("fn")
            if fn is None:
                return None
            try:
                return str(fn(**(args or {})))
            except Exception as exc:
                logger.warning("toolpack tool '%s' failed: %s", name, exc)
                return None
        if entry.get("server") == "registry":
            # registry 도구: 읽기=실API/웹검색, 액션=상태유지 시뮬 (registry.dispatch 규약 그대로)
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
    def _llm_pick_ollama(query: str, tools: list[dict[str, Any]],
                         history: Optional[list[str]] = None) -> Optional[dict[str, Any]]:
        query = _with_history(query, history)
        payload = {
            "model": _LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "tools": tools,
            "options": {"temperature": 0.0},
            "keep_alive": -1,        # 모델 상주 고정 — 세션 사이 idle에도 언로드 금지(콜드 방지)
        }
        try:
            resp = requests.post(f"{_LLM_URL}/api/chat", json=payload, timeout=30)
            data = resp.json()
        except Exception as exc:
            logger.warning("MCP router LLM call failed: %s", exc)
            return None
        calls = (data.get("message") or {}).get("tool_calls") or []
        if not calls:
            return None
        return calls[0].get("function") or {}

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
        # MCP_ROUTER_DEBUG=1 이면 라우터에 실제로 들어간 유저 메시지를 그대로 찍는다.
        # (도구 카탈로그는 124개라 너무 길어 제외 — 시스템 규칙은 코드에 고정되어 있다.)
        if os.environ.get("MCP_ROUTER_DEBUG") == "1":
            _ts = _time.strftime("%H:%M:%S") + f".{int(_time.time() % 1 * 1000):03d}"
            print(f"\n[{_ts}] [router-in stage={stage}] ─────────────────────────\n{query}\n"
                  f"[/router-in]", flush=True)
        _t0 = _time.time()
        try:
            resp = _post_chat(payload, headers, _TIMEOUT_S)
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            logger.warning("MCP router LLM call failed: %s", exc)
            return None
        finally:
            # 라우터 지연 실측(호출당 wall-time) — r4 대비 +20% 예산 검증용.
            print(f"[{_time.strftime('%H:%M:%S')}] [router-llm] stage={stage} "
                  f"dt={_time.time() - _t0:.2f}s", flush=True)
        if os.environ.get("MCP_ROUTER_DEBUG") == "1":
            _ts = _time.strftime("%H:%M:%S") + f".{int(_time.time() % 1 * 1000):03d}"
            print(f"[{_ts}] [router-out stage={stage}] dt={_time.time()-_t0:.2f}s "
                  f"{content.strip()[:300]!r}", flush=True)
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
            # JSON이 아니면 모델이 그냥 문장으로 답한 것 — 직접답변으로 살린다(버리면 RAG 2콜).
            return {"answer": text} if len(text) > 2 else None
        if not isinstance(picked, dict):
            return None
        if picked.get("answer"):                      # 단일콜 직접답변 (일반지식/abstain)
            return {"answer": str(picked["answer"]).strip()}
        if not picked.get("name"):
            return None
        return picked

    def _select_tool(self, query: str,
                     history: Optional[list[str]] = None, aux: Optional[str] = None,
                     convo: Optional[str] = None):
        """Ask the LLM to pick one tool. Returns (tool_name, args), {"answer": str}, or None.

        aux(내적독백)는 LLM 메시지에만 라벨링해 붙인다(대명사 해소용) — 이후의 인텐트
        게이트들은 클린 `query`로 판정해 aux 단어가 게이트를 잘못 여는 일을 막는다."""
        tools = self._tool_schemas()
        if not tools:
            return None
        # ── 프롬프트 길이 사전 캐치 (2026-08-25 유저: "router는 빨라야 하는데 프롬프트가
        # 너무 길어져도 사전에 캐치돼 있어야"): 발화-단위 창이 무너지면 누적 전사가 통째로
        # 들어와 라우팅이 느려지고 첫 질문에 끌린다. 상한 초과 시 WARN을 남기고(창 붕괴
        # 신호 — VAD/AGC 점검 트리거) 꼬리만 유지한다. '마지막 질문' 규칙과 정합.
        self._last_convo = convo or ""     # 환각-city 판정용 (프로필 출처 인자는 정당)
        _qcap = int(os.environ.get("MOSHICP_ROUTER_QUERY_CAP", "480") or 480)
        if len(query) > _qcap:
            print(f"[router] WARN: transcript {len(query)} chars > cap {_qcap} — "
                  f"tail-trim (utterance-window collapse signal)", flush=True)
            query = query[-_qcap:]
        if aux and len(aux) > _qcap:
            aux = aux[-_qcap:]
        # aux = 에이전트 자신의 내적 독백(= 이미 소리내어 말한 답). 예전 라벨
        # "Context (earlier conversation)"는 출처를 안 밝혀서, 라우터가 이걸 '이미 답한 것'으로
        # 읽지 못했다 — 실측: 에이전트가 시간을 이미 말한 뒤에도 같은 전사에서 get_time 을
        # 다시 골랐다(2026-08-03). 출처와 함의를 명시해 미응답 요청으로 넘어가게 한다.
        # 라벨은 '사실 서술'만 둔다. 여기에 지시문("do not route it again" 등)을 넣었더니
        # 라우터가 대화체로 응답해 JSON 대신 'call:get_weather{...}' 를 텍스트로 뱉었다.
        # 규칙은 전부 _SYSTEM_PROMPT 가 소유하고, 유저 메시지는 라벨링된 자료만 담는다.
        msg = (f"ASSISTANT_ALREADY_SAID (the assistant's own spoken answer so far):\n{aux}"
               f"\n\nASR_TRANSCRIPT:\n{query}") if aux else query
        if _LLM_API == "openai":
            # 도메인-우선 2단 (registry 병합으로 카탈로그가 프롬프트 예산 초과 시):
            # 0단 = 라이브 툴 개별 + registry는 도메인 그룹으로만 → 1단 = 그 도메인 툴+스키마.
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
        else:
            fn = self._llm_pick_ollama(msg, tools, history)
        if isinstance(fn, dict) and fn.get("answer"):
            return {"answer": fn["answer"]}
        if fn is None:
            return None
        name = fn.get("name")
        if name not in self._tools:
            # 결정적 이름 복구 (실험 _canonical 이식) — 복구 불가면 안전한 no-tool
            name = _canonical_name(self._tools, name)
            if name is None:
                return None
        # ── SEARCH vs BOOK 코드 강제 (실험 safe-abstain 이식): 예약/결제류 툴은 발화에
        # 명시적 예약 동사 증거가 있어야만 허용. 없으면 같은 도메인의 유일한 검색 툴로
        # 강등하고, 그마저 없으면 no-tool — '틀린 확신 호출'을 구조적으로 abstain으로 바꾼다
        # (heavy-noise 실측: 오툴콜이 abstain보다 훨씬 해로움 — NOISE_ROBUSTNESS.md).
        if _TXN_RE.search(name) and not _has_intent(query, _TXN_VERBS):
            parts = name.split("_")
            dom = f"{parts[0]}_{parts[1]}_" if len(parts) >= 3 and parts[1].isdigit() else None
            finds = [k for k in self._tools if dom and k.startswith(dom)
                     and not _TXN_RE.search(k)]
            if len(finds) == 1:
                logger.info("txn-guard: %s -> %s (예약 동사 부재, 검색 강등)", name, finds[0])
                name = finds[0]
            else:
                logger.info("txn-guard: %s vetoed (예약 동사 부재) -> no-tool", name)
                return None
        # llama3.2:3b cannot reliably suppress web_search for ordinary
        # general-knowledge questions via prompting alone (verified: it calls
        # web_search for "who wrote Dune" even when told not to). Gate web_search
        # to explicit search intent so plain factual questions fall back to the
        # caller's LLM-RAG path, per the routing contract.
        if name == "web_search" and not _has_search_intent(query):
            return None
        # per-tool intent gate — 기원: llama3.2:3b가 "capital of France"에 get_weather를
        # 과호출하던 시절의 하드 차단. 단일콜 설계(직접답변/abstain 선택지 보유) + a4b급
        # 라우터에선 툴 선택이 의도적이라, 하드 게이트는 ASR 오타("tmie")가 키워드를 못
        # 맞추면 정답 라우팅까지 죽인다. 기본 SOFT(경고만); MCP_ROUTER_HARD_GATES=1 복원.
        gate = _TOOL_INTENT_GATE.get(name)
        if gate is not None and not _has_intent(query, gate):
            # 2026-08-25: 매처가 퍼지(오타 내성)가 되면서 소프트로 물러났던 근거('tmie'류가
            # 정답 라우팅을 죽임)가 해소됨 → 기본 HARD 복원 (유저: "잘못된 툴콜 아예 없애").
            # MCP_ROUTER_SOFT_GATES=1 로만 종전 소프트 동작 복귀.
            if not os.environ.get("MCP_ROUTER_SOFT_GATES"):
                logger.info("intent-gate veto (hard): %s for %r -> no-tool", name, query[:60])
                return None
            logger.info("intent-gate miss (soft): %s for %r — trusting router", name, query[:60])
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
        # (2026-08-25 유저 설계: "폴백/구현 없이 router LLM에게 정보를 줘서 툴콜링하게") —
        # 종전엔 쿼리에 없는 city를 무조건 삭제해 LLM이 프로필로 옳게 채운 인자까지 지웠고,
        # 그 빈자리를 _inject_ctx가 결정적으로 재주입했다. 이제 프로필 인자 채움은 LLM 소관:
        # 쿼리에도, 대화 컨텍스트(프로필 포함)에도 없는 city만 환각으로 보고 떨군다.
        if name == "get_weather":
            city = str(args.get("city") or "")
            if city and city.lower() not in query.lower() \
                    and city.lower() not in (self._last_convo or "").lower():
                args.pop("city", None)
        return name, args

    def _llm_fill_args(self, name: str, query: str,
                       history: Optional[list[str]] = None,
                       convo: Optional[str] = None) -> dict[str, Any]:
        """2단 라우팅: 선택된 registry 도구의 풀 스키마로 인자만 생성 (1단 카탈로그는 무스키마).
        convo(Context DB 스냅샷)+history(이전 콜+결과)를 함께 줘 직렬 체인 인자("결과 A의
        주소/ID를 B에")를 채우고, 자기수정 발화에선 최신 값만 채택하게 한다."""
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
                             # required 누락 → 백엔드 TypeError → 콜 전체가 소실되는 경로 차단:
                             # 필수 인자는 컨텍스트/이전 결과/상식 디폴트로 반드시 채운다.
                             "Every field in the schema's 'required' list MUST be present: if "
                             "the user did not state it, infer the most plausible value from the "
                             "request, the earlier results, or a common default (e.g. bedrooms: "
                             "1, mode: 'driving') — never omit a required field. "
                             "The request is live ASR and the user may self-correct ('no wait', "
                             "'actually', 'I mean', 'not X, Y'): ONLY the latest corrected value "
                             "is valid — never use a value the user replaced. "
                             # 음성값 표기 규범(ASR 발화체 → API 값): 의미 동일, 표기만 정규화.
                             "Value formatting rules for spoken input: write dates as "
                             "'<Month> <number>' with NO ordinal suffix ('November 1', never "
                             "'November 1st'). When the schema description names canonical "
                             "values (e.g. 'passport', 'id_card'), use exactly that snake_case "
                             "token ('driver_license', never \"driver's license\"). For id-like "
                             "fields (order id, document number, confirmation code), join any "
                             "letters/digits the user spelled out into ONE compact uppercase "
                             "token ('d l 5 5 5' -> 'DL555'); if the id sounds incomplete or "
                             "garbled, prefer the fullest version heard in the conversation. "
                             "Numeric fields must be JSON numbers (1500, not \"1500\"); boolean "
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
        # MCP_ROUTER_DEBUG=1 이면 라우터에 실제로 들어간 유저 메시지를 그대로 찍는다.
        # (도구 카탈로그는 124개라 너무 길어 제외 — 시스템 규칙은 코드에 고정되어 있다.)
        if os.environ.get("MCP_ROUTER_DEBUG") == "1":
            _ts = _time.strftime("%H:%M:%S") + f".{int(_time.time() % 1 * 1000):03d}"
            print(f"\n[{_ts}] [router-in stage=fill] ─────────────────────────\n{query}\n"
                  f"[/router-in]", flush=True)
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
        convo: Context DB working_text() 스냅샷 — 대화 누적 상태(유저 턴+도구 결과)를
        stage0/stage1/인자채움의 1급 입력으로 주입(체인의 '결과 A를 인자로 B' 일원화)."""
        ctx = ctx or {}
        selection = self._select_tool(query, history, aux, convo)
        if selection is None:
            return None
        if isinstance(selection, dict) and selection.get("answer"):
            # 단일콜 직접답변: 라우터 모델이 일반지식/abstain을 즉답 — 별도 RAG 콜 불필요.
            return {"answer": selection["answer"], "tool": None, "server": None, "args": {}}
        name, args = selection
        entry = self._tools.get(name, {})
        if entry.get("server") in ("registry", "toolpack"):
            if not args:
                args = self._llm_fill_args(name, query, history, convo)  # 2단: 인자 생성
            # ── dispatch 직전 required 최종 가드 ──────────────────────────────
            # 누락 required는 백엔드 TypeError → '콜 자체 소실'(선택 정답이어도 미기록)로
            # 이어진다(r2 실측: add_to_cart quantity). 검증 → fill 재시도 1회 → 그래도
            # 없으면 타입별 합리적 기본값. 빈 인자로 죽이는 것보다 기록되는 편이 항상 낫다.
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
            # 스키마 타입 강제(후처리): "1500"→1500, "true"→true — 값 불변, 타입만.
            args = _coerce_types(args, sch)
        # Profile arguments for the live tools without a second LLM call: a `get_time` with no timezone or a
        # `get_weather` with no place takes the user's profile value (ContextSpanning #12: the argument
        # round trip cost as much as the pick). Values the router did give are never overwritten.
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

    def close(self) -> None:
        try:
            if self._stack is not None:
                self._submit(self._stack.aclose()).result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


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
