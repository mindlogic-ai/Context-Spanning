# -*- coding: utf-8 -*-
"""REAL realtime retrieval backend (web search + live tools) for MoshiCP.

Replaces the mock `retrieve.Backend` (TF-IDF over a static pool + synthesized
"(tool result) …" clauses). This module ACTUALLY fetches:

  * RAG / open-domain  -> live DuckDuckGo (Instant-Answer API, then HTML snippet)
  * weather            -> open-meteo (free, no key): geocode city -> current weather
  * time / date        -> local clock

Every result is normalized to the SAME spoken one-line contract the model trains on
so data-generation and serving go through one code path (train==infer):

  * MCP/tool result  -> "(tool result) <spoken clause>"
  * RAG passage      -> "<spoken factual sentence>"

Key-free, pure-Python (requests + bs4). No GPU, no model load. A small on-disk cache
makes repeated queries (data-gen) cheap and deterministic.
"""

import datetime as _dt
import json
import os
import re
import time
import urllib.parse

import requests
from contextspan.duetaspan.common import paths

NO_INFO = "(no information found)"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
_CACHE_DIR = f"{paths.DATA}/moshicp/realtime_cache"

# Realtime/tool intent → MCP path (mirrors retrieve.MCP_INTENT_WORDS, kept local on purpose).
_MCP_WORDS = {
    "weather", "temperature", "forecast", "rain", "raining", "sunny", "cloudy", "hot", "cold",
    "time", "clock", "timezone", "date", "today", "tonight", "now", "current", "currently",
    "price", "stock", "cost", "rate", "score", "scores", "game", "match", "news", "headlines",
    "flight", "flights", "departure", "arrival", "schedule", "open", "hours",
}
_WEATHER_WORDS = {"weather", "temperature", "forecast", "rain", "raining", "sunny", "cloudy", "hot", "cold", "humid", "wind"}
_TIME_WORDS = {"time", "clock", "o'clock", "date", "day", "today"}
_STOCK_WORDS = {"stock", "stocks", "share", "shares", "ticker", "price", "shareprice", "trading",
                "nasdaq", "kospi", "nyse", "valuation", "marketcap"}

_WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "drizzling", 53: "drizzling", 55: "drizzling",
    61: "rainy", 63: "rainy", 65: "heavy rain", 71: "snowy", 73: "snowy", 75: "heavy snow",
    80: "showery", 81: "showery", 82: "heavy showers", 95: "thunderstorms", 96: "thunderstorms",
}


def _tok(s):
    return set(re.findall(r"[a-z']+", (s or "").lower()))


def _fmt_time(tzname, place=""):
    """'(tool result) It is 10:37 pm on Thursday, June 18 [in Seoul].' for an IANA tz, or None."""
    if not tzname:
        return None
    try:
        from zoneinfo import ZoneInfo
        now = _dt.datetime.now(ZoneInfo(tzname))
    except Exception:
        return None
    where = f" in {place}" if place else ""
    return (f"(tool result) It is {now.strftime('%-I:%M %p').lower()} "
            f"on {now.strftime('%A, %B %-d')}{where}.")


def _ctx_place(ctx):
    """User's place label from a ctx dict: city, else timezone leaf ('Asia/Seoul' -> 'Seoul')."""
    if not ctx:
        return ""
    if ctx.get("city"):
        return ctx["city"]
    tz = ctx.get("timezone") or ""
    return tz.rsplit("/", 1)[-1].replace("_", " ") if "/" in tz else tz


def _personal_from_ctx(q, ctx):
    """Answer a question about the user from their profile (no web). Returns a spoken factual
    sentence the model will ground on, or None to defer to tool/web retrieval."""
    if not ctx:
        return None
    ql = (q or "").lower()
    toks = set(re.findall(r"[a-z']+", ql))
    name = ctx.get("name") or ""
    place = _ctx_place(ctx)
    persona = ctx.get("persona") or ""
    # AGENT self-identity ("who are you", "what's your name", "what do you do / your job") grounds on
    # the Context DB persona, in first person. Gated to 'you/your' (not 'my'/'am i', which is the user).
    if persona and ("you" in toks or "your" in toks) and "my" not in toks:
        if (("who" in toks and ("you" in toks or "are" in toks))
                or ("your" in toks and "name" in toks)
                or (("what" in toks or "whats" in ql or "what's" in ql) and "do" in toks and "you" in toks)
                or ("your" in toks and (toks & {"job", "work", "profession", "occupation", "do"}))):
            return f"I am {persona}."
    # USER name/location — STRICT contiguous PHRASES (scattered tokens like "where did he start? i am
    # skeptical" must NOT match "where am i"). re on the raw lowered string with apostrophes normalized.
    qn = re.sub(r"[^a-z ]+", " ", ql)
    if re.search(r"\b(my name is|what s my name|whats my name|what is my name|who am i)\b", qn):
        return f"Your name is {name}." if name else None
    if re.search(r"\b(where am i|where i am|my location|my city|what city am i|which city am i)\b", qn):
        return f"You are in {place}." if place else None
    if "timezone" in toks or ("time" in toks and "zone" in toks):
        return f"Your timezone is {ctx.get('timezone')}." if ctx.get("timezone") else None
    # Korean — the [a-z'] tokenizer above yields NOTHING for Korean speech, so every Korean
    # personal question silently fell through to tools/web. Same phrase-level strictness: a
    # self-referential subject (나/내/저) must be present, so "여기 어디야" (a reverse-geocode
    # question about a place) does NOT match.
    qk = q or ""
    if persona and re.search(r"[너넌]\s*(는|은)?\s*(누구|뭐\s*하는|뭐[야니냐]|정체)|[너네니]\s*이름", qk):
        return f"저는 {persona}입니다."
    if re.search(r"[내제]\s*이름|나\s*이름|내가\s*누구", qk):
        return f"당신 이름은 {name}이에요." if name else None
    if re.search(r"(나|내가|저)\s*(지금)?\s*어디(에)?\s*(있|야|지|인지)|내\s*위치|내가\s*있는\s*[곳데]", qk):
        return f"지금 {place}에 계세요." if place else None
    if re.search(r"[내제]\s*(시간대|타임존)", qk):
        return f"사용 중인 시간대는 {ctx.get('timezone')}입니다." if ctx.get("timezone") else None
    return None


class RealtimeBackend:
    def __init__(self, cache=True, timeout=8.0, deadline=None, fast=True):
        if deadline is None:
            deadline = float(os.environ.get("MOSHICP_RAG_DEADLINE_S", "1.0"))
        self.timeout = timeout          # per-request socket timeout
        self.deadline = deadline        # hard wall-clock budget for the hot path (serving)
        self.fast = fast                # serving: parallel-race fast sources, cut at deadline
        self.cache = cache
        if cache:
            os.makedirs(_CACHE_DIR, exist_ok=True)
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = _UA
        self.last_source = None    # which source produced the last answer (pipeline logging)
        self.last_args = None      # args of the last MCP/tool call (None if not a tool) — tool-use capture
        self.last_trace = []       # [{src, hit, detail}] sources attempted on the last retrieve
        # MoshiRAG-style RAG: an LLM reads the conversation context and GENERATES a concise factual
        # reference (vs our raw DuckDuckGo/Wiki top-hit). Resolves pronouns + abstains when unknown.
        self.mcp_on = os.environ.get("MOSHICP_MCP", "1") not in ("0", "off", "false", "")
        self.llm_on = os.environ.get("MOSHICP_RAG_LLM", "1") not in ("0", "off", "false", "")
        self.llm_url = os.environ.get("MOSHICP_RAG_LLM_URL", "http://localhost:11434/v1/chat/completions")
        self.llm_model = os.environ.get("MOSHICP_RAG_LLM_MODEL", "llama3.2:3b")
        self.llm_timeout = float(os.environ.get("MOSHICP_RAG_LLM_TIMEOUT", "6.0"))
        # gpt-5.x / o-series "reasoning" endpoints (e.g. gpt-5.6-luna) reject `max_tokens`,
        # `temperature != 1`, and Ollama's `keep_alive`; they require `max_completion_tokens`.
        # Auto-detect by model name; override with MOSHICP_RAG_LLM_REASONING=1/0.
        _rz = os.environ.get("MOSHICP_RAG_LLM_REASONING", "").strip().lower()
        if _rz in ("1", "true", "on"):
            self.llm_reasoning = True
        elif _rz in ("0", "false", "off"):
            self.llm_reasoning = False
        else:
            self.llm_reasoning = bool(re.search(r"(gpt-5|luna|\bo[1-9])", self.llm_model, re.I))
        self.llm_max_tokens = int(os.environ.get("MOSHICP_RAG_LLM_MAX_TOKENS", "120"))
        # MoshiRAG Table 15 summarization variant; MoshiRAG_summ wins on math (Table 3).
        # OFF by default: unset env -> behaviour is byte-identical to the non-summarizing backend.
        self.ref_summ = os.environ.get("MOSHICP_REF_SUMM", "0").strip().lower() in ("1", "true", "on", "yes")
        self.summ_max_tokens = int(os.environ.get("MOSHICP_REF_SUMM_MAX_TOKENS", "120"))
        # Remote OpenAI-compatible endpoints (GPT-Luna) need a bearer key; localhost ollama/vllm
        # ignore the header harmlessly. Attach when a key env is present.
        _key = (os.environ.get("MOSHICP_RAG_LLM_KEY") or os.environ.get("API_BACKEND_KEY")
                or os.environ.get("OPENAI_API_KEY"))
        if _key:
            self.sess.headers["Authorization"] = "Bearer " + _key

    def _trace(self, src, hit, detail=""):
        self.last_trace.append({"src": src, "hit": bool(hit), "detail": str(detail)[:120]})
        if hit:
            self.last_source = src

    _LLM_SYS = (
        "You are the retrieval backend of a voice assistant (MoshiRAG-style). Read the conversation "
        "context and output ONE concise, factual reference sentence that directly helps the assistant "
        "answer the LATEST user turn. Resolve pronouns ('that','this','he') from the context. If you do "
        "not know, or the answer needs real-time data you lack (live stock/crypto prices, current "
        "weather, exact current time, today's news), output EXACTLY: (no information found). "
        "Never answer by repeating the assistant's own earlier sentence, restating the request, or "
        "confirming an action you cannot perform (bookings, reservations, purchases) — those are "
        "also (no information found). "
        # MoshiRAG Table 15 reference-writing clauses (2026-08-11 audit: these three were the
        # only ones missing here, and train-time REFs already obey them — see onepass.ONEPASS_SYS
        # "Reference (REF)": one line, under fifty words, plain prose).
        "Use no more than fifty words. Do not use newlines. Do not use markdown. "
        "Output ONLY the reference sentence — no preamble, no questions, no chat."
    )

    def _llm_reference(self, context):
        """MoshiRAG-style: prompt an instruct LLM to read `context` and write a reference (or abstain).
        Returns the reference string, NO_INFO if the LLM abstains, or None if the LLM is unreachable
        (so the caller can fall back to web search)."""
        try:
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "system", "content": self._LLM_SYS},
                             {"role": "user", "content": f"Conversation context:\n{context}"}],
            }
            if self.llm_reasoning:
                # reasoning models budget hidden reasoning tokens too → higher floor; default temp(1)
                payload["max_completion_tokens"] = max(self.llm_max_tokens, 256)
            else:
                payload.update({"temperature": 0, "max_tokens": self.llm_max_tokens,
                                "keep_alive": -1})   # 모델 상주 고정(콜드 방지)
            r = self.sess.post(self.llm_url, json=payload, timeout=self.llm_timeout)
            txt = ((r.json().get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
            txt = txt.strip('"').strip()
            hit = bool(txt) and "no information found" not in txt.lower()
            self._trace(f"llm:{self.llm_model}", hit, txt)
            return txt if hit else NO_INFO
        except Exception as e:
            self._trace(f"llm:{self.llm_model}(unreachable)", False, str(e))
            return None

    # ── MoshiRAG Table 15 summarization variant; MoshiRAG_summ wins on math (Table 3) ─────
    # Table 15's "Reference LLM Prompt" carries two underlined clauses that apply ONLY to the
    # "reference LLM with summarization" configuration. Both are ported VERBATIM below, together
    # with the surrounding reference-writing guidelines they qualify. The paper's hypothesis:
    # raw LLM references carry excessive numerical detail, symbolic expressions and lengthy
    # reasoning, which makes knowledge integration less effective; compressing them lets the
    # speech LM focus on the core concepts (Section 5.4).
    _SUMM_SYS = (
        "You are generating a short reference document for a chatbot named “moshi”. "
        "This document will directly inform moshi’s next response in an ongoing conversation "
        "between a user and Moshi. "
        "Guidelines: "
        "- The reference document should be concise, factual, and directly relevant to the ongoing "
        "conversation. "
        "- The reference document must be helpful for moshi to generate its responses in the next "
        "turn of the conversation. "
        "- Avoid complex or unreadable punctuations and symbols. "
        "- Do not use markdown like asterisks (*) or hashes (#) within the reference content itself. "
        "- Each reference document should contain no more than fifty words. "
        "- Do not include any newline symbol in your results. "
        # ↓ the two Table 15 clauses that are underlined ("only for reference LLM with summarization")
        "- After generating the reference, please summarize it to include only information that is "
        "relevant to the conversation and is helpful for moshi to generate its responses. "
        "- The final summarized reference should be concise, labeled as "
        "`Summarized reference: [summarized reference content]`. "
        # output contract for this backend: we already hold the reference, we want only the summary
        "The reference document has already been generated and is given to you below. "
        "Output ONLY the final `Summarized reference: [summarized reference content]` line."
    )
    _SUMM_LABEL = re.compile(r"^\s*(summarized\s+reference|reference)\s*:\s*", re.I)

    def _summarize_reference(self, ref, context=""):
        """ONE extra LLM call that compresses an already-produced reference (MoshiRAG_summ).
        Returns the summarized reference, or the untouched `ref` if the call fails/returns junk."""
        try:
            user = (f"Conversation context:\n{context}\n\nReference: {ref}" if context
                    else f"Reference: {ref}")
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "system", "content": self._SUMM_SYS},
                             {"role": "user", "content": user}],
            }
            if self.llm_reasoning:
                payload["max_completion_tokens"] = max(self.summ_max_tokens, 256)
            else:
                payload.update({"temperature": 0, "max_tokens": self.summ_max_tokens,
                                "keep_alive": -1})
            r = self.sess.post(self.llm_url, json=payload, timeout=self.llm_timeout)
            txt = ((r.json().get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
            txt = self._SUMM_LABEL.sub("", txt.strip('"').strip()).strip().strip('"').strip()
            txt = re.sub(r"\s+", " ", txt)
            # traced, but NOT via _trace(): summarization is a post-step, so `last_source` must
            # keep naming the source that actually produced the knowledge (llm/web/cache).
            if not txt or NO_INFO in txt.lower():
                self.last_trace.append({"src": "summ", "hit": False, "detail": str(txt)[:120]})
                return ref
            self.last_trace.append({"src": "summ", "hit": True, "detail": str(txt)[:120]})
            return txt
        except Exception as e:
            self.last_trace.append({"src": "summ(unreachable)", "hit": False, "detail": str(e)[:120]})
            return ref

    def _maybe_summarize(self, ref, context=""):
        """Gate for the Table 15 summarization variant. Tool results and abstentions BYPASS:
        the paper summarizes knowledge/LLM references only, and a tool result is already a
        single spoken clause of live data that compression can only corrupt."""
        if not self.ref_summ or not ref:
            return ref
        if ref == NO_INFO or "no information found" in ref.lower():
            return ref
        if ref.startswith("(tool result)"):
            return ref
        return self._summarize_reference(ref, context)

    def _llm_chat_raw(self, system, user, max_tokens=60):
        """One bare chat call on the reference-LLM endpoint (same url/model/timeout). Returns the
        content string or None on any failure — callers must treat None as 'skip'."""
        payload = {"model": self.llm_model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": 0, "max_tokens": max_tokens, "keep_alive": -1}
        r = self.sess.post(self.llm_url, json=payload, timeout=self.llm_timeout)
        return ((r.json().get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()

    def _resolve_agent_subject(self, q, persona):
        """Rewrite an agent-addressed question ('your latest song') into third person with the
        persona's explicit subject ('the latest song by A2O May'), so the tool router and the
        web/RAG paths operate on a resolvable query. Falls back to the original q untouched."""
        try:
            out = self._llm_chat_raw(
                "Rewrite the user's question in the third person, replacing 'you/your' with the "
                "most publicly searchable subject from this identity: " + str(persona) + ". "
                "If the identity is a member of a group/band and the question concerns songs, "
                "releases, or concerts, use the GROUP's FULL name VERBATIM as the subject "
                "(never shorten or split it). Keep it a single short question. Output ONLY "
                "the rewritten question.",
                q, max_tokens=60)
            out = (out or "").strip().strip('"')
            if out and "?" in out and len(out) < 200:
                self.last_trace.append({"src": "subj-resolve", "hit": True, "detail": out[:120]})
                return out
        except Exception as e:
            self.last_trace.append({"src": "subj-resolve", "hit": False, "detail": repr(e)[:80]})
        return q

    def _rag(self, q):
        """RAG path: MoshiRAG-style LLM reference (primary), DuckDuckGo/Wikipedia fallback if the LLM
        is unreachable. An LLM abstention (NO_INFO) is respected (clean abstain, not web junk)."""
        if self.llm_on:
            ref = self._llm_reference(q)
            if ref is not None and ref != NO_INFO:
                return self._maybe_summarize(ref, q)
            if ref == NO_INFO:
                # LLM abstained. For LONG-TAIL ENTITIES (new group, niche place) the abstain is a
                # knowledge-cutoff artifact and the web has the answer (owner 2026-08-12: "구글
                # 기준으로는 매우 쉽게 찾긴 했어"). Try the web chain — but gate on RELEVANCE:
                # the wiki source's high-recall failure mode returned a generic list page for a
                # concert query (measured: "2026 in music in South Korea"), and junk is worse
                # than a clean abstain. Require a rare query token to appear in the result.
                w = self._web(q)
                if w and w != NO_INFO:
                    _rare = {t for t in re.findall(r"[a-z0-9]{3,}", q.lower())
                             if t not in ("what", "when", "where", "who", "the", "next",
                                          "latest", "new", "newest", "recent", "song",
                                          "concert", "album", "track", "release", "does",
                                          "did", "will", "are", "is", "was", "how")}
                    if _rare and not (_rare & set(re.findall(r"[a-z0-9]{3,}", w.lower()))):
                        self._trace("web(irrelevant)", False, w[:80])
                        return NO_INFO
                    return self._maybe_summarize(w, q)
                return NO_INFO
        return self._maybe_summarize(self._web(q), q)   # LLM off/unreachable → raw web

    # ── public API (drop-in for retrieve.Backend.retrieve) ───────────────────────────────
    def retrieve(self, query, kind="auto", ctx=None, aux_context=None, history=None,
                 convo=None):
        """Fluid MoshiRAG-style routing from ONE ASR transcript:
          clean transcript -> LLM tool router (MCP) ; if no live tool fires -> RAG enriched with
          `aux_context` (the Moshi inner-monologue, for pronoun/subject resolution) ; else abstain.
        The CLEAN transcript drives the tool picker (a context prefix leaks stale city/subject names
        that mis-route tools); only the RAG fallback gets the enriched query.

        ctx (optional dict from ContextProfile.as_dict()) grounds the answer in the user's
        identity/place: time & weather with no city spoken resolve to the user's timezone/city,
        and personal questions ('what's my name', 'where am I') are answered from context."""
        q = (query or "").strip()
        self.last_source = None
        self.last_args = None
        self.last_trace = []
        if not q:
            return NO_INFO
        # AGENT-SUBJECT resolution (possessive 'your/yours' + a session persona, 2026-08-12
        # fan→artist scenario): "your latest song" is about the roleplayed identity's world.
        # Measured failure modes of the alternatives: (a) routing the raw q → the router's
        # llm-direct path broke character ("I am an AI"); (b) bypassing the router → the real
        # music/web tools never got their shot at a REAL group. The fix is subject RESOLUTION:
        # one fast LLM rewrite turns the question third-person ("latest song by A2O May"),
        # then the NORMAL pipeline (tool router + RAG + web) runs at full strength. Identity
        # questions ("who are you") were already direct-answered above. Never cached under
        # the persona-less q.
        _pers = (ctx or {}).get("persona") if isinstance(ctx, dict) else None
        _to_agent = bool(_pers) and bool(re.search(r"\b(your|yours)\b", (q or "").lower()))
        if _to_agent:
            q = self._resolve_agent_subject(q, _pers)
        # personal questions answered from context (no web, never cached)
        # (2026-08-25 단일 결정자: 서빙에선 이 선행 층도 안 탄다 — "do you remember what i
        # asked?" 같은 기억 질문을 가로채 페르소나 원문을 통짜 주입한 실사고. 프로필·이력은
        # convo로 라우터에 이미 제공되므로 정체성/개인 질문도 라우터 llm-direct가 답한다.)
        if ctx and not self.fast:
            pa = _personal_from_ctx(q, ctx)
            if pa:
                self._trace("context-db(profile)", True, pa)
                return pa
        # MULTI-INTENT fast path (2026-08-13, owner "둘 다 잘 대답"): time+weather in ONE utterance
        # ("what time is it? and how's the weather?") — the router picks a single tool and the old
        # _route elif suppressed time under weather. Both are keyless REAL lookups, so serve one
        # COMBINED span directly, before the router. (This lives in retrieve(), the LIVE path —
        # the twin patch in _route() turned out to be a dead branch for serving.)
        # (2026-08-25 유저 설계: "router LLM에게 정보를 줘서 툴콜링하게 — 지엽 구현 금지")
        # 종전의 시계/날씨 선행 결정 분기는 라우터보다 먼저 실행돼 누적 전사의 'time' 토큰이
        # 모든 질문을 하이재킹했다(주가 질문에 시각 주입 실사고). 이제 라우터 LLM이 1순위로
        # 도구·인자(프로필 기반 city/timezone 포함)를 채우고, 이 결정 분기는 **라우터가 아예
        # 없는 배포**(mcp_on=False)의 최후 안전망으로만 남는다 (2026-08-13 실사고 대비).
        _tq = _tok(q)
        if not self.mcp_on:
            if (_tq & _WEATHER_WORDS) and (_tq & _TIME_WORDS):
                _t = self._tool_time(q, ctx)
                _w = self._tool_weather(q, ctx)
                if _t and _w:
                    _ts = _t.replace("(tool result) ", "", 1).rstrip(". ")
                    _ws = _w.replace("(tool result) ", "", 1)
                    self._trace("mcp:time+weather", True, _ts[:40])
                    return self._as_tool_result(_ts + ". " + _ws)
                if _t or _w:
                    return _t or _w
            elif _tq & _TIME_WORDS:
                _t = self._tool_time(q, ctx)
                if _t:
                    return _t
            elif (_tq & _WEATHER_WORDS) and ctx:
                _w = self._tool_weather(q, ctx)
                if _w:
                    return _w
        # REAL MCP tool router (live MCP servers: time / weather / finance) — primary for real-time
        # intents. The LLM router picks a tool + args (ctx fills timezone/city defaults) or no tool.
        if self.mcp_on:
            try:
                from contextspan.duetaspan.runtime.mcp.client import (
                    mcp_route as _mcp_route, _has_search_intent as _has_search_intent_q)
                # convo = Context DB working_text() 스냅샷(대화 누적 상태) — 라우터·인자
                # 채움의 1급 입력: ASR 윈도우 밖의 앞선 도구 결과/값을 체인이 참조한다.
                m = _mcp_route(q, ctx, history, aux=aux_context, convo=convo)
                if m and m.get("answer"):
                    # NON-CANONICAL ABSTAIN NORMALIZATION (2026-08-12, ritsuje session 5): the
                    # llm-direct path sometimes "answers" with an apology ("I am sorry, but I
                    # could not find any information regarding …") instead of the canonical
                    # NO_INFO. Returning that verbatim (a) injects a useless span and (b) EATS
                    # the downstream shots (music tool, web fallback, relevance-gated RAG).
                    # Detect abstain-ish phrasing → fall through to the rest of the chain.
                    _ans = m["answer"]
                    if re.search(r"(?:could\s*n[o']t\s+find|cannot\s+find|no\s+information|"
                                 r"i\s+do\s*n[o']t\s+have\s+(?:any\s+)?information|"
                                 r"unable\s+to\s+find)", _ans, re.I):
                        self._trace("llm-direct(abstain-ish)", False, _ans[:80])
                    else:
                        self._trace("llm-direct(1call)", _ans != NO_INFO, _ans)
                        return _ans
                # web_search는 일반지식 오남용 방지로 제외해 왔지만, 사용자가 명시적으로
                # 검색을 요청한 경우("검색해줘", "search the web")는 그 결과가 곧 정답이다 —
                # 제외하면 _rag의 뉴스-abstain 규칙에 걸려 명시 요청까지 죽는다.
                if m and m.get("tool") and (m.get("tool") != "web_search"
                                            or _has_search_intent_q(q)):
                    self._trace(f"mcp:{m['tool']}", True, m.get("reference", ""))
                    self.last_args = m.get("args")   # tool-use capture: surface router args

                    # CONTRACT (training == generation == inference): every tool-produced
                    # span is "(tool result) <spoken line>" — the built-in live tools and
                    # the generated training data both use this prefix; the MCP-router path
                    # was the one place that didn't. RAG spans stay prefix-less.
                    return self._as_tool_result(m["reference"])
            except Exception as e:
                self._trace("mcp(unavailable)", False, str(e))
            # ── 단일 결정자 (2026-08-25 유저 설계 확정: "router LLM에게 맡겨라, 레이어를
            # 부풀리지 마라"): 서빙 경로에서 라우팅 결정은 라우터 LLM이 유일하게 내린다 —
            # 툴콜 / 직접답변 / 아무것도 안 함(NO_INFO→빈쌍). 아래의 RAG·웹 레이스 폴백은
            # 라우터의 옳은 no-tool 결정을 뒤집고 무관 스팬을 물어온 실사고(뉴욕 시간 재질의
            # → 위키 O.J. Simpson 주입)의 근원이라 서빙에선 타지 않는다. 오프라인 데이터젠
            # (fast=False)은 종전 전체 체인 유지.
            if self.fast:
                self._trace("router-final(no-tool)", False, q[:60])
                return NO_INFO
        # context-reading LLM-RAG (MoshiRAG-style: resolves pronouns, abstains; web fallback).
        # ctx-sensitive intents bypass the shared cache so they never serve a stale/other-user answer.
        cacheable = (not self._ctx_sensitive(q, ctx)) and not _to_agent
        if cacheable:
            ck = self._cache_get(q)
            if ck is not None:
                self._trace("cache", True, ck)
                return ck
        # RAG fallback only: enrich the query with the Moshi inner-monologue so pronouns/subjects
        # resolve ("when did it release?" -> "...the Nintendo DS..."). Cache stays keyed on clean q.
        rag_q = (q + " " + aux_context).strip() if aux_context else q
        try:
            out = self._rag(rag_q)
        except Exception:
            out = NO_INFO
        out = out or NO_INFO
        if cacheable and out != NO_INFO:
            self._cache_put(q, out)
        return out

    fetch = retrieve  # alias

    @staticmethod
    def _ctx_sensitive(q, ctx):
        toks = _tok(q)
        if toks & _TIME_WORDS:                       # clock moves → never cache
            return True
        if ctx and (toks & _WEATHER_WORDS):          # user-location weather → don't cross-cache
            return True
        return False

    @staticmethod
    def _is_mcp_intent(query):
        return bool(_tok(query) & _MCP_WORDS)

    # ── routing ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _last_question(q):
        """누적 전사 방어 (2026-08-25 실사고: 40s 연속발화의 전사 전체가 들어와 첫 질문의
        'time' 토큰이 인텐트 게이트를 선점 → 주가 질문에 시각이 주입됨). 라우팅 대상은 항상
        '가장 최근 질문' 하나다(유저 지시) — 마지막 문장부호 뒤 꼬리가 실질 질문이면 그것,
        아니면(꼬리가 공백/한두 단어) 마지막 완결 문장을 쓴다. 단문 입력은 그대로 통과."""
        parts = [p.strip() for p in re.split(r"[?!.]", q or "") if p.strip()]
        if len(parts) <= 1:
            return q
        tail = parts[-1]
        return tail if len(tail.split()) >= 3 else parts[-2]

    def _route(self, q, kind, ctx=None):
        q = self._last_question(q)
        toks = _tok(q)
        if kind == "rag":
            return self._rag(q)
        if kind == "mcp" or (kind == "auto" and self._is_mcp_intent(q)):
            if (toks & _WEATHER_WORDS) and (toks & _TIME_WORDS):
                # multi-intent single utterance ("what time is it? and how's the weather?"):
                # both are keyless REAL lookups, so serve ONE combined span and the model can
                # answer both (2026-08-13; the old elif suppressed time whenever weather won).
                t = self._tool_time(q, ctx)
                w = self._tool_weather(q, ctx)
                if t and w:
                    tt = t.replace("(tool result) ", "", 1).rstrip(". ")
                    ww = w.replace("(tool result) ", "", 1)
                    self._trace("mcp:time+weather", True, tt[:40])
                    return self._as_tool_result(tt + ". " + ww)
                if t or w:
                    return t or w
            if toks & _WEATHER_WORDS:
                r = self._tool_weather(q, ctx)
                if r:
                    return r
            # news intent → Google News RSS (the fast _web race only runs ddg/wiki, which
            # mismatch a bare "latest news" query). Checked before time so an incidental
            # "news today" routes to news, not the clock. Fall through to web if it yields nothing.
            if toks & {"news", "headlines", "breaking"}:
                try:
                    nr = self._news(q)
                except Exception:
                    nr = None
                if nr:
                    return self._normalize(nr)
            elif (toks & _TIME_WORDS) and not (toks & _WEATHER_WORDS):
                return self._tool_time(q, ctx)
            if toks & _STOCK_WORDS:                    # 실시간 주가 (Yahoo Finance, 무키)
                r = self._tool_stock(q)
                if r:
                    return r
            if toks & self._MUSIC_WORDS:               # 최신곡/발매 (iTunes Search, 무키 실API)
                r = self._tool_music(q)
                if r:
                    return r
            # other realtime intents (score/flight…) → LLM reference (abstains when it lacks
            # the live data), framed as tool result
            w = self._rag(q)
            return self._as_tool_result(w) if w and w != NO_INFO else NO_INFO
        return self._rag(q)

    @staticmethod
    def _as_tool_result(text):
        t = text.strip()
        return t if t.startswith("(tool result)") else f"(tool result) {t}"

    # ── live tools ─────────────────────────────────────────────────────────────────────
    def _tool_time(self, q, ctx=None):
        # Resolve the queried city's timezone so "time in Tokyo" gives Tokyo's clock, not the
        # server's. With no city named, use the USER's timezone from ctx (so "what time is it"
        # from Korea answers in Asia/Seoul, not the server's clock). Server-local is the last resort.
        city = self._extract_city(q)
        if city:
            loc = self._geocode(city)
            tzname = (loc or {}).get("timezone")
            place = (loc or {}).get("name", city)
            s = _fmt_time(tzname, place)
            if s:
                self._trace(f"local-clock(geocoded:{place})", True, s)
                return s
        if ctx:                                          # no city spoken → user's own timezone/place
            s = _fmt_time(ctx.get("timezone"), _ctx_place(ctx))
            if s:
                self._trace(f"local-clock(ctx-tz:{ctx.get('timezone')})", True, s)
                return s
        # (2026-08-25 유저 설계 확정: "기본 tz는 UTC로 하되, '내' 정보(Context DB 프로필)가
        # 명확하게 들어가서 한국시간으로 '바꿔서' 가져오는 능력을 기르게") — 특정 지역을
        # 하드코딩하지 않는다. 유저 tz는 위의 ctx(ContextProfile.timezone) 경로가 정본이고,
        # 도시·ctx가 전무한 마지막 폴백만 UTC로 명시해 답한다(서버 소재지 시계 금지).
        _dtz = os.environ.get("MOSHICP_DEFAULT_TZ", "UTC")
        if _dtz:
            s = _fmt_time(_dtz, "UTC" if _dtz == "UTC" else "")
            if s:
                self._trace(f"local-clock(default-tz:{_dtz})", True, s)
                return s
        now = _dt.datetime.now()
        s = f"(tool result) It is {now.strftime('%-I:%M %p').lower()} on {now.strftime('%A, %B %-d')}."
        self._trace("local-clock(server)", True, s)
        return s

    # City names rarely run past these qualifiers; stop extraction when one appears so
    # "weather in Paris today" -> "Paris" (not "Paris today", which geocodes to nothing).
    _CITY_STOP = {
        "today", "tonight", "tomorrow", "yesterday", "now", "right", "currently", "current",
        "this", "next", "week", "weekend", "morning", "afternoon", "evening", "night",
        "please", "on", "at", "for", "in", "the", "is", "be", "like", "going", "gonna",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    }

    def _extract_city(self, q):
        m = re.search(r"\b(?:in|for|at|near)\s+([A-Za-z][\w .,'-]*)", q)
        if not m:
            m = re.search(r"\b(?:weather|temperature|forecast|time|clock)\s+(?:in|for|at)?\s*"
                          r"([A-Za-z][\w .,'-]*)", q, re.I)
        if not m:
            return None
        out = []
        for w in re.split(r"[\s,]+", m.group(1)):
            wl = w.strip(" .?!").lower()
            if not wl:
                continue
            if wl in self._CITY_STOP:
                break
            out.append(w.strip(" .?!"))
            if len(out) >= 3:        # e.g. "New York City"
                break
        city = " ".join(out).strip(" .,?!")
        return city or None

    def _geocode(self, city):
        try:
            g = self.sess.get("https://geocoding-api.open-meteo.com/v1/search",
                              params={"name": city, "count": 1}, timeout=self.timeout)
            res = (g.json() or {}).get("results") or []
            return res[0] if res else None
        except Exception:
            return None

    def _tool_weather(self, q, ctx=None):
        # City spoken → geocode it. Otherwise fall back to the USER's location from ctx: prefer
        # their exact lat/lon (from geolocation), else geocode their city.
        city = self._extract_city(q)
        lat = lon = None
        name = city
        if city:
            loc = self._geocode(city)
            if loc:
                lat, lon, name = loc["latitude"], loc["longitude"], loc.get("name", city)
        elif ctx:
            if ctx.get("lat") is not None and ctx.get("lon") is not None:
                lat, lon, name = ctx["lat"], ctx["lon"], _ctx_place(ctx) or "your area"
            elif ctx.get("city"):
                loc = self._geocode(ctx["city"])
                if loc:
                    lat, lon, name = loc["latitude"], loc["longitude"], loc.get("name", ctx["city"])
        if lat is None or lon is None:
            return None
        f = self.sess.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon,
                                 "current": "temperature_2m,weather_code"}, timeout=self.timeout)
        cur = (f.json() or {}).get("current") or {}
        if "temperature_2m" not in cur:
            return None
        temp = round(cur["temperature_2m"])
        cond = _WMO.get(int(cur.get("weather_code", -1)), "")
        cond_str = f" and {cond}" if cond else ""
        s = f"(tool result) It's {temp} degrees Celsius{cond_str} in {name} right now."
        self._trace(f"open-meteo({name})", True, s)
        return s

    _STOCK_DROP = {"what", "whats", "what's", "the", "of", "is", "are", "now", "current", "currently",
                   "today", "right", "much", "how", "tell", "me", "a", "an", "for", "s", "and",
                   "please", "do", "you", "know", "whats", "stock", "stocks", "share", "shares",
                   "price", "ticker", "trading", "worth", "value", "quote", "cost", "market"}

    _MUSIC_WORDS = {"song", "single", "album", "track", "release", "released", "discography", "ep"}

    def _tool_music(self, q):
        """REAL iTunes Search API (keyless) — the policy table's designated music backend.
        Extracts the artist from a resolved query ("A2O May's latest song" / "latest song by
        A2O May"), returns the newest release by date. None when no artist parses or no rows."""
        m = (re.search(r"(?:latest|new(?:est)?|recent)\s+(?:song|single|album|track|release)s?\s+"
                       r"(?:by|from|of)\s+(.+?)\s*\??$", q, re.I)
             or re.search(r"^(?:what\s+is\s+)?(.+?)'s\s+(?:latest|new(?:est)?|recent)\s+"
                          r"(?:song|single|album|track|release)", q, re.I))
        if not m:
            return None
        artist = m.group(1).strip().strip('"')
        try:
            r = self.sess.get("https://itunes.apple.com/search",
                              params={"term": artist, "entity": "song", "limit": 25}, timeout=5)
            rows = [x for x in (r.json().get("results") or [])
                    if artist.lower() in (x.get("artistName") or "").lower()]
            if not rows:
                return None
            top = max(rows, key=lambda x: x.get("releaseDate") or "")
            date = (top.get("releaseDate") or "")[:10]
            self._trace("music:itunes", True, f"{top.get('artistName')} — {top.get('trackName')}")
            return self._as_tool_result(
                f"The latest song by {top.get('artistName')} is '{top.get('trackName')}'"
                + (f", released {date}." if date else "."))
        except Exception as e:
            self._trace("music:itunes", False, str(e)[:80])
            return None

    def _tool_stock(self, q):
        """Real-time stock price via Yahoo Finance (key-free): resolve company→symbol, fetch quote."""
        name = " ".join(w for w in re.findall(r"[A-Za-z][A-Za-z&.\-]*", q)
                        if w.lower() not in self._STOCK_DROP).strip()
        if not name:
            return None
        try:
            s = self.sess.get("https://query1.finance.yahoo.com/v1/finance/search",
                              params={"q": name}, timeout=self.timeout)
            quotes = [x for x in ((s.json() or {}).get("quotes") or [])
                      if x.get("quoteType") == "EQUITY" and x.get("symbol")]
            if not quotes:
                return None
            q0 = quotes[0]
            sym = q0["symbol"]
            longname = q0.get("longname") or q0.get("shortname") or name
            c = self.sess.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                              timeout=self.timeout)
            meta = ((((c.json() or {}).get("chart") or {}).get("result") or [{}])[0].get("meta") or {})
            price = meta.get("regularMarketPrice")
            if price is None:
                return None
            cur = meta.get("currency", "")
            pstr = f"{price:,.0f}" if price >= 100 else f"{price:,.2f}"
            chg = meta.get("regularMarketChangePercent")
            chg_str = ""
            if chg is None and meta.get("chartPreviousClose"):
                try:
                    chg = (price / meta["chartPreviousClose"] - 1) * 100
                except Exception:
                    chg = None
            if chg is not None:
                chg_str = f" ({'+' if chg >= 0 else ''}{chg:.1f}% today)"
            out = f"(tool result) {longname} ({sym}) is trading at {pstr} {cur}{chg_str} right now."
            self._trace(f"yahoo-finance({sym})", True, out)
            return out
        except Exception:
            return None

    # ── live web (key-free) ──────────────────────────────────────────────────────────────
    def _web(self, q):
        """Hot path (serving, fast=True): race fast sources, hard cut at self.deadline.
        Offline (fast=False, data-gen): full sequential chain, no time limit, fills cache."""
        if self.fast:
            # Both run in parallel within the deadline; PREFER DDG instant-answer (precise direct
            # answer) over wiki full-text (high recall but can grab a related/wrong article).
            import concurrent.futures as _cf
            ex = _cf.ThreadPoolExecutor(max_workers=3)
            try:
                f_ia, f_wiki = ex.submit(self._ddg_instant, q), ex.submit(self._wiki, q)
                f_srch = ex.submit(self._ddg_search_row, q)
                _cf.wait([f_ia, f_wiki, f_srch], timeout=self.deadline)

                def _safe(f):
                    if not f.done():
                        return None
                    try:
                        return f.result()
                    except Exception:   # one source failing must not void the other's answer
                        return None

                ia, wk, sr = _safe(f_ia), _safe(f_wiki), _safe(f_srch)
            finally:
                # NEVER join: a with-block would block on the slowest source and defeat the
                # deadline entirely (the search adapter can retry for tens of seconds).
                ex.shutdown(wait=False, cancel_futures=True)
            self._trace("duckduckgo-instant", bool(ia), ia or "")
            self._trace("wikipedia", bool(wk), wk or "")
            self._trace("duckduckgo-search", bool(sr), sr or "")
            ans = ia or wk or sr
            return self._normalize(ans) if ans else NO_INFO
        # full chain (offline data-gen): clean sources only; ddg_html/lite return ad junk → excluded.
        for src in (lambda: self._wiki(q), lambda: self._ddg_instant(q),
                    lambda: self._news(q) if (_tok(q) & {"news", "headlines", "breaking", "latest"}) else None):
            try:
                ans = src()
            except Exception:
                ans = None
            if ans:
                return self._normalize(ans)
        return NO_INFO

    @staticmethod
    def _race(funcs, deadline):
        """Run funcs in parallel threads; return first non-empty result within `deadline` s."""
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=len(funcs)) as ex:
            futs = [ex.submit(f) for f in funcs]
            try:
                for fut in _cf.as_completed(futs, timeout=deadline):
                    try:
                        r = fut.result()
                    except Exception:
                        r = None
                    if r:
                        return r
            except _cf.TimeoutError:
                pass
        return None

    _WIKI_STOP = re.compile(r"^(who|what|whats|when|where|why|how|which|whose|is|are|was|were|do|does|"
                            r"did|can|could|tell|me|about|the|a|an|of|in|on|please)\b[\s,'’]*", re.I)

    def _wiki_query(self, q):
        """Strip leading question/filler words so OpenSearch matches the entity title."""
        s = q.strip().rstrip("?.!")
        prev = None
        while s and s != prev:                                  # peel leading stopwords
            prev = s
            s = self._WIKI_STOP.sub("", s).strip()
        s = re.sub(r"\b(voiced|voice of|played|starred in|located|right now|currently)\b", "", s, flags=re.I)
        if re.search(r"[가-힣]", s):
            # Korean: drop interrogative/function words so full-text search sees the entity
            # ("베수비오 화산이 마지막으로 분화한 게 언제야?" -> "베수비오 화산이 마지막으로 분화한").
            drop = re.compile(r"^(언제|어디|누구|뭐|무엇|몇|얼마|어때|왜|게|건|거|그게|이게|지금|혹시|좀|요즘|어떻게)")
            s = " ".join(w for w in s.split() if not drop.match(w))
        return re.sub(r"\s+", " ", s).strip() or q

    def _wiki(self, q):
        """Wikipedia FULL-TEXT search (srsearch handles question phrasing) → top article summary.
        A Hangul query searches ko.wikipedia (the EN index simply misses Korean questions)."""
        lang = "ko" if re.search(r"[가-힣]", q) else "en"
        s = self.sess.get(f"https://{lang}.wikipedia.org/w/api.php",
                         params={"action": "query", "list": "search", "srsearch": self._wiki_query(q),
                                 "srlimit": 1, "format": "json"}, timeout=self.timeout)
        try:
            data = s.json() or {}
        except ValueError:                      # blocked/ratelimited HTML body
            return None
        hits = ((data.get("query") or {}).get("search")) or []
        if not hits:
            return None
        title = hits[0]["title"]
        r = self.sess.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/" +
                         urllib.parse.quote(title.replace(" ", "_")), timeout=self.timeout)
        ex = (r.json() or {}).get("extract")
        return ex or None

    def _ddg_search_row(self, q):
        """DuckDuckGo HTML search via the bank's search adapter (ad-filtered, warmed session) —
        the one open-web source reachable when en.wikipedia is blocked on this network.
        Returns the abstract when present, else the first hit's title+snippet."""
        try:
            from contextspan.duetaspan.runtime.mcp.adapters_search import search as _srch
            out = _srch(q, max_results=2)
        except Exception:
            return None
        if not out or out == NO_INFO:
            return None
        row = out.split(" ; ")[0].strip()
        row = re.sub(r"^answer:\s*", "", row)
        row = re.sub(r"\s*—\s*https?://\S+:", ":", row)
        return row or None

    def _ddg_lite(self, q):
        from bs4 import BeautifulSoup
        r = self.sess.post("https://lite.duckduckgo.com/lite/", data={"q": q}, timeout=self.timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        for td in soup.select("td.result-snippet"):
            t = td.get_text(" ", strip=True)
            if t and len(t) > 20:
                return t
        return None

    def _news(self, q):
        """Google News RSS (key-free) → latest headline for realtime news intent."""
        import xml.etree.ElementTree as ET
        r = self.sess.get("https://news.google.com/rss/search",
                         params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}, timeout=self.timeout)
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        if not items:
            return None
        titles = [it.findtext("title") for it in items[:2] if it.findtext("title")]
        return "(tool result) Latest: " + "; ".join(titles) if titles else None

    def _ddg_instant(self, q):
        r = self.sess.get("https://api.duckduckgo.com/",
                         params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
                         timeout=self.timeout)
        try:
            d = r.json()
        except ValueError:              # rate-limited/HTML body — no instant answer
            return None
        for key in ("AbstractText", "Answer", "Definition"):
            v = d.get(key)
            if v and isinstance(v, str) and len(v) > 8:
                return v
        for rt in (d.get("RelatedTopics") or []):
            if isinstance(rt, dict) and rt.get("Text"):
                return rt["Text"]
        return None

    def _ddg_html(self, q):
        from bs4 import BeautifulSoup
        r = self.sess.post("https://html.duckduckgo.com/html/", data={"q": q}, timeout=self.timeout)
        soup = BeautifulSoup(r.text, "html.parser")
        for sn in soup.select(".result__snippet"):
            t = sn.get_text(" ", strip=True)
            if t and len(t) > 20:
                return t
        return None

    # ── normalization → spoken one-liner (same form train & infer) ───────────────────────
    def _normalize(self, text, max_chars=150):
        """Spoken one-clause for Context Span: drop parenthetical/IPA junk, 1 sentence, short."""
        t = re.sub(r"\s+", " ", text).strip()
        t = re.sub(r"\[\d+\]", "", t)                          # citation markers
        t = re.sub(r"\s*\([^)]*[ˈ/ⓘ][^)]*\)", "", t)          # IPA / pronunciation parentheticals
        t = re.sub(r"\s{2,}", " ", t).strip()
        tool = t.startswith("(tool result)")
        body = t[len("(tool result)"):].strip() if tool else t
        body = re.split(r"(?<=[.!?])\s+", body)[0]             # FIRST sentence only (Context-Span fit)
        if len(body) > max_chars:                              # hard cap → cut at last word boundary
            body = body[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
        return ("(tool result) " + body) if tool else body

    # ── tiny disk cache ──────────────────────────────────────────────────────────────────
    def _cache_path(self, q):
        h = urllib.parse.quote_plus(q.lower())[:120]
        # summarized references live in their own keyspace, so a raw/summ A/B never serves the
        # other variant's answer from disk (default OFF -> unchanged path).
        suffix = ".summ.json" if self.ref_summ else ".json"
        return os.path.join(_CACHE_DIR, h + suffix)

    def _cache_get(self, q):
        if not self.cache:
            return None
        p = self._cache_path(q)
        if os.path.exists(p):
            try:
                return json.load(open(p))["ref"]
            except Exception:
                return None
        return None

    def _cache_put(self, q, ref):
        if not self.cache:
            return
        try:
            json.dump({"q": q, "ref": ref, "t": time.time()}, open(self._cache_path(q), "w"))
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    b = RealtimeBackend(cache=False)
    tests = sys.argv[1:] or [
        "What's the weather in London?",
        "what time is it now",
        "who voiced Zuba in Madagascar 2",
        "what is the capital of Australia",
        "latest news headlines",
    ]
    for qq in tests:
        print(f"\nQ: {qq}\n-> {b.retrieve(qq)!r}")
