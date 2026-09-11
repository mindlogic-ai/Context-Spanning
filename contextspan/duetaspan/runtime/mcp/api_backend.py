# -*- coding: utf-8 -*-
"""GPU-free single-model MCP backend over any OpenAI-compatible chat API.

One wire format (chat/completions + native `tools`) covers the providers we validated
(437 clean/heavy + 37 bench cases): OpenAI gpt-5.x and Gemini via its OpenAI-compat endpoint.
Default model = gpt-5.6-luna (owner-specified, clean 97%/heavy 95%, p50 1207ms).

Flow (same single-model contract as gemini_backend, but provider-agnostic):
  route: transcript + ctx + answered-history --tools--> tool_calls
  exec:  our FastMCP servers via MCPRouter (CPU-only)
  format: SAME model phrases "(tool result) <spoken one-liner>"

Env: API_BACKEND_URL / API_BACKEND_MODEL / API_BACKEND_KEY
     API_BACKEND_REASONING (none|minimal, default none)  API_BACKEND_TIER (e.g. priority)
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

URL = os.environ.get("API_BACKEND_URL", "https://api.openai.com/v1/chat/completions")
MODEL = os.environ.get("API_BACKEND_MODEL", "gpt-5.6-luna")
KEY = os.environ.get("API_BACKEND_KEY", "")
REASONING = os.environ.get("API_BACKEND_REASONING", "none")
TIER = os.environ.get("API_BACKEND_TIER", "")

_SYSTEM = (
    "You are the retrieval backend of a full-duplex voice assistant. Input is a raw live-ASR "
    "transcript: fillers, typos, self-corrections, possibly SEVERAL requests in a row. If a "
    "list of already-answered requests is given, those are done — serve the newest request "
    "NOT answered yet; never re-serve an answered one. "
    "If exactly one registered tool answers it, call that tool (args from the user's words; "
    "user profile fills defaults like city/timezone; omit unknown args). "
    "If no tool applies and it is ordinary general knowledge, ANSWER DIRECTLY from your own "
    "knowledge — fillers/typos don't invalidate a clear request. NEVER call web_search for "
    "knowledge you already have. Only when the request is unintelligible or truncated beyond "
    "recovery, output exactly: (no information found). "
    "FINAL ANSWER: one concise SPOKEN sentence (no markdown). Tool-derived answers start "
    "with '(tool result) '. Plain-knowledge answers have no prefix."
)


class ApiBackend:
    def __init__(self, timeout: float = 20.0):
        if not KEY:
            raise RuntimeError("API_BACKEND_KEY is not set")
        from contextspan.duetaspan.runtime.mcp.client import get_mcp_router
        self._router = get_mcp_router()                     # CPU-only FastMCP servers
        self.timeout = timeout
        self.sess = requests.Session()
        self.last_trace: dict = {}
        self._tools = [{"type": "function", "function": e.get("function", e)}
                       for e in self._router._tool_schemas()]

    def _call(self, messages: list[dict]) -> dict:
        body = {"model": MODEL, "messages": messages, "tools": self._tools,
                "max_completion_tokens": 300, "reasoning_effort": REASONING}
        if TIER:
            body["service_tier"] = TIER
        r = self.sess.post(URL, json=body, headers={"Authorization": f"Bearer {KEY}"},
                           timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    def retrieve(self, query: str, ctx: Optional[dict] = None,
                 history: Optional[list[str]] = None, convo: Optional[str] = None) -> str:
        # convo = ContextDB.working_text() snapshot. Routing must be grounded in the
        # accumulated conversation state, not in a single ASR window — when a window misses the
        # head of a token (FAST99 -> "t99") or fires before its tail ("ABC123" -> "a"), that
        # window alone cannot recover it, but an earlier transcript already loaded into the
        # Context DB resolves it. While this parameter did not exist, framestream's inspect
        # check (_convo_ok) quietly became False and this backend path never once saw the
        # Context DB.
        t0 = time.perf_counter()
        user = ""
        if ctx:
            user += f"User profile: {json.dumps(ctx, ensure_ascii=False)}\n"
        if convo:
            user += ("Conversation state so far (Context DB — AUTHORITATIVE for resolving "
                     "entities, IDs and values already established; prefer these over a "
                     "partial transcript):\n" + convo + "\n")
        if history:
            user += "Already answered (do NOT re-serve):\n" + \
                    "\n".join(f"- {h}" for h in history[-5:]) + "\n"
        user += f"ASR transcript:\n{query}"
        msgs = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]

        m = self._call(msgs)
        tcs = m.get("tool_calls") or []
        if not tcs:                                          # direct answer / abstain
            text = (m.get("content") or "").strip() or "(no information found)"
            self.last_trace = {"model": MODEL, "tool": None,
                               "ms": round((time.perf_counter() - t0) * 1000)}
            return text

        tc = tcs[0]
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except ValueError:
            args = {}
        try:
            raw = self._router._submit(self._router._async_call_tool(
                name, self._router._inject_ctx(name, dict(args), ctx or {}))).result(timeout=30)
        except Exception as e:
            raw = f"tool execution failed: {e}"

        # All four of our live FastMCP servers (time/weather/finance/websearch) return a single
        # spoken line (finance delegates the prefix to the consuming layer — normalized here by
        # the same rule as realtime._as_tool_result). The 2nd model call (formatting, ~1s) is
        # pure duplication, so it is skipped by default. API_BACKEND_FORMAT=model forces
        # always-model formatting (for external MCP servers outside this contract).
        raw_s = str(raw).strip()
        if os.environ.get("API_BACKEND_FORMAT") != "model":
            if not raw_s.startswith("(tool result)") and not raw_s.startswith("(no information"):
                raw_s = f"(tool result) {raw_s}"
            self.last_trace = {"model": MODEL, "tool": name, "args": args, "fmt": "tool",
                               "ms": round((time.perf_counter() - t0) * 1000)}
            return raw_s

        msgs += [{"role": "assistant", "content": None, "tool_calls": [tc]},
                 {"role": "tool", "tool_call_id": tc.get("id", "call_0"),
                  "content": raw_s}]
        m2 = self._call(msgs)
        text = (m2.get("content") or "").strip()
        if text and not text.startswith("(tool result)") and not text.startswith("(no information"):
            text = f"(tool result) {text}"
        self.last_trace = {"model": MODEL, "tool": name, "args": args,
                           "ms": round((time.perf_counter() - t0) * 1000)}
        return text or "(no information found)"


if __name__ == "__main__":
    b = ApiBackend()
    ctx = {"city": "Seoul", "timezone": "Asia/Seoul"}
    hist: list[str] = []
    for q in ["what time is it now?",
              "what time is it now? and then, how's the weather like today?",
              "음 어 다달음 일에 인천에서 도쿄 는가 도, 전오 출로 있?"]:
        out = b.retrieve(q, ctx=ctx, history=hist)
        print(f"\nQ: {q}\n→ {b.last_trace}\n  {out!r}")
        if b.last_trace.get("tool"):
            hist.append(f"{q!r} -> {b.last_trace['tool']}")
