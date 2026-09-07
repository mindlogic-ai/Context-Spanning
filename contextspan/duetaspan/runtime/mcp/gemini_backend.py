# -*- coding: utf-8 -*-
"""Single-model MCP backend on Gemini API — routing + result phrasing in ONE model.

Replaces the two-stage local pipeline (gemma router prompt → per-tool spoken-line
templates) with Gemini native function calling over the SAME MCP tools:

  1. MCP tool schemas are PRE-REGISTERED as `tools.functionDeclarations` (from the
     live FastMCP servers via MCPRouter, so tool defs stay single-sourced).
  2. One `generateContent` call: transcript + ctx + answered-history → functionCall.
  3. We execute the tool on our MCP servers (unchanged), send functionResponse back.
  4. The SAME model writes the final spoken line — it owns the "(tool result) ..."
     contract too (no per-tool format templates).

Key via GEMINI_API_KEY env (never hardcoded). Model via GEMINI_MODEL
(default gemini-2.5-flash, thinking disabled for latency).

Smoke: GEMINI_API_KEY=... python -m contextspan.duetaspan.runtime.mcp.gemini_backend
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import requests

_API = "https://generativelanguage.googleapis.com/v1beta/models"
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_KEY = os.environ.get("GEMINI_API_KEY", "")

_SYSTEM = (
    "You are the retrieval backend of a full-duplex voice assistant. Input is a raw live-ASR "
    "transcript: it may contain fillers, typos, self-corrections, and SEVERAL user requests in "
    "a row. If a list of already-answered requests is given, those are done — serve the newest "
    "request that has NOT been answered yet; never re-serve an answered one. "
    "If exactly one registered tool answers it, call that tool (fill arguments from the user's "
    "words; use the user profile for defaults like city/timezone; omit unknown arguments). "
    "If no tool applies and it is ordinary general knowledge, ANSWER IT DIRECTLY from your "
    "own knowledge — fillers and typos do not make a request invalid when the intent is clear. "
    "NEVER call web_search for general knowledge you already know; only call it when the user "
    "explicitly asks to search or the answer needs live web data no other tool covers. "
    "Only when the request itself is unintelligible or truncated beyond recovery, output "
    "exactly: (no information found). "
    "FINAL ANSWER FORMAT: one concise SPOKEN sentence, as said aloud by a voice assistant — "
    "no markdown, no lists. When the answer comes from a tool call, prefix it with "
    "'(tool result) '. A plain-knowledge answer gets no prefix."
)

# Gemini functionDeclarations accept an OpenAPI-subset schema; strip fields it rejects.
_SCHEMA_KEEP = {"type", "description", "properties", "required", "items", "enum"}


def _clean_schema(node: Any) -> Any:
    """Reduce a JSON schema to Gemini's OpenAPI subset. `properties` keys are ARGUMENT
    NAMES (not schema keywords) and must survive; FastMCP's anyOf-optionals collapse to
    their first non-null branch; `required` may only list surviving properties."""
    if not isinstance(node, dict):
        return node
    if "anyOf" in node:                                   # Optional[X] → X
        branch = next((b for b in node["anyOf"]
                       if isinstance(b, dict) and b.get("type") != "null"), {})
        node = {**branch, "description": node.get("description", branch.get("description", ""))}
    out = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: _clean_schema(sub) for name, sub in v.items()}
        elif k == "items":
            out[k] = _clean_schema(v)
        elif k in _SCHEMA_KEEP:
            out[k] = v
    if "required" in out and isinstance(out.get("properties"), dict):
        out["required"] = [r for r in out["required"] if r in out["properties"]]
        if not out["required"]:
            del out["required"]
    return out


class GeminiBackend:
    """One Gemini model doing route + execute-loop + spoken formatting over our MCP tools."""

    def __init__(self, timeout: float = 20.0):
        if not _KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from contextspan.duetaspan.runtime.mcp.client import get_mcp_router
        self._router = get_mcp_router()                       # MCP servers + dispatch (unchanged)
        self.timeout = timeout
        self.sess = requests.Session()
        self.last_trace: list[dict] = []
        decls = []
        for entry in self._router._tool_schemas():
            fn = entry.get("function", entry)
            decls.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": _clean_schema(fn.get("parameters") or {"type": "object"}),
            })
        self._tools = [{"functionDeclarations": decls}]       # ← 사전 등록되는 MCP 카탈로그

    # ── raw generateContent call ──────────────────────────────────────────────
    def _call(self, contents: list[dict]) -> dict:
        body = {
            "system_instruction": {"parts": [{"text": _SYSTEM}]},
            "contents": contents,
            "tools": self._tools,
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 120,
                "thinkingConfig": {"thinkingBudget": 0},      # latency: no hidden CoT
            },
        }
        r = self.sess.post(f"{_API}/{_MODEL}:generateContent?key={_KEY}",
                           json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parts(resp: dict) -> list[dict]:
        return (((resp.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []

    # ── public API (mirrors RealtimeBackend.retrieve) ────────────────────────
    def retrieve(self, query: str, ctx: Optional[dict] = None,
                 history: Optional[list[str]] = None, convo: Optional[str] = None) -> str:
        # convo = ContextDB.working_text() 스냅샷. api_backend.retrieve 와 같은 이유로
        # 필수 입력이다 — 라우팅 근거를 단일 ASR 윈도우가 아니라 대화 누적 상태에 둔다.
        self.last_trace = []
        user = ""
        if ctx:
            user += f"User profile: {json.dumps(ctx, ensure_ascii=False)}\n"
        if convo:
            user += ("Conversation state so far (Context DB — AUTHORITATIVE for resolving "
                     "entities, IDs and values already established; prefer these over a "
                     "partial transcript):\n" + convo + "\n")
        if history:
            user += ("Already answered (do NOT re-serve):\n"
                     + "\n".join(f"- {h}" for h in history[-5:]) + "\n")
        user += f"ASR transcript:\n{query}"
        contents = [{"role": "user", "parts": [{"text": user}]}]

        t0 = time.perf_counter()
        resp = self._call(contents)
        t_route = time.perf_counter() - t0

        parts = self._parts(resp)
        fc = next((p["functionCall"] for p in parts if "functionCall" in p), None)
        if fc is None:                                        # direct answer / abstain
            text = " ".join(p.get("text", "") for p in parts).strip() or "(no information found)"
            self.last_trace.append({"src": f"gemini:{_MODEL}", "tool": None,
                                    "t_route": t_route, "t_total": t_route})
            return text

        name, args = fc["name"], fc.get("args") or {}
        t1 = time.perf_counter()
        try:
            raw = self._router._submit(
                self._router._async_call_tool(name, self._router._inject_ctx(name, dict(args), ctx or {}))
            ).result(timeout=30)
        except Exception as e:
            raw = f"tool execution failed: {e}"
        t_exec = time.perf_counter() - t1

        # feed the tool result back — the SAME model phrases the spoken line.
        # Gemini 3.x rejects a rebuilt model turn: the original parts (incl. thoughtSignature)
        # and the functionCall id must be echoed back verbatim.
        fr: dict = {"name": name, "response": {"result": str(raw)}}
        if fc.get("id"):
            fr["id"] = fc["id"]
        contents += [
            {"role": "model", "parts": parts},
            {"role": "user", "parts": [{"functionResponse": fr}]},
        ]
        t2 = time.perf_counter()
        resp2 = self._call(contents)
        t_fmt = time.perf_counter() - t2
        text = " ".join(p.get("text", "") for p in self._parts(resp2)).strip()
        if text and not text.startswith("(tool result)") and not text.startswith("(no information"):
            text = f"(tool result) {text}"
        self.last_trace.append({"src": f"gemini:{_MODEL}", "tool": name, "args": args,
                                "t_route": t_route, "t_exec": t_exec, "t_fmt": t_fmt,
                                "t_total": time.perf_counter() - t0})
        return text or "(no information found)"


if __name__ == "__main__":
    b = GeminiBackend()
    ctx = {"city": "Seoul", "timezone": "Asia/Seoul"}
    hist: list[str] = []
    for q in [
        "what time is it now?",
        "what time is it now? and then, how's the weather like today?",       # 재포착 전사
        "how's the weather like today? and how price is the samsung electronics?",
        "uh um so like who wrote the old man and the sea",                      # no-tool 일반지식
        "음 어 다달음 일에 인천에서 도쿄 는가 도, 전오 출로 있?",                 # heavy noise → abstain 기대
    ]:
        out = b.retrieve(q, ctx=ctx, history=hist)
        tr = b.last_trace[-1]
        lat = " ".join(f"{k}={tr[k]*1000:.0f}ms" for k in ("t_route", "t_exec", "t_fmt", "t_total") if k in tr)
        print(f"\nQ: {q}\n→ tool={tr.get('tool')} {lat}\n  {out!r}")
        if tr.get("tool"):
            hist.append(f"{q!r} -> {tr['tool']}")
