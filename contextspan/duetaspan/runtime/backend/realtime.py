# -*- coding: utf-8 -*-
"""The retrieval backend of the runtime: one question in, one spoken reference line out.

The tool router (`runtime/mcp/client.py`) is the only thing that decides what a question needs: a tool
call, a direct answer, or nothing. Every result is normalized to the one-line contract the speech model
was trained on:

  * tool result    -> "(tool result) <spoken clause>"
  * direct answer  -> "<spoken factual sentence>"
  * nothing found  -> "(no information found)"
"""

import os
import re

import requests

NO_INFO = "(no information found)"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# The router sometimes "answers" with an apology instead of the canonical NO_INFO. Spoken as a span it is
# useless, so it is normalized to an abstain.
_ABSTAIN_RE = re.compile(r"(?:could\s*n[o']t\s+find|cannot\s+find|no\s+information|"
                         r"i\s+do\s*n[o']t\s+have\s+(?:any\s+)?information|unable\s+to\s+find)", re.I)


class RealtimeBackend:
    def __init__(self):
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = _UA
        self.last_source = None    # which source produced the last answer
        self.last_args = None      # arguments of the last tool call (None when no tool ran)
        self.last_trace = []       # [{src, hit, detail}] sources attempted on the last retrieve
        # The LLM that rewrites agent-addressed questions (see `_resolve_agent_subject`).
        self.llm_url = os.environ.get("MOSHICP_RAG_LLM_URL", "http://localhost:8004/v1/chat/completions")
        self.llm_model = os.environ.get("MOSHICP_RAG_LLM_MODEL", "google/gemma-4-26B-A4B-it")
        self.llm_timeout = float(os.environ.get("MOSHICP_RAG_LLM_TIMEOUT", "6.0"))
        # A hosted OpenAI-compatible endpoint needs a bearer key; a local server ignores the header.
        key = os.environ.get("MOSHICP_RAG_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            self.sess.headers["Authorization"] = "Bearer " + key

    def _trace(self, src, hit, detail=""):
        self.last_trace.append({"src": src, "hit": bool(hit), "detail": str(detail)[:120]})
        if hit:
            self.last_source = src

    def _llm_chat_raw(self, system, user, max_tokens=60):
        """One bare chat call on the LLM endpoint. Returns the content string ('' when empty)."""
        payload = {"model": self.llm_model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": 0, "max_tokens": max_tokens, "keep_alive": -1}
        r = self.sess.post(self.llm_url, json=payload, timeout=self.llm_timeout)
        return ((r.json().get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()

    def _resolve_agent_subject(self, q, persona):
        """Rewrite an agent-addressed question ('your latest song') into third person with the
        persona's explicit subject ('the latest song by A2O May'), so the tool router operates on a
        resolvable query. Routing the raw question breaks character ("I am an AI"); bypassing the
        router keeps the real tools from a real subject. Falls back to the original q untouched."""
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

    def retrieve(self, query, ctx=None, aux_context=None, history=None, convo=None):
        """Route ONE ASR transcript through the tool router and return the reference line.

        The CLEAN transcript drives the tool picker: a context prefix leaks stale city or subject names
        that mis-route tools. `ctx` (the user's profile) fills timezone and city defaults, `aux_context`
        is the agent's inner monologue, `history` the references already given, and `convo` the Context
        DB working text, which lets a question refer to earlier results outside the ASR window."""
        q = (query or "").strip()
        self.last_source = None
        self.last_args = None
        self.last_trace = []
        if not q:
            return NO_INFO
        persona = (ctx or {}).get("persona") if isinstance(ctx, dict) else None
        if persona and re.search(r"\b(your|yours)\b", q.lower()):
            q = self._resolve_agent_subject(q, persona)
        try:
            from contextspan.duetaspan.runtime.mcp.client import mcp_route
            m = mcp_route(q, ctx, history, aux=aux_context, convo=convo)
            if m and m.get("answer"):
                ans = m["answer"]
                if _ABSTAIN_RE.search(ans):
                    self._trace("llm-direct(abstain-ish)", False, ans[:80])
                else:
                    self._trace("llm-direct(1call)", ans != NO_INFO, ans)
                    return ans
            if m and m.get("tool"):
                self._trace(f"mcp:{m['tool']}", True, m.get("reference", ""))
                self.last_args = m.get("args")
                # Every tool-produced span carries the "(tool result)" prefix, as in the training data.
                return self._as_tool_result(m["reference"])
        except Exception as e:
            self._trace("mcp(unavailable)", False, str(e))
        # The router is the single decider: when it picks no tool and gives no answer, nothing is
        # injected. A web fallback here would override correct no-tool decisions with irrelevant spans.
        self._trace("router-final(no-tool)", False, q[:60])
        return NO_INFO

    @staticmethod
    def _as_tool_result(text):
        t = text.strip()
        return t if t.startswith("(tool result)") else f"(tool result) {t}"
