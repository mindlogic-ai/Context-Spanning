# -*- coding: utf-8 -*-
"""Context DB: the per-session state the tool router reads next to the question.

Two layers:
  * ContextProfile: who the user is and where they are (name, city, timezone, lat/lon, language,
    free-form notes) plus the agent's persona. A fresh session already carries it, so location-aware
    tools work from the first question.
  * ContextDB: the append-only log of the session's user turns and tool results, with the salient facts
    the user stated about themselves ("my name is ...", "I'm in ...").

`working_text()` is the compact view handed to the router: one profile line and the most recent turns.
A tool turn carries the call and its result, so a later question can refer to a value that the ASR window
no longer contains.
"""

from __future__ import annotations

import re
import time
from typing import Optional


# The agent's own identity. It lives in the Context DB, not in the text prompt, so the model answers
# "who are you / what do you do" by grounding on it through <ret>.
AGENT_PERSONA = ("a professional AI development engineer and researcher, "
                 "and an ordinary office worker")


class ContextProfile:
    """Who the user is and where they are, plus the agent's own persona."""

    __slots__ = ("name", "city", "timezone", "lat", "lon", "language", "notes", "persona")

    def __init__(self, name="", city="", timezone="", lat=None, lon=None, language="", notes="",
                 persona=""):
        self.name = (name or "").strip()
        self.city = (city or "").strip()
        self.timezone = (timezone or "").strip()
        self.lat = _as_float(lat)
        self.lon = _as_float(lon)
        self.language = (language or "").strip()
        self.notes = (notes or "").strip()
        self.persona = (persona or "").strip() or AGENT_PERSONA

    def place_label(self) -> str:
        """Human label for the user's place: city, else timezone leaf ('Asia/Seoul' -> 'Seoul')."""
        if self.city:
            return self.city
        if self.timezone and "/" in self.timezone:
            return self.timezone.rsplit("/", 1)[-1].replace("_", " ")
        return self.timezone or ""

    def __repr__(self):
        return f"ContextProfile(name={self.name!r}, city={self.city!r}, tz={self.timezone!r})"


def _as_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


class Turn:
    __slots__ = ("idx", "role", "text", "t", "frame", "meta")

    def __init__(self, idx, role, text, t, frame=None, meta=None):
        self.idx = idx
        self.role = role                 # 'user' | 'tool'
        self.text = text
        self.t = t                       # epoch seconds
        self.frame = frame               # model frame index when logged (or None)
        self.meta = meta or {}


class ContextDB:
    """Per-session context store: profile + append-only conversation log + derived facts."""

    def __init__(self, profile: Optional[ContextProfile] = None):
        self.profile = profile or ContextProfile()
        self.turns: list[Turn] = []
        self.facts: dict[str, str] = {}          # salient extracted facts (name, location)
        # seed facts from the profile so grounding works from frame 0
        if self.profile.name:
            self.facts["name"] = self.profile.name
        if self.profile.place_label():
            self.facts["location"] = self.profile.place_label()

    def _add(self, role, text, frame=None, **meta) -> Optional[Turn]:
        text = (text or "").strip()
        if not text:
            return None
        turn = Turn(len(self.turns), role, text, time.time(), frame, meta)
        self.turns.append(turn)
        return turn

    def add_user_turn(self, text, frame=None) -> Optional[Turn]:
        """A user utterance (from ASR); salient facts are extracted from it."""
        t = self._add("user", text, frame)
        if t:
            self._extract_facts(t.text)
        return t

    def add_tool_turn(self, text, frame=None, **meta) -> Optional[Turn]:
        """A tool result that was injected as grounding."""
        return self._add("tool", text, frame, **meta)

    _NAME_PAT = re.compile(r"\b(?:my name is|i am|i'm|call me|this is)\s+([A-Z][a-zA-Z'-]+)", re.I)
    _LOC_PAT = re.compile(r"\b(?:i(?:'m| am)\s+(?:in|at|from)|i live in|located in)\s+"
                          r"([A-Z][a-zA-Z .'-]+)", re.I)

    def _extract_facts(self, text: str):
        m = self._NAME_PAT.search(text)
        if m and "name" not in self.facts:
            self.facts["name"] = m.group(1).strip(" .,")
        m = self._LOC_PAT.search(text)
        if m:
            self.facts["location"] = m.group(1).strip(" .,")

    def last_user_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user":
                return t.text
        return ""

    def working_text(self, recent: int = 10, max_chars: int = 1400) -> str:
        """Compact single-string view for the router's prompts: one profile+facts line, then the last
        `recent` turns oldest to newest ("user: ..." / "tool: ..."). Budget-clipped by dropping the
        OLDEST lines first; the full log stays in self.turns."""
        head = []
        bits = {}
        if self.profile.name:
            bits["user"] = self.profile.name
        if self.profile.place_label():
            bits["place"] = self.profile.place_label()
        for k, v in self.facts.items():
            bits.setdefault(k, v)
        if bits:
            head.append("profile: " + ", ".join(f"{k}={v}" for k, v in bits.items()))
        lines = head + [f"{t.role}: {t.text}" for t in self.turns[-recent:]]
        while lines and sum(len(l) + 1 for l in lines) > max_chars:
            lines.pop(1 if head and len(lines) > 1 else 0)   # drop oldest turn, keep profile
        return "\n".join(lines)
