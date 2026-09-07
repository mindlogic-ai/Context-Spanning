# -*- coding: utf-8 -*-
"""Backend Context DB for MoshiCP — per-session user context + full conversation log.

NOT a training component (yet) — a serving-side store the retrieval/MCP layer consults so
answers are grounded in *who the user is* and *where the conversation happens*.

Two layers:
  * ContextProfile — identity + place: name, city, timezone, lat/lon, language, free-form notes.
    This is the "initialized" state of a fresh session: even with no history it already carries
    name/location so the MCP layer is immediately useful (location-aware time/weather, "what's my
    name"). Seeded from the client's ctx_* connect params (explicit fields default-filled by the
    browser's geolocation/Intl timezone).
  * ConversationLog — APPEND-ONLY, lossless record of EVERY turn (user / assistant / tool / system),
    each timestamped + frame-stamped. Nothing is ever dropped. This is the hard requirement:
    the whole conversation must always remain.

Context-management practice baked in (how a good assistant keeps context):
  * the full transcript is the source of truth and is never truncated;
  * a bounded *working set* (profile + salient facts + recent turns) is derived on demand for
    consumers that want compact context without losing the full log;
  * salient user facts are extracted incrementally from speech ("my name is …", "I'm in …");
  * deterministic JSON serialization → a session can be persisted per-user (default OFF) and
    later replayed as a training record (to_training_record()).

Scope default is per-session (in-memory; reset each connect) per the user's call, but the same
object persists/loads to disk so per-user continuity can be switched on without code changes.
"""

from __future__ import annotations

import json
import os
import re
import time
import datetime as _dt
from typing import Optional


# ── identity + place ─────────────────────────────────────────────────────────────────────
# The agent's own identity. It lives in the Context DB (not the text prompt) so the model answers
# "who are you / what do you do" by GROUNDING on it via <ret>, consistent with the retrieval design —
# the persona is never pre-baked into conditioning, and carries no Task/tool/location wording.
AGENT_PERSONA = ("Seonghyeon Go, a professional AI development engineer and researcher, "
                 "and an ordinary office worker")


class ContextProfile:
    """Who the user is and where they are, plus the agent's own persona. Seeds a session's grounding
    context. `persona` = the agent's self-identity (grounded, not prompted); `name`/`city` = the
    listener + environment."""

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

    @classmethod
    def from_params(cls, q) -> "ContextProfile":
        """Build from a dict-like of ctx_* connect params (aiohttp request.query, dict, …)."""
        g = q.get
        return cls(
            name=g("ctx_name", ""), city=g("ctx_city", ""), timezone=g("ctx_tz", ""),
            lat=g("ctx_lat", None), lon=g("ctx_lon", None),
            language=g("ctx_lang", ""), notes=g("ctx_notes", ""), persona=g("ctx_persona", ""),
        )

    def has_location(self) -> bool:
        return bool(self.city or self.timezone or (self.lat is not None and self.lon is not None))

    def local_now(self) -> _dt.datetime:
        """Current time in the user's timezone (server-local fallback)."""
        if self.timezone:
            try:
                from zoneinfo import ZoneInfo
                return _dt.datetime.now(ZoneInfo(self.timezone))
            except Exception:
                pass
        return _dt.datetime.now()

    def place_label(self) -> str:
        """Human label for the user's place: city, else timezone leaf ('Asia/Seoul' → 'Seoul')."""
        if self.city:
            return self.city
        if self.timezone and "/" in self.timezone:
            return self.timezone.rsplit("/", 1)[-1].replace("_", " ")
        return self.timezone or ""

    def as_dict(self) -> dict:
        return {
            "name": self.name, "city": self.city, "timezone": self.timezone,
            "lat": self.lat, "lon": self.lon, "language": self.language, "notes": self.notes,
            "persona": self.persona,
        }

    def __repr__(self):
        return f"ContextProfile(name={self.name!r}, city={self.city!r}, tz={self.timezone!r})"


def _as_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


# ── conversation turns ───────────────────────────────────────────────────────────────────
class Turn:
    __slots__ = ("idx", "role", "text", "t", "frame", "meta")

    def __init__(self, idx, role, text, t, frame=None, meta=None):
        self.idx = idx
        self.role = role                 # 'user' | 'assistant' | 'tool' | 'system'
        self.text = text
        self.t = t                       # epoch seconds
        self.frame = frame               # model frame index when logged (or None)
        self.meta = meta or {}

    def as_dict(self) -> dict:
        return {"idx": self.idx, "role": self.role, "text": self.text,
                "t": round(self.t, 3), "frame": self.frame, "meta": self.meta}

    def stamp(self) -> str:
        return _dt.datetime.fromtimestamp(self.t).strftime("%H:%M:%S")


class ContextDB:
    """Per-session context store: profile + full append-only conversation + derived facts."""

    def __init__(self, profile: Optional[ContextProfile] = None, session_id: str = ""):
        self.profile = profile or ContextProfile()
        self.session_id = session_id
        self.created_at = time.time()
        self.turns: list[Turn] = []
        self.facts: dict[str, str] = {}          # salient extracted facts (name, location, …)
        self._asst_buf: list[str] = []           # streaming assistant pieces awaiting a flush
        # seed facts from the profile so grounding works from frame 0
        if self.profile.name:
            self.facts["name"] = self.profile.name
        if self.profile.place_label():
            self.facts["location"] = self.profile.place_label()

    # ── append-only logging (the whole conversation, always) ─────────────────────────────
    def _add(self, role, text, frame=None, **meta) -> Optional[Turn]:
        text = (text or "").strip()
        if not text:
            return None
        turn = Turn(len(self.turns), role, text, time.time(), frame, meta)
        self.turns.append(turn)
        return turn

    def add_user_turn(self, text, frame=None) -> Optional[Turn]:
        """A user utterance (from ASR). Flushes any pending assistant text first so turn order
        is preserved, then extracts salient facts."""
        self.flush_assistant(frame)
        t = self._add("user", text, frame)
        if t:
            self._extract_facts(t.text)
        return t

    def add_tool_turn(self, text, frame=None, **meta) -> Optional[Turn]:
        """A retrieval / MCP tool result that was injected as grounding."""
        return self._add("tool", text, frame, **meta)

    def add_system_turn(self, text, frame=None, **meta) -> Optional[Turn]:
        return self._add("system", text, frame, **meta)

    def feed_assistant_piece(self, piece: str):
        """Accumulate streamed model text; materialized into one turn on flush."""
        if piece:
            self._asst_buf.append(piece)

    def flush_assistant(self, frame=None) -> Optional[Turn]:
        """Materialize buffered assistant text into a single turn (turn boundary / session end)."""
        if not self._asst_buf:
            return None
        text = "".join(self._asst_buf).strip()
        self._asst_buf.clear()
        return self._add("assistant", text, frame)

    # ── incremental fact extraction (lightweight, deterministic) ─────────────────────────
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

    # ── derived views ────────────────────────────────────────────────────────────────────
    def transcript(self, max_turns: Optional[int] = None) -> str:
        turns = self.turns if max_turns is None else self.turns[-max_turns:]
        return "\n".join(f"[{t.stamp()}] {t.role}: {t.text}" for t in turns)

    def last_user_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user":
                return t.text
        return ""

    def working_set(self, recent: int = 12) -> dict:
        """Bounded context for consumers: profile + facts + recent turns + totals. The FULL log
        always remains in self.turns; this is only a compact lens over it."""
        return {
            "profile": self.profile.as_dict(),
            "facts": dict(self.facts),
            "recent_turns": [t.as_dict() for t in self.turns[-recent:]],
            "n_turns_total": len(self.turns),
            "session_id": self.session_id,
        }

    def working_text(self, recent: int = 10, max_chars: int = 1400) -> str:
        """Compact single-string lens over the working set for LLM prompts (tool router /
        argument filler): one profile+facts line, then the last `recent` turns oldest→newest
        ("user:/assistant:/tool: …" — tool turns carry the call AND its result, so a serial
        chain can reference earlier results even when the ASR window no longer contains them).
        Budget-clipped by dropping OLDEST lines first; the full log stays in self.turns."""
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

    def snapshot(self) -> dict:
        """Complete, lossless state (full transcript included)."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "profile": self.profile.as_dict(),
            "facts": dict(self.facts),
            "turns": [t.as_dict() for t in self.turns],
        }

    # ── personal grounding (answered from context, no web) ───────────────────────────────
    def answer_personal(self, query: str) -> Optional[str]:
        """If the query asks about the user themselves, answer from the profile/facts.
        Returns a plain factual sentence (the model will speak it, grounded), or None to defer
        to web/tool retrieval. Time/weather are handled by the realtime tool layer with the
        profile's timezone/location, not here."""
        q = (query or "").lower()
        toks = set(re.findall(r"[a-z']+", q))
        name = self.facts.get("name") or self.profile.name
        place = self.facts.get("location") or self.profile.place_label()

        if ("name" in toks and ("my" in toks or "whats" in q or "what's" in q or "what" in toks)) \
                or ("who" in toks and "am" in toks and "i" in toks):
            return f"Your name is {name}." if name else None
        if (("where" in toks and ("am" in toks or "i" in toks)) or "location" in toks
                or ("which" in toks and "city" in toks)):
            return f"You are in {place}." if place else None
        if "timezone" in toks or ("time" in toks and "zone" in toks):
            return f"Your timezone is {self.profile.timezone}." if self.profile.timezone else None
        return None

    # ── persistence (default OFF; enables per-user continuity + training reuse later) ────
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContextDB":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        db = cls(ContextProfile(**d.get("profile", {})), d.get("session_id", ""))
        db.created_at = d.get("created_at", db.created_at)
        db.facts = d.get("facts", {}) or {}
        for td in d.get("turns", []):
            db.turns.append(Turn(td["idx"], td["role"], td["text"], td.get("t", 0.0),
                                 td.get("frame"), td.get("meta")))
        return db

    def to_training_record(self) -> dict:
        """Future training reuse: one structured dialogue record (profile + ordered turns)."""
        return {
            "profile": self.profile.as_dict(),
            "turns": [{"role": t.role, "text": t.text} for t in self.turns],
        }

    def __len__(self):
        return len(self.turns)
