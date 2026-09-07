# -*- coding: utf-8 -*-
"""MoshiRAG-style retrieval backend (RAG + MCP) for MoshiCP.

Returns ONE reference string in the same spoken form the training data uses.
For RAG that is a factual passage; for MCP it is a "(tool result) ..." clause.
The returned text is what gets injected into the model's Context Span at
inference time. Pure Python, no GPU, no model loading.
"""

import glob
import json
import os
import re
from contextspan.duetaspan.common import paths

DEFAULT_DATA_DIR = f"{paths.DATA}/moshicp"
NO_INFO = "(no information found)"

# Words that signal realtime / tool-call intent -> route to MCP.
MCP_INTENT_WORDS = {
    "weather", "temperature", "forecast", "rain", "raining", "sunny",
    "time", "clock", "timezone",
    "price", "stock", "cost", "trading", "exchange", "rate",
    "book", "booking", "reserve", "reservation", "table",
    "play", "pause", "song", "music", "track",
    "remind", "reminder", "alarm", "timer",
    "score", "scores", "game", "match", "live",
    "flight", "flights", "departure", "arrival", "gate",
    "news", "headlines", "breaking",
    "schedule", "appointment", "calendar",
    "buy", "order", "ride", "cab", "taxi", "uber",
    "now", "current", "currently", "today", "tonight", "right",
}

# Query keywords -> tool-bank domain, used to pick a tool when no spoken
# scenario exists and a "(tool result) ..." clause must be synthesized.
DOMAIN_KEYWORDS = {
    "Weather": ("weather", "temperature", "forecast", "rain", "sunny", "hot", "cold"),
    "Restaurants": ("table", "restaurant", "reservation", "reserve", "dinner", "lunch", "eat"),
    "Alarm": ("remind", "reminder", "alarm", "timer", "wake"),
    "Flights": ("flight", "fly", "departure", "arrival", "gate", "airport"),
    "Music": ("song", "music", "track", "play", "album", "artist"),
    "Movies": ("movie", "film", "cinema", "showtime"),
    "Hotels": ("hotel", "room", "stay", "checkin", "lodging"),
    "RideSharing": ("ride", "cab", "taxi", "uber", "lyft", "pickup"),
    "Events": ("event", "concert", "ticket", "show", "game", "match", "score"),
    "Payment": ("pay", "payment", "transfer", "send", "money"),
    "Media": ("news", "headlines", "breaking", "story"),
    "Trains": ("train", "rail"),
    "Buses": ("bus", "coach"),
}

_WORD_RE = re.compile(r"[a-z0-9]+")

_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve",
}


def _tokens(text):
    return _WORD_RE.findall((text or "").lower())


def _spell_numbers(text):
    """Spell out small standalone digits as words (best-effort, spoken form)."""
    def repl(m):
        return _NUMBER_WORDS.get(m.group(0), m.group(0))
    return re.sub(r"\b\d+\b", repl, text)


class _TokenOverlapRetriever:
    """Last-resort retriever: Jaccard-style token overlap. No dependencies."""

    def __init__(self, passages):
        self.passages = passages
        self.token_sets = [set(_tokens(p)) for p in passages]

    def top1(self, query):
        q = set(_tokens(query))
        if not q or not self.passages:
            return None
        best_i, best_score = -1, 0.0
        for i, ts in enumerate(self.token_sets):
            if not ts:
                continue
            inter = len(q & ts)
            if not inter:
                continue
            score = inter / float(len(q | ts))
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            return None
        return self.passages[best_i]


class _TfidfRetriever:
    """scikit-learn TF-IDF + cosine top-1."""

    def __init__(self, passages):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.passages = passages
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(passages)

    def top1(self, query):
        from sklearn.metrics.pairwise import linear_kernel

        if not self.passages:
            return None
        qv = self.vectorizer.transform([query])
        sims = linear_kernel(qv, self.matrix)[0]
        i = int(sims.argmax())
        if sims[i] <= 0.0:
            return None
        return self.passages[i]


class _SentenceTransformerRetriever:
    """sentence-transformers embedding + cosine top-1."""

    def __init__(self, passages):
        from sentence_transformers import SentenceTransformer

        self.passages = passages
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(
            passages, normalize_embeddings=True, convert_to_numpy=True
        )

    def top1(self, query):
        if not self.passages:
            return None
        qv = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        sims = self.embeddings @ qv
        i = int(sims.argmax())
        if float(sims[i]) <= 0.0:
            return None
        return self.passages[i]


def _build_retriever(passages):
    """Return (retriever, backend_name), preferring richer backends first."""
    if not passages:
        return None, "none"
    try:
        return _SentenceTransformerRetriever(passages), "sentence-transformers"
    except Exception:
        pass
    try:
        return _TfidfRetriever(passages), "tfidf"
    except Exception:
        pass
    return _TokenOverlapRetriever(passages), "token-overlap"


class Backend:
    """RAG + MCP retrieval backend over the MoshiCP corpus.

    Usage:
        backend = Backend()
        ref = backend.retrieve("what's the weather in Tokyo")
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.retriever_backend = "none"

        # RAG passages, and MCP scenarios (query -> "(tool result) ..." ref).
        rag_passages, mcp_queries, mcp_refs = self._load_corpus()

        # An optional pre-built scenario pool can override / extend mined MCP.
        pool_q, pool_refs = self._load_scenario_pool()
        mcp_queries = pool_q + mcp_queries
        mcp_refs = pool_refs + mcp_refs

        self.rag_passages = rag_passages
        self.mcp_refs = mcp_refs
        self.tool_bank = self._load_tool_bank()

        self.rag_retriever, self.retriever_backend = _build_retriever(rag_passages)
        # MCP scenario retriever indexes user queries -> reference at same offset.
        self.mcp_retriever, _ = _build_retriever(mcp_queries)
        # Tool-bank retriever for synthesis fallback (no scenarios available).
        self._tool_texts = [
            "%s %s" % (t.get("description", ""), t.get("domain", ""))
            for t in self.tool_bank
        ]
        self.tool_retriever, _ = _build_retriever(self._tool_texts)

    # ----- corpus loading -------------------------------------------------

    def _context_files(self):
        pattern = os.path.join(self.data_dir, "tensors", "train_*.context.jsonl")
        return sorted(glob.glob(pattern))

    def _load_corpus(self):
        """Mine RAG passages and MCP (query, reference) pairs from contexts."""
        rag_seen, rag_passages = set(), []
        mcp_queries, mcp_refs = [], []

        for path in self._context_files():
            try:
                handle = open(path, "r", encoding="utf-8")
            except OSError:
                continue
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        dialogue = json.loads(line)
                    except ValueError:
                        continue
                    self._ingest_dialogue(
                        dialogue, rag_seen, rag_passages,
                        mcp_queries, mcp_refs,
                    )
        return rag_passages, mcp_queries, mcp_refs

    def _ingest_dialogue(
        self, dialogue, rag_seen, rag_passages, mcp_queries, mcp_refs
    ):
        turns = dialogue.get("turns_meta") or []
        # turn_idx -> user text, for pairing a span with the query that prompted it.
        user_by_turn = {
            t.get("turn_idx"): t.get("text", "")
            for t in turns
            if t.get("speaker") == "user"
        }
        ordered_turns = sorted(
            (t for t in turns if t.get("turn_idx") is not None),
            key=lambda t: t.get("turn_idx"),
        )

        def nearest_user_query(turn_idx):
            if turn_idx is None:
                return None
            best = None
            for t in ordered_turns:
                idx = t.get("turn_idx")
                if idx is not None and idx <= turn_idx and t.get("speaker") == "user":
                    best = t.get("text", "")
            return best or next(iter(user_by_turn.values()), None)

        spans = list(dialogue.get("context_spans") or [])
        # turns_meta may also carry references in some schema versions.
        for t in turns:
            if isinstance(t, dict) and t.get("reference"):
                spans.append(t)

        for span in spans:
            reference = (span.get("reference") or "").strip()
            if not reference or reference == NO_INFO:
                continue
            is_mcp = (
                span.get("retrieval_type") == "mcp"
                or reference.startswith("(tool result)")
            )
            if is_mcp:
                query = nearest_user_query(span.get("turn_idx"))
                if query:
                    mcp_queries.append(query)
                    mcp_refs.append(reference)
            else:
                if reference not in rag_seen:
                    rag_seen.add(reference)
                    rag_passages.append(reference)

    def _load_scenario_pool(self):
        """Load an explicit scenario pool if present: returns (queries, refs)."""
        path = os.path.join(self.data_dir, "mcp_scenario_pool.json")
        if not os.path.exists(path):
            return [], []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                pool = json.load(handle)
        except (OSError, ValueError):
            return [], []
        queries, refs = [], []
        for item in pool if isinstance(pool, list) else []:
            ref = (item.get("reference") or "").strip()
            query = (item.get("user_query") or "").strip()
            if ref and query:
                queries.append(query)
                refs.append(ref)
        return queries, refs

    def _load_tool_bank(self):
        path = os.path.join(self.data_dir, "mcp_tool_bank.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                bank = json.load(handle)
        except (OSError, ValueError):
            return []
        return bank if isinstance(bank, list) else []

    # ----- routing & retrieval -------------------------------------------

    @staticmethod
    def _is_mcp_intent(query):
        return bool(set(_tokens(query)) & MCP_INTENT_WORDS)

    def retrieve(self, query, kind="auto"):
        """Return one reference string for ``query``.

        ``kind`` in {"auto", "rag", "mcp"}; "auto" routes realtime/tool
        queries to MCP and everything else to RAG.
        """
        if kind not in ("auto", "rag", "mcp"):
            kind = "auto"
        if kind == "auto":
            kind = "mcp" if self._is_mcp_intent(query) else "rag"

        if kind == "mcp":
            result = self._retrieve_mcp(query)
            if result:
                return result
            # Fall through to RAG if MCP has nothing useful.
            result = self._retrieve_rag(query)
            return result or NO_INFO

        result = self._retrieve_rag(query)
        return result or NO_INFO

    def _retrieve_rag(self, query):
        if self.rag_retriever is None:
            return None
        return self.rag_retriever.top1(query)

    def _retrieve_mcp(self, query):
        # 1) Closest mined/pooled scenario -> spoken "(tool result) ..." ref.
        if self.mcp_retriever is not None:
            idx = self._mcp_match_index(query)
            if idx is not None:
                return self.mcp_refs[idx]
        # 2) No scenarios: synthesize a clause from the best-matching tool.
        return self._synthesize_from_tool(query)

    def _mcp_match_index(self, query):
        """Return index into mcp_refs of the best-matching scenario query."""
        retriever = self.mcp_retriever
        if retriever is None or not retriever.passages:
            return None
        match = retriever.top1(query)
        if match is None:
            return None
        try:
            return retriever.passages.index(match)
        except ValueError:
            return None

    def _synthesize_from_tool(self, query):
        if not self.tool_bank:
            return None
        tool = self._best_tool(query)
        domain = (tool.get("domain") or "the requested service").lower()
        clause = "(tool result) here is the latest %s information for your request" % domain
        return _spell_numbers(clause)

    def _best_tool(self, query):
        """Pick the closest tool: domain-keyword map first, then TF-IDF text."""
        q = set(_tokens(query))
        best_domain, best_hits = None, 0
        for domain, keywords in DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in q)
            if hits > best_hits:
                best_domain, best_hits = domain, hits
        if best_domain is not None:
            for tool in self.tool_bank:
                if tool.get("domain") == best_domain:
                    return tool
        if self.tool_retriever is not None:
            match = self.tool_retriever.top1(query)
            if match is not None:
                return self.tool_bank[self._tool_texts.index(match)]
        return self.tool_bank[0]


def _smoke_test():
    backend = Backend()
    print("retriever backend: %s" % backend.retriever_backend)
    print("rag corpus size: %d passages" % len(backend.rag_passages))
    print("mcp scenarios: %d | tools: %d" % (len(backend.mcp_refs), len(backend.tool_bank)))
    print("-" * 60)
    samples = [
        "what's the weather in Tokyo",
        "who painted the mona lisa",
        "book a table for two",
        "what's the score in the game",
        "remind me to call mom",
        "tell me about boston history",
    ]
    for query in samples:
        print("Q: %s" % query)
        print("R: %s" % backend.retrieve(query))
        print()


if __name__ == "__main__":
    _smoke_test()
