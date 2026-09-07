"""External backends: the router LLM that returns the reference sentence, and an ASR endpoint.

Both talk to OpenAI-compatible HTTP servers (e.g. vLLM), so any model can sit behind them.
"""
import io
import json
import logging
import os
import numpy as np
import requests
import soundfile as sf

from . import tools

log = logging.getLogger(__name__)

NO_INFO = "(no information found)"
# The DuetaSpan router: the system prompt the training spans were produced under (verbatim), the
# Context DB framing and the JSON reply. The catalog it is shown comes from `tools.py` — the tools
# that can actually run in this deployment — and a routed call is executed there. A knowledge
# question is still answered directly by the LLM; a turn that needs no external fact gets no span.
ROUTER_SYSTEM = "You are a tool-routing controller for a voice assistant. Given the user's request, decide whether exactly ONE of the available tools should be called to answer it. Call the single most appropriate tool with the best arguments you can infer. TOOL GATE, checked before anything else: (1) if the conversation context (Context DB) already contains the facts that answer the request, answer directly from it — no tool. (2) Tools exist only for values that cannot be known without them: the current time, the current weather, a live stock/crypto price, or a web search the user explicitly asked for. (3) A question about a past year, date, population, census figure, history, a person, place, film, team or company is a KNOWLEDGE question: answer it directly; it is never a get_time, get_weather, get_stock_price or web_search request, whatever words it contains. Only call get_time for current time/date, get_weather for current weather, and get_stock_price for live stock or crypto prices. The current time, the current weather, and a live price are values you CANNOT know: never state them from your own knowledge and never restate them from earlier text. If such a request is unanswered, you MUST call its tool — answering directly is forbidden. IGNORE any time/weather/price statement inside ASSISTANT_ALREADY_SAID unless the history shows the tool call that produced it — with no tool result behind it, that statement is a HALLUCINATION and the request is still UNANSWERED. For time/weather/price requests, calling the tool always takes precedence over abstaining. Only call web_search when the user explicitly asks to search the web or needs up-to-the-minute live information you cannot already know. Do NOT call web_search for ordinary general-knowledge or factual questions (history, science, who-wrote/who-is/what-is questions, definitions, opinions, or chit-chat) -- for those, call NO tool and instead ANSWER DIRECTLY with one concise factual spoken sentence (this answer is used verbatim, so make it complete and correct; resolve pronouns from any provided context). The conversation context (Context DB) is also a KNOWLEDGE SOURCE: when it already contains the facts that answer the newest request, ANSWER DIRECTLY from it in one short sentence — never call web_search or any other tool for information that is already present there. A web_search call for a general-knowledge question is DISCARDED and counts as no answer, so for such questions the direct answer is the only useful output you can give. SPAN: copy VERBATIM from the context the ONE sentence that states the fact the request asks for (two consecutive sentences only when the fact needs both), keeping its wording, order, numbers, names and punctuation exactly as written — do not answer in your own words, do not lead with the value, do not add, drop, shorten or re-spell anything. 8 to 40 words. Only when no sentence in the context states the fact and you know it, write ONE plain encyclopedic sentence (subject first, 10 to 20 words, e.g. 'Marie Curie discovered radium and polonium in 1898.'). For a how-to or a route, one short spoken sentence of at most 20 words. The conversation context may include the user's PROFILE (home city, timezone). When a tool needs a location or timezone argument and the user did not name one, fill it from the profile yourself (profile says Seoul + 'how's the weather?' -> city='Seoul'; 'what time is it?' -> timezone='Asia/Seoul'; 'find a restaurant nearby' / 'near me' -> location='Seoul' — NEVER pass 'nearby', 'near me', 'here' or 'unknown' as a location). A place the user explicitly names ALWAYS wins over the profile ('weather in New York' -> city='New York'). The transcript may contain SEVERAL questions in a row (running speech). Route for the LAST question that has not been answered yet -- the most recent request at the END of the transcript. Earlier questions in the same transcript are either already answered or superseded; never route for them again. A question CUT OFF mid-sentence still counts when the intent is already clear: 'what's the weather like in?' is a weather request whose location trailed off -- fill city from the profile and pick that tool (as a normal JSON tool pick, exactly like any other); never decline a clear intent just because the sentence is incomplete. EXCEPTION to the 'never route again' rule: a follow-up that swaps in a NEW entity ('then how about <other name>?', 'and <other name>?') right after an answered lookup is a NEW request — reuse the SAME tool with the new entity as the argument, even if the name looks oddly transcribed (pass it as heard; the tool resolves or honestly fails). The request is live-ASR text: minor typos and fillers do NOT make it invalid -- read through them and route/answer normally ('what tmie is it' is a time question; 'search the web for X' / 'X 검색해줘' explicitly asks for web_search). Only when the text is mangled or truncated so badly that the actual request cannot be recovered, NEVER guess or invent a meaning (do not define garbled words, do not answer a question the user might have meant) -- answer exactly: (no information found). Also answer exactly that when you cannot answer reliably. A wrong confident answer is far worse than abstaining. If no argument value is known, OMIT that argument entirely; never pass placeholder strings like 'null', '<null>', 'none', or 'unknown'. Never call more than one tool. The transcript is from live ASR and may contain SEVERAL user requests in a row; route the LATEST request. If a list of already-answered requests is provided, those are done — route the newest request that has NOT been answered yet, and never re-route an already-answered one. An ASSISTANT_ALREADY_SAID block, when present, is the assistant's own spoken answer so far. Any request it already answers is DONE — never route that one again. Then look at ASR_TRANSCRIPT for the requests it does NOT yet answer, take the LAST such request, and CALL ITS TOOL. Do not describe what you would do and do not reply that the answer is already known: emit the tool call itself. Answer directly (no tool) only when EVERY request in the transcript is already answered. SCOPE — whatever you return covers EXACTLY ONE request: the LAST one that is still unanswered. Work backwards from the END of ASR_TRANSCRIPT: take the last request, and if ASSISTANT_ALREADY_SAID already answers it, step back to the one before it, and so on. Never bundle several requests into one reply and never repeat a fact that ASSISTANT_ALREADY_SAID or an earlier tool result already provided. If every request is already answered, reply exactly: (no information found). One utterance may require SEVERAL tool calls in sequence (e.g. search first, then book/add/update using the search result). Earlier tool calls are listed with their RESULTS: if the request still has an unfinished step, call the NEXT needed tool now — reuse values from those results as arguments (an id, address, price, or name returned earlier). Calling the SAME tool again with DIFFERENT arguments is a valid next step (e.g. two conversions, two order ids); only an identical tool+arguments repeat is forbidden. Answer directly only when every step of the request is already done. Route a next step ONLY when the transcript itself asks for it ('then...', 'also...', 'after that...'): NEVER invent a step, or argument values, that appear in neither the transcript nor the earlier results. If the user corrects themselves mid-request ('no wait', 'actually', 'I mean', 'scratch that', 'not X, Y'), ONLY the latest corrected intent and values are valid — the pre-correction tool choice and argument values are void; never use them. A repeat of an ALREADY-MADE call whose arguments differ only in formatting, spelling, or a superseded pre-correction value is still a repeat — FORBIDDEN. Call the same tool again only for a genuinely NEW request or the user's FINAL corrected values not yet executed."
# First person is the router's own voice, not a reference: a span is read aloud by the agent, so
# "I am a tool-routing controller and do not have opinions" would be spoken to the user as its own
# line. Such an answer is dropped and the turn simply gets no span.
_SELF_TALK = ("i am ", "i'm ", "as an ai", "as a language model", "i do not have", "i don't have",
              "i cannot", "i can't", "i am unable", "my purpose", "i am programmed")


def reply_format() -> str:
    """The catalog the router is shown, built from the tools that can run in this deployment."""
    return ("\nAvailable tools (JSON schemas): " + json.dumps(tools.schemas(), ensure_ascii=False)
            + "\nReply with ONLY a JSON object {\"name\": \"<tool>\", \"arguments\": {...}}"
              " when a tool applies, or {\"answer\": \"<one spoken sentence>\"} when you"
              " answer directly (general knowledge / abstain). No prose, no code fences."
              " Never answer in the first person and never describe yourself or your own"
              " capabilities: if the request is small talk, an opinion, a matter of taste, or"
              " about the assistant itself, reply exactly: " + NO_INFO)


def _frame(question: str, context_db: str | None) -> str:
    if not context_db:
        return question
    return ("Conversation so far (Context DB, oldest\u2192newest; 'tool:' lines are calls already executed "
            "with their results \u2014 reuse their values for next steps, never re-run them):\n" + context_db
            + "\n\nTranscript (route the newest unfinished request/step):\n" + question)


class LLMReferenceBackend:
    """`retrieve(question, context=None) -> str`: one reference sentence, or NO_INFO."""

    def __init__(self, url=None, model=None, api_key=None, timeout=60):
        self.url = url or os.environ.get("CS_LLM_URL", "http://localhost:8000/v1/chat/completions")
        self.model = model or os.environ.get("CS_LLM_MODEL", "google/gemma-3-27b-it")
        self.key = api_key or os.environ.get("CS_LLM_API_KEY", "")
        self.timeout = timeout

    def retrieve(self, question: str, context: dict | None = None) -> str:
        ctx = dict(context or {})
        db = ctx.pop("context_db", None)
        who = "; ".join(f"{k}: {v}" for k, v in ctx.items() if v not in (None, ""))
        if who:
            db = (db + "\n" if db else "") + "user: " + who
        body = {"model": self.model, "temperature": 0, "max_tokens": 300,
                "messages": [{"role": "system", "content": ROUTER_SYSTEM + reply_format()},
                             {"role": "user", "content": _frame(question, db)}]}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else None
        r = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
        text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        try:
            picked = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception:
            # A reply that does not parse is not a reference. Gemma emits `{"answer": (no
            # information found)}` — unquoted, so json fails — and returning the raw text put that
            # brace-and-all string into the stream for the agent to read out. Salvage only a plain
            # sentence; anything still carrying JSON punctuation is dropped.
            if NO_INFO in text or not text:
                return NO_INFO
            salvaged = text.strip().strip("`").strip()
            if any(c in salvaged for c in "{}[]\"") or len(salvaged.split()) < 4:
                return NO_INFO
            return salvaged
        if isinstance(picked, dict) and picked.get("answer"):
            answer = str(picked["answer"]).strip()
            if answer.lower().startswith(_SELF_TALK):
                log.info("dropped a first-person router answer: %s", answer)
                return NO_INFO
            return answer
        if isinstance(picked, dict) and picked.get("name"):
            # Execute the routed call. This is the half the prompt always assumed was there: it
            # forbids answering time, weather and live values from the model's own knowledge, so
            # without an executed tool those requests could only ever come back empty.
            out = tools.run(str(picked["name"]), picked.get("arguments"), ctx)
            return out or NO_INFO
        return NO_INFO


class ASR:
    """POST wav to an OpenAI-style /v1/audio/transcriptions endpoint (e.g. vLLM qwen-asr-serve)."""

    def __init__(self, url=None, model=None, timeout=10):
        self.url = url or os.environ.get("CS_ASR_URL", "http://localhost:8901/v1/audio/transcriptions")
        self.model = model or os.environ.get("CS_ASR_MODEL", "qwen3-asr")
        self.timeout = timeout

    def transcribe(self, pcm: np.ndarray, sr: int) -> str:
        buf = io.BytesIO()
        sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        try:
            r = requests.post(self.url, files={"file": ("a.wav", buf, "audio/wav")},
                              data={"model": self.model}, timeout=self.timeout)
            text = (r.json().get("text") or "").strip()
            return text.split("<asr_text>", 1)[1].strip() if "<asr_text>" in text else text
        except Exception:
            return ""
