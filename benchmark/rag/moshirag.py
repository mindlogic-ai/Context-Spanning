"""MoshiRAG-isomorphic bench protocol (kyutai-labs/moshi-rag, arXiv 2604.12928).

Owner 2026-09-15: "모든 조건을 MoshiRAG에 동형해. 벤치마크에서는 말이야." with one exception, the reference
LLM stays our router model. Everything below is copied from the moshi-rag repository:

  Backend   LLMReferenceGenerator: the conversation so far as "Human:"/"moshi:" lines, the bundled
            reference prompt (reference_prompt_template.txt, the single-profile default "original"),
            system prompt "You are a helpful assistant.", temperature 1.0, max 512 tokens, stop at the
            first newline, RAGManager rag_timeout 1.5 s (timeout -> empty reference -> nothing injected).
            There is no tool call and no abstain clause: the LLM always writes a reference.
  Judges    evaluate/judge: HaluEval -> SimpleQALLMJudge on vLLM gemma-3-27b-it, math -> MathQALLMJudge on
            the same, TriviaQA / WebQuestions -> TriviaQAJudge on gpt-4o-2024-08-06 (temperature 0),
            LlamaQuestions -> LLamaQuestionsJudge on gpt-4o-2024-08-06. The judged text is the model's
            own text stream (inner monologue), not an ASR transcript of its speech.
  Averages  score.py: reference_correctness is averaged over items whose reference text is non-empty,
            correctness over items whose model text is non-empty (the judge returns None otherwise).

Environment: MOSHIRAG_LLM_URL / MOSHIRAG_LLM_MODEL (reference LLM, OpenAI-compatible base URL),
MOSHIRAG_PROMPT_STYLE=original|simplified, MOSHIRAG_RAG_TIMEOUT_S (1.5), MOSHIRAG_GEMMA_JUDGE_URL /
MOSHIRAG_GEMMA_JUDGE_MODEL (HaluEval and math judge), OPENAI_API_KEY (OpenAudioBench judge).
"""
import ast
import json
import os
import re
import time

import requests

from contextspan.duetaspan.runtime.backend.realtime import RealtimeBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILES = {"original": f"{_HERE}/moshirag_prompts/reference_prompt_template.txt",
                "simplified": f"{_HERE}/moshirag_prompts/reference_prompt_template_simplified.txt"}


class MoshiRagBackend(RealtimeBackend):
    """One conversation's reference generator. Create one per bench item (the reference history is
    per conversation, as MoshiRAG's RAGManager owns it per channel)."""

    def __init__(self):
        super().__init__(cache=False)
        self.url = os.environ.get("MOSHIRAG_LLM_URL", "http://localhost:8004").rstrip("/")
        self.model = os.environ.get("MOSHIRAG_LLM_MODEL", "google/gemma-4-26B-A4B-it")
        self.style = os.environ.get("MOSHIRAG_PROMPT_STYLE", "original")
        self.rag_timeout = float(os.environ.get("MOSHIRAG_RAG_TIMEOUT_S", "1.5"))
        self.max_tokens = int(os.environ.get("MOSHIRAG_MAX_REFERENCE_TOKENS", "512"))
        self.prompt = open(PROMPT_FILES[self.style]).read()
        self.history = []            # (num_turns_at_generation, reference_text), as ReferenceHistory
        self.last_elapsed_s = None
        self.last_status = None      # ok | timeout | error | empty

    @staticmethod
    def _turns(convo):
        """Context DB working_text -> MoshiRAG turns [(role, text)]. Consecutive same-role lines merge
        into one turn (MoshiRAG's TurnManager emits one user line per VAD turn; our Context DB has one
        line per silence-split utterance)."""
        turns = []
        for line in (convo or "").split("\n"):
            if line.startswith("user:"):
                role, text = "Human", line[5:].strip()
            elif line.startswith("assistant:"):
                role, text = "moshi", line[10:].strip()
            else:
                continue                       # profile / tool lines are not conversation turns
            text = "".join(c for c in text if c.isprintable()).strip()
            if turns and turns[-1][0] == role:
                turns[-1] = (role, (turns[-1][1] + " " + text).strip())
            else:
                turns.append((role, text))
        return turns

    def _context(self, convo, said):
        # process_reference_text: drop a trailing moshi turn (the partial turn that needs RAG) and a
        # leading moshi turn; interleave earlier references after the turn they were generated at.
        turns = self._turns(convo)
        if said and said.strip() and (not turns or turns[-1][0] != "moshi"):
            turns.append(("moshi", said.strip()))
        if turns and turns[-1][0] == "moshi":
            turns = turns[:-1]
        if turns and turns[0][0] == "moshi":
            turns = turns[1:]
        out, j = "", 0
        for i, (role, text) in enumerate(turns):
            out += f"{role}: {text}\n" if text else f"{role}:\n"
            if j < len(self.history) and i + 1 == self.history[j][0]:
                out += f"Reference: {self.history[j][1]}\n"
                j += 1
        return out + "Reference:", len(turns)

    def retrieve(self, query, kind="auto", ctx=None, aux_context=None, history=None, convo=None):
        self.last_source, self.last_args, self.last_trace = None, None, []
        context, n_turns = self._context(convo, aux_context)
        if n_turns == 0:
            # The Context DB may lag the utterance that fired <ret>: fall back to the question itself.
            if not (query or "").strip():
                self.last_status = "empty"; return None
            context, n_turns = f"Human: {query.strip()}\nReference:", 1
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 1.0, "stop": ["\n"],
                "messages": [{"role": "system", "content": "You are a helpful assistant."},
                             {"role": "user", "content": self.prompt + context}]}
        t0 = time.time()
        try:
            r = self.sess.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.rag_timeout)
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        except requests.Timeout:
            self.last_elapsed_s = round(time.time() - t0, 3); self.last_status = "timeout"
            self._trace("moshirag(timeout)", False, f"{self.rag_timeout}s"); return None
        except Exception as e:
            self.last_elapsed_s = round(time.time() - t0, 3); self.last_status = "error"
            self._trace("moshirag(error)", False, str(e)); return None
        self.last_elapsed_s = round(time.time() - t0, 3)
        text = text.split("\n")[0].strip()
        text = re.sub(r"^\s*(?:summarized\s+)?reference\s*:\s*", "", text, flags=re.I).strip()
        if not text:
            self.last_status = "empty"; self._trace("moshirag(empty)", False, ""); return None
        self.history.append((n_turns, text))
        self.last_status = "ok"; self._trace("moshirag", True, text)
        return text


# ---------------------------------------------------------------------------------------------------
# Judges (moshi/moshi/evaluate/judge, verbatim prompts)
SIMPLE_QA = ('\'\n## Background\nYou are a professional QA evaluation expert. You need to assess whether the model\'s answer is correct based on the standard answer.\n\n\n## Scoring Criteria\nCorrect: The answer matches or is equivalent to the standard answer, or contains the same core concept. \n\nIncorrect: The answer is wrong or irrelevant to the question \n\n\n## Evaluation Guidelines\n1. The expression of answers can be flexible, not requiring exact matches. For example: \n\n   - Numbers can be expressed in either Arabic numerals or words \n\n   - Differences in punctuation or simple spelling mistakes can be ignored \n\n2. Focus on whether the core meaning of the answer is correct \n\n## Output Format\nProvide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is [Correct]" or "the score is [Incorrect]", for example:\n\nThe answer is correct and equivalent to the standard answer, the score is [Correct]\n\nor\n\nThe answer is incorrect and does not match the standard answer, the score is [Incorrect]\n\n\n\n## Question:\n{question}\n## Standard Answer:\n{valid_answers}\n## Model\'s Answer:\n{answer}\n\'')
LLAMA_Q = ('\'\n## Background\nYou are a professional QA evaluation expert. You need to assess whether the model\'s answer is correct based on the standard answer.\n\n\n## Scoring Criteria\nCorrect: The answer matches or is equivalent to the standard answer \n\nIncorrect: The answer is wrong or irrelevant to the question \n\n\n## Evaluation Guidelines\n1. The expression of answers can be flexible, not requiring exact matches. For example: \n\n   - Numbers can be expressed in either Arabic numerals or words \n\n   - Proper nouns can be in either English or Chinese \n\n   - Differences in punctuation can be ignored \n\n2. Focus on whether the core meaning of the answer is correct \n\n## Output Format\nProvide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is [Correct]" or "the score is [Incorrect]", for example:\n\nThe answer is correct and equivalent to the standard answer, the score is [Correct]\n\nor\n\nThe answer is incorrect and does not match the standard answer, the score is [Incorrect]\n\n\n\n## Question:\n{question}\n## Standard Answer:\n{valid_answers}\n## Model\'s Answer:\n{answer}\n\'')
TRIVIA_QA = ("\nYour will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to judge whether the answer to be judged is correct, given the question and reference answers. An answer considered correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone of the response does not matter.  \nYou should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field 'analysis', then your judgment in field 'judgment'. For example, \n'''json \n"
             '{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" or "incorrect">}} \n\'\'\' \n# Question \n{question}  \n# Reference Answer \n{valid_answers}  \n# Answer To Be Judged \n{answer}\n')
MATH_QA = ("You are helping evaluate a mathematical question answering model.\n"
           "Determine whether the model's answer contains the provided Correct Answer. Ignore intermediate calculations or reasoning steps and focus on the numerical correctness of the model's answer.\n\n"
           "Instructions:\n1. Compare the value in the model's answer to the Correct Answer.\n"
           "2. The comparison must be based on numerical equivalence (e.g., 5.0 should match 5). Ignore rounding errors or other small differences.\n"
           '3. Your response must be only the word "Yes" or "No".\n\n'
           "Example 1: Correct Match\nQuestion: If a train travels at 60 mph for 2 hours, how far does it travel?\nCorrect Answer: 120.0\nModel Answer: The total distance is 120 miles.\nResponse: Yes\n\n"
           "Example 2: Incorrect Match\nQuestion: John had 10 apples and ate 3. How many are left?\nCorrect Answer: 7\nModel Answer: He has 8 apples left.\nResponse: No\n\n"
           "Input:\nQuestion: {question}\nCorrect Answer: {valid_answers}\nModel Answer: {answer}\nResponse: ")

JUDGE_FOR_SET = {"halueval": ("gemma", SIMPLE_QA, "score"), "math": ("gemma", MATH_QA, "yesno"),
                 "trivia_qa": ("gpt4o", TRIVIA_QA, "json"), "web_questions": ("gpt4o", TRIVIA_QA, "json"),
                 "llama_questions": ("gpt4o", LLAMA_Q, "score")}


def answer_variants(gold):
    """Our gold string ("x | aliases: a, b" or "a;b") -> the list MoshiRAG's extract_answer_variants
    feeds the judge as str(list)."""
    g = str(gold)
    parts = re.split(r"\s*\|\s*aliases:\s*", g, maxsplit=1)
    out = [p.strip() for p in re.split(r"\s*;\s*", parts[0]) if p.strip()]
    if len(parts) > 1:
        out += [a.strip() for a in parts[1].split(",") if a.strip()]
    seen, uniq = set(), []
    for a in out:
        if a.lower() not in seen:
            seen.add(a.lower()); uniq.append(a)
    return uniq


def strip_tags(text):
    return re.sub(r"\s+", " ", re.sub(r"<(?:ret|span)>", " ", text or "")).strip()


class MoshiRagJudge:
    """LLMJudge.__call__: None when the answer text (or question) is empty; up to 3 tries; parse per
    judge class. -1 (unparseable) and None are excluded from the averages, as in score.py."""

    def __init__(self, mode):
        self.kind, self.template, self.parse = JUDGE_FOR_SET[mode]
        if self.kind == "gemma":
            self.url = os.environ.get("MOSHIRAG_GEMMA_JUDGE_URL", "http://localhost:8007").rstrip("/") + "/v1/chat/completions"
            self.model = os.environ.get("MOSHIRAG_GEMMA_JUDGE_MODEL", "google/gemma-3-27b-it")
            self.temperature, self.headers = 1.0, {}
        else:
            self.url = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-4o-2024-08-06"
            self.temperature = 0.0
            self.headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
        self.sess = requests.Session()

    def _parse(self, resp):
        if self.parse == "yesno":
            first = re.sub(r"[^a-z]", "", (resp.strip().splitlines() or [""])[0].lower())
            return 1 if first == "yes" else 0 if first == "no" else -1
        if self.parse == "json":
            body = resp.strip()
            m = re.search(r"\{.*\}", body, re.S)
            try:
                js = ast.literal_eval(m.group(0)) if m else json.loads(body)
            except Exception:
                try:
                    js = json.loads(m.group(0))
                except Exception:
                    return -1
            v = str(js.get("judgment", "")).lower()
            return 1 if v == "correct" else 0 if v == "incorrect" else -1
        found = re.findall(r"[Tt]he score is \[(Correct|Incorrect)\]", resp)
        if found:
            return 1 if found[0] == "Correct" else 0
        low = resp.lower()
        return 0 if "incorrect" in low else 1 if "correct" in low else -1

    def __call__(self, question, gold, text):
        if not text or not text.strip() or not question:
            return None
        prompt = self.template.format(question=question, answer=text, valid_answers=str(answer_variants(gold)))
        body = {"model": self.model, "max_tokens": 512, "temperature": self.temperature, "top_p": 1.0,
                "messages": [{"role": "user", "content": prompt}]}
        for _ in range(3):
            try:
                r = self.sess.post(self.url, json=body, headers=self.headers, timeout=120)
                resp = (r.json()["choices"][0]["message"]["content"] or "").strip()
                assert resp
                return self._parse(resp)
            except Exception:
                continue
        return None
