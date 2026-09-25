# Backends

The backend is the DuetaSpan runtime in `contextspan/duetaspan/`: the tool router, the tool bank with its
SQLite world and MCP tool servers, the Context DB (user profile + conversation log the router sees) and
the ASR client. On `<ret>` the router LLM is the single decider: it picks ONE tool, answers directly, or
abstains. A tool runs for real and its result is the span; a request that needs no external knowledge
injects nothing.

## Servers

| server | default | port | GPU | started by |
|---|---|---|---|---|
| router LLM | `google/gemma-4-26B-A4B-it` on vLLM | 8004 | `ROUTER_GPUS` (1) | `scripts/backends.sh start` |
| ASR | Qwen3-ASR-1.7B: `qwen-asr-serve` on vLLM (`ASR_BACKEND=vllm`, default) or `python -m contextspan.duetaspan.runtime.asr_server` (`ASR_BACKEND=transformers`) | 8990 | `ASR_GPU` (0) | `scripts/backends.sh start` |
| rewrite LLM (optional second server) | `RAG_MODEL` when set | `RAG_PORT` (8005) | `RAG_GPUS` | `scripts/backends.sh start` |

```bash
pip install -e '.[asr-server]' vllm
bash scripts/backends.sh start|status|stop
source scripts/env.sh          # MCP_ROUTER_LLM_* / MOSHICP_RAG_LLM_* / MOSHICP_ASR_URL / JUDGE_LLM_*
```

Any OpenAI-compatible server replaces the router (`ROUTER_MODEL`, or the `MCP_ROUTER_LLM_*` variables
directly; a hosted API takes `MCP_ROUTER_LLM_KEY` / `MOSHICP_RAG_LLM_KEY`). The ASR is any OpenAI audio
API (`MOSHICP_ASR_URL` ending in `/v1/audio/transcriptions`, `MOSHICP_ASR_MODEL` = served name) or any
`POST /transcribe -> {"text"}` endpoint. The transformers server accepts `language` as a name or a code.

The two ASR servers run the same weights; they differ in speed under the runtime's load. The runtime sends a
partial transcript every 1.6 s while the user speaks and the final one when the utterance ends, and the
transformers server (plain `HTTPServer`, one request at a time, sharing its GPU with the speech model) makes
the final transcript of a question wait behind the partial: 0.45 s median, up to 1.1 s, on a 6 s question —
half of the `<ret>` -> span time. vLLM batches them: 0.13 s for the same clip, 0.76 s for a 31 s one (1.5 s
on transformers). `ASR_MEM=0.12` is enough for the 1.7B model at `--max-model-len 4096`.

Footprint on one 96 GB GPU with everything on it: the router needs its own GPU (~82 GB at `ROUTER_MEM`
0.85 — the A4B weights alone are 48.5 GiB and the 8192-token KV cache needs the rest; 0.58 fails to start);
the ASR (~5 GB) and the speech model (~20 GB) share another. The same server also rewrites agent-addressed
questions ("your latest song" -> "the latest song by <persona>") before they are routed, through the
`MOSHICP_RAG_LLM_*` variables; set `RAG_MODEL` (and `RAG_GPUS`) to put that call on a separate server so a
smaller model can take the tool pick. The acceptance test for a smaller router is FDB v3 tool-selection and
argument accuracy (`benchmark/`), which is exactly the job it would do.

## Why these vLLM flags

- `--max-model-len 8192` — the router prompt carries the tool catalogue (~2.9k tokens at the first stage,
  ~3.5k for the largest tool group) plus the conversation; 4k truncates.
- `--speculative-config ngram` — the router's replies repeat the prompt (tool names, argument values, the
  sentence it copies from the Context DB), so prompt-lookup speculation shortens the round trip without
  changing the output at temperature 0.
- `--enable-prefix-caching` — every call shares the same 10k-character system prompt + catalogue prefix.
- Gemma-4-26B-A4B (26B MoE, 4B active) — the router's decision is a classification plus a few argument
  strings and its cost is decode, so a 4B-active model answers much faster than a dense 27B. Router decode
  dominates the cost of a retrieval.

## Latency knobs

| variable | default | effect |
|---|---|---|
| `MCP_ROUTER_JSON_MODE` | 1 | the router asks the server for a JSON object; dropped automatically if the server rejects it |
| `MCP_ROUTER_MAX_TOKENS` | 120 | router decode budget |
| `MCP_ROUTER_TIMEOUT_S` | 8 | router call budget |
| MCP server HTTP calls (weather, finance, web search) | 4 s | a transport failure yields no span, never a sentence about the failure |
| `MCP_MAPS_BUDGET_S` | unset | when set, the map adapters make one attempt with this timeout instead of their 10-20 s retries |
| `CS_RET_DEADLINE_S` | 2.5 | a span that would land later than this after `<ret>` is dropped (`late` event): the corpus's ret-to-span delays have p99 2.3 s, and a span that arrives after the model has answered is worse than none |
| `CS_RET_UTT_WAIT_S` | 1.0 | on `<ret>` mid-sentence, how long the question waits for the utterance to end before the fixed window is transcribed |
| `CS_USER_TARGET_LUFS` | -24.0 | loudness the user channel is levelled to (the training median); `--raw-user-audio` bypasses it |

A `get_time` / `get_weather` without an explicit place takes the timezone / city from the user profile
instead of a second LLM call. Tool results and the SQLite world live under `contextspan/datasets/`
(override the root with `DUETASPAN_DATA`; see `contextspan/duetaspan/common/paths.py`).

## Tool bank

Of DuetaSpan's 125-tool bank this package mounts the 60 tools whose result is something a voice assistant
says to the listener; browser automation, filesystem, web crawling, PayPal back-office, coordinate/IP/
elevation and app deep-link tools are excluded. The list and the reasoning:
[`contextspan/datasets/moshicp/TOOLS.md`](../contextspan/datasets/moshicp/TOOLS.md). Benchmarks may
mount a different universe (FDB v3 runs with only the benchmark's 12 mock APIs, `MOSHICP_TOOLPACK_ONLY=1`).
