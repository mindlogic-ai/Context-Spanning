# Backends

The backend is the DuetaSpan runtime in `contextspan/duetaspan/`: the tool router, the tool bank with its
SQLite world and MCP tool servers, the LLM-RAG fallback for knowledge questions, the Context DB (user
profile + conversation log the router sees) and the ASR client. On `<ret>` the router LLM picks ONE tool
or answers directly; a tool runs for real and its result is the span; a request that needs no external
knowledge injects nothing.

## Servers

| server | default | port | GPU | started by |
|---|---|---|---|---|
| router LLM (also RAG fallback and eval judge) | `google/gemma-4-26B-A4B-it` on vLLM | 8004 | `ROUTER_GPUS` (1) | `scripts/backends.sh start` |
| ASR | Qwen3-ASR-1.7B, `python -m contextspan.duetaspan.runtime.asr_server` | 8990 | `ASR_GPU` (0) | `scripts/backends.sh start` |
| RAG / judge (optional second server) | `RAG_MODEL` when set | `RAG_PORT` (8005) | `RAG_GPUS` | `scripts/backends.sh start` |

```bash
pip install -e '.[asr-server]' vllm
bash scripts/backends.sh start|status|stop
source scripts/env.sh          # MCP_ROUTER_* / MOSHICP_RAG_LLM_* / MOSHICP_ASR_URL / JUDGE_LLM_*
```

Any OpenAI-compatible server replaces the router (`ROUTER_MODEL`, or the `MCP_ROUTER_LLM_*` variables
directly; a hosted API takes `MCP_ROUTER_LLM_KEY` / `MOSHICP_RAG_LLM_KEY`). Any `POST /transcribe ->
{"text"}` endpoint replaces the ASR (`MOSHICP_ASR_URL`). The ASR server accepts `language` as a name or a
code (#27).

Footprint on one 96 GB GPU with everything on it: the router needs its own GPU (~82 GB at `ROUTER_MEM`
0.85 — the A4B weights alone are 48.5 GiB and the 8192-token KV cache needs the rest; 0.58 fails to start);
the ASR (~5 GB) and the speech model (~20 GB) share another. The router server also answers the RAG
fallback and judges evaluations; set `RAG_MODEL` (and `RAG_GPUS`) to put those two on a separate server so
a smaller model can take the tool pick. The acceptance test for a smaller router is FDB v3 tool-selection
and argument accuracy (`eval/`), which is exactly the job it would do (#13).

## Why these vLLM flags

Measured on RTX PRO 6000 Blackwell, 2026-09:

- `--max-model-len 8192` — the router prompt carries the tool catalogue (~2.9k tokens at the first stage,
  ~3.5k for the largest tool group) plus the conversation; 4k truncates.
- `--speculative-config ngram` — the router's replies repeat the prompt (tool names, argument values, the
  sentence it copies from the Context DB); prompt-lookup speculation cut the round trip by ~40% at
  temperature 0 with no output change.
- `--enable-prefix-caching` — every call shares the same 10k-character system prompt + catalogue prefix.
- Gemma-4-26B-A4B (26B MoE, 4B active) — the router's decision is a classification plus a few argument
  strings and its cost is decode, so the 4B-active model answers in roughly a third of the time of a dense
  27B at the same quality on the tool-selection probes. Router decode is ~95% of a retrieval.

## Latency knobs (defaults = what the benchmarks used)

| variable | default | effect |
|---|---|---|
| `MCP_ROUTER_JSON_MODE` | 1 | the router asks the server for a JSON object; dropped automatically if the server rejects it |
| `MCP_ROUTER_MAX_TOKENS` | 120 | router decode budget |
| `MCP_ROUTER_TIMEOUT_S` | 8 | router call budget |
| MCP server HTTP calls (weather, finance, web search) | 4 s | a transport failure yields no span, never a sentence about the failure (#10) |
| `MCP_MAPS_BUDGET_S` | unset | when set, the map adapters make one attempt with this timeout instead of their 10-20 s retries |
| `CS_RET_DEADLINE_S` | 2.5 | a span that would land later than this after `<ret>` is dropped (`late` event): the corpus's ret-to-span delays have p99 2.3 s, and a span that arrives after the model has answered is worse than none |
| `CS_RET_UTT_WAIT_S` | 1.0 | on `<ret>` mid-sentence, how long the question waits for the utterance to end before the fixed window is transcribed (#7) |
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
