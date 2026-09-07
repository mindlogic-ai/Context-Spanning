# Context Spanning

Official code for **Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends**.

Context Spanning lets a full-duplex speech model call an external backend while it keeps listening and
talking: when the model emits `<ret>`, the recent user audio is transcribed, the backend returns one
reference sentence, and that sentence is written into the model's context stream as a masked
*Context Span* block at whatever frame it arrives. Training and inference share one sequence convention.

Weights: [mindlogicinc/context-spanning-7b](https://huggingface.co/mindlogicinc/context-spanning-7b) (fine-tuned from `nvidia/personaplex-7b-v1`).

| file | what it is |
|---|---|
| `contextspan/model.py` | model loading, the streaming `Engine`: persona prefix, `step`, `inject_context_span`, `clone_voice` |
| `contextspan/inject.py` | the Context Span block, read in one batched forward |
| `contextspan/spans.py` | the sequence conventions shared by training and inference |
| `contextspan/duetaspan/` | the DuetaSpan runtime backend, unchanged: tool router, tool bank + MCP servers, LLM-RAG, Context DB, ASR client |
| `contextspan/datasets/` | the shipped tool bank (60 tools, see `TOOLS.md`), the SGD-seeded world and the geo index |
| `eval/` | the benchmark harness (MoshiRAG RAG suite, Full-Duplex-Bench v1/v1.5/v2/v3), separate from the runtime |
| `contextspan/stream.py` | frame-clock loop for a wav file |
| `contextspan/serve.py` | WebSocket server + browser page |
| `contextspan/train.py` | data preparation and fine-tuning |
| `main.py` | `infer` / `serve` / `prepare` / `train` |

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
export HF_TOKEN=<token with access to nvidia/personaplex-7b-v1 and the weights repo>
```

Everything the model needs is in this repository: `contextspan/moshi/` is the PersonaPlex fork of
Kyutai's `moshi` package (MIT; license files alongside). Nothing is fetched from another code
repository; only the weights come from the Hub.

`CS_BASE_DIR` / `CS_WEIGHTS_DIR` point at local copies of the PersonaPlex base and of the weights
repo (`context_spanning_7b.pt`, `voices/*.pt`) instead of downloading them.

## Backends

The backend is the DuetaSpan runtime, vendored unchanged in `contextspan/duetaspan/` (router, tool
bank with its SQLite world, MCP tool servers, LLM-RAG fallback, Context DB, ASR client). On `<ret>` the
router LLM picks ONE tool (or answers a knowledge question directly); the tool runs for real (time,
weather, prices, web search, places and routes, SGD-seeded bookings) and its result is the span.

Of DuetaSpan's 125-tool bank this package uses the **60 tools whose result is something a voice
assistant says to the listener**; browser automation, filesystem, web crawling, PayPal back-office,
coordinate/IP/elevation and app deep-link tools are excluded — the list and the reasoning are in
[`contextspan/datasets/moshicp/TOOLS.md`](contextspan/datasets/moshicp/TOOLS.md). Three servers are
needed; any OpenAI-compatible LLM server works (vLLM shown).

```bash
# router + RAG LLM (one server serves both). The router's prompt carries the tool catalogue: ~2.9k tokens
# for the first-stage pick and ~3.5k for the largest tool group, plus the conversation, so the server
# needs a context of at least 8192 tokens.
vllm serve google/gemma-3-27b-it --port 8004 --max-model-len 8192
export MCP_ROUTER_LLM_API=openai MCP_ROUTER_LLM_URL=http://localhost:8004 MCP_ROUTER_LLM_MODEL=google/gemma-3-27b-it
export MOSHICP_RAG_LLM_URL=http://localhost:8004/v1/chat/completions MOSHICP_RAG_LLM_MODEL=google/gemma-3-27b-it
# ASR: any server that answers POST /transcribe (multipart `file` = wav) with {"text": ...}; the vendored
# Qwen3-ASR server below is one, a Whisper endpoint with the same contract works as well.
pip install -e '.[asr-server]' && python -m contextspan.duetaspan.runtime.asr_server   # :8990
export MOSHICP_ASR_URL=http://localhost:8990/transcribe
```

A hosted API works the same way (`MCP_ROUTER_LLM_URL=https://api.openai.com`, `MCP_ROUTER_LLM_KEY`,
`MOSHICP_RAG_LLM_KEY`). Tool results and the SQLite world live under `contextspan/datasets/` (override
with `DUETASPAN_DATA`).

## Inference

```bash
python main.py infer --input-wav assets/test/question.wav --output-wav out.wav --voice f0 \
    --text-prompt "You are a helpful and friendly voice assistant."
```

`out.wav` is stereo: left = user, right = agent. `--output-text` writes the transcript with the
`<ret>` / span events and their timing.

## Live conversation

```bash
python main.py serve --voice f0 --host 0.0.0.0 --port 8080 --token <value>
```

Browsers open the microphone only over https or on localhost. The page shows the agent's text and
each injected span as it lands; name, location and timezone are the Context DB profile, the **Knowledge** field its notes;
the router sees the profile, the user's turns and every tool result (call and value). Stop offers the conversation as a stereo wav and a
JSON transcript (`ret` / `question` / `span` / `no_span` / `text` events with timing; `question` is what the ASR heard, so a wrong span can be traced to the ASR or to the backend). "Clone my voice" records 12 s and speaks with that voice.

## Evaluation

The benchmark harness (MoshiRAG RAG suite, Full-Duplex-Bench v1/v1.5/v2/v3) lives in [`eval/`](eval/README.md),
separate from the runtime, and runs the same stack a user talks to.

## Training

```bash
python main.py prepare --in-dir data/raw --out-dir data/prepared
python main.py train   --data-dir data/prepared --out-dir runs/ft
```

`data/raw` holds one JSON per dialogue (turns with `speaker`, `text`, and for retrieval turns
`reference`) next to its stereo wav; `prepare` encodes them and splices the Context Span blocks at
sampled retrieval delays; `train` fine-tunes from the PersonaPlex base (or `--checkpoint`).

## License

MIT — see `LICENSE`.
