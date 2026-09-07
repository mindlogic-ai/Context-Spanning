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
| `contextspan/backend.py` | the router (the DuetaSpan router prompt, one OpenAI-compatible LLM call) and the ASR client |
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

The backend is the DuetaSpan runtime, vendored unchanged in `contextspan/duetaspan/` (router,
125-tool bank with its SQLite world, MCP tool servers, LLM-RAG fallback, Context DB, ASR client).
On `<ret>` the router LLM picks ONE tool from the bank (or answers a knowledge question directly);
the tool runs for real (time, weather, prices, web search, maps, browser, SGD-seeded bookings) and
its result is the span. Three servers are needed; any OpenAI-compatible LLM server works (vLLM shown).

```bash
# router + RAG LLM (one server serves both)
vllm serve google/gemma-3-27b-it --port 8004
export MCP_ROUTER_LLM_API=openai MCP_ROUTER_LLM_URL=http://localhost:8004 MCP_ROUTER_LLM_MODEL=google/gemma-3-27b-it
export MOSHICP_RAG_LLM_URL=http://localhost:8004/v1/chat/completions MOSHICP_RAG_LLM_MODEL=google/gemma-3-27b-it
# ASR (Qwen3-ASR; POST /transcribe with a wav -> {"text": ...})
pip install -e '.[asr-server]' && python -m contextspan.duetaspan.runtime.asr_server   # :8990
export MOSHICP_ASR_URL=http://localhost:8990/transcribe
```

A hosted API works the same way (`MCP_ROUTER_LLM_URL=https://api.openai.com`, `MCP_ROUTER_LLM_KEY`,
`MOSHICP_RAG_LLM_KEY`). The 32 browser tools need `pip install -e '.[browser]' && playwright install
chromium`; without them the bank reports 93 tools. Tool results and the SQLite world live under
`contextspan/datasets/` (override with `DUETASPAN_DATA`).

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
