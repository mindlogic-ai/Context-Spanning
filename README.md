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
[`contextspan/datasets/moshicp/TOOLS.md`](contextspan/datasets/moshicp/TOOLS.md).

Two servers are needed. `scripts/backends.sh` starts them with the settings every number in this
repository was measured with, so a fresh machine gets the same span latency without tuning:

```bash
pip install -e '.[asr-server]' vllm
bash scripts/backends.sh start        # router LLM: vLLM Gemma-4-26B-A4B on GPU 1; ASR: Qwen3-ASR on GPU 0
source scripts/env.sh                 # MCP_ROUTER_* / MOSHICP_RAG_LLM_* / MOSHICP_ASR_URL / JUDGE_LLM_*
```

The vLLM flags matter for latency and are explained in the script: `--max-model-len 8192` (the router
prompt carries the tool catalogue, ~2.9k-3.5k tokens, plus the conversation), n-gram speculative decoding
(the router's replies repeat the prompt, which cut its round trip by ~40% at temperature 0), prefix
caching (every call shares the 10k-character system prompt). `ROUTER_GPUS` / `ROUTER_TP` / `ASR_GPU`
select GPUs; `ROUTER_MEM` (0.58) sizes the router's share of its GPU (~56 GB of 96 GB, leaving room for
the ASR and the speech model on the same card). The router server also answers the RAG fallback and is the
eval judge; set `RAG_MODEL` (and `RAG_GPUS`) to put those two on a separate server so a smaller model can
take the tool pick — the acceptance test for a smaller router is FDB v3 tool-selection / argument accuracy
(`eval/`), which is exactly the job it would do. Any OpenAI-compatible server and any `POST /transcribe -> {"text"}` ASR (a Whisper endpoint,
for instance) can replace them; a hosted LLM API takes `MCP_ROUTER_LLM_KEY` / `MOSHICP_RAG_LLM_KEY`.
Tool results and the SQLite world live under `contextspan/datasets/` (override with `DUETASPAN_DATA`).

Latency knobs, all with the defaults the benchmarks used: the router asks the server for a JSON object
(`MCP_ROUTER_JSON_MODE=1`, dropped automatically if the server rejects it), decodes at most
`MCP_ROUTER_MAX_TOKENS=120` tokens and waits at most `MCP_ROUTER_TIMEOUT_S=8`; a `get_time` / `get_weather`
without an explicit place takes the timezone / city from the user profile instead of a second LLM call; each
tool HTTP call has a 4 s budget and a transport failure yields no span rather than a sentence about the
failure; a span that would land more than `CS_RET_DEADLINE_S=2.5` s after `<ret>` is dropped (`late`
event) — the training corpus's ret-to-span delays have p99 2.3 s, and a span that arrives after the model
has already answered is worse than none. Router decode is ~95% of a retrieval, which is why the router is Gemma-4-26B-A4B (4B active parameters);
`ROUTER_MODEL` swaps it for any OpenAI-compatible model.

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

Your speech is transcribed continuously, one utterance at a time (a **You:** line updates while you speak
and freezes when you stop); on `<ret>` the transcript of the sentence you just finished is the question
sent to the router, a sentence still in progress is waited for (at most `CS_RET_UTT_WAIT_S`, 1 s), and the
line that was routed is marked. Browsers open the microphone only over https or on localhost. The page shows the agent's text and
each injected span as it lands; name, location and timezone are the Context DB profile, the **Knowledge** field its notes;
the router sees the profile, the user's turns and every tool result (call and value). Stop offers the conversation as a stereo wav and a
JSON transcript (`ret` / `question` / `span` / `no_span` / `text` events with timing; `question` is what the ASR heard, so a wrong span can be traced to the ASR or to the backend). "Clone my voice" is done before a conversation: it records 12 s of your speech on the connection, and Start then talks in that voice (the transcript shows what the ASR heard as **You:** lines, each span with the tool or source it came from).

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
