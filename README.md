# Context Spanning

Official code for **Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends**.

A full-duplex speech model calls an external backend while it keeps listening and talking. When the
model emits `<ret>`, the recent user audio is transcribed, the backend (a tool router over a tool bank,
with an LLM for knowledge questions) returns one reference, and that reference is written into the
model's context stream as a masked *Context Span* block at whatever frame it arrives. The frame clock
never waits. Training and inference share one sequence convention (`contextspan/model/sequence_convention.py`).

## Weights

[mindlogicinc/context-spanning-7b](https://huggingface.co/mindlogicinc/context-spanning-7b) — **DuetaSpan v7, step 8000**,
fine-tuned from [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) on
`manifest_v6h`: 823,659 dialogues, ~10,009 h indexed, every dialogue passed a frame-level audio QA and a
full-text QA. Two-group selective text loss: the tokens of the asked-for answer form one group and the rest of the
text row the other, weighted by a detached softmax over the two group losses; audio codebook losses as in PersonaPlex;
span and prefix columns masked out. Sequence convention: `<ret>` = 4, Context Span open = 12, close = 13 (checkpoints
trained before 2026-09-08 used 12 on both sides: set `CS_SPAN_CLOSE_ID=12`). Three released voices (`f0`, `f1`, `f2`).

| benchmark (step 8000) | resp | ref | P(resp \| ref) | `<ret>` rate |
|---|---|---|---|---|
| HaluEvalAudio (120, router: Gemma-4-26B-A4B) | 0.642 | 0.725 | **0.851** | 0.925 |
| math word problems (40) | 0.80 | 0.80 | **1.00** | 0.925 |
| Full-Duplex-Bench v1.0 (40/task) | pause TOR 0.725 (lower is better) · interruption rating 4.43 / take-turn 0.925 / latency 1.21 s · backchannel TOR 0.65, JSD 0.73 | | | |

Protocols and scorers: [`eval/README.md`](eval/README.md). All numbers are `resp`/`ref` accuracy in [0, 1], higher is better.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
export HF_TOKEN=<token with access to nvidia/personaplex-7b-v1 and the weights repo>
```

Everything the model needs is in this repository: `contextspan/moshi/` is the PersonaPlex fork of Kyutai's
`moshi` package (MIT, license files alongside). Only the weights come from the Hub; `CS_BASE_DIR` /
`CS_WEIGHTS_DIR` point at local copies instead.

## Backends

Two servers: the router LLM (OpenAI-compatible; default Gemma-4-26B-A4B on vLLM) and an ASR endpoint
(`POST /transcribe` -> `{"text"}`; default Qwen3-ASR). `scripts/backends.sh` starts both with the settings
every number here was measured with; `scripts/env.sh` exports the endpoints.

```bash
pip install -e '.[asr-server]' vllm
bash scripts/backends.sh start        # router on GPU 1, ASR on GPU 0; waits until both answer
source scripts/env.sh
```

The defaults assume two GPUs: the router alone takes ~82 GB at `ROUTER_MEM=0.85`, and the ASR (~5 GB)
shares the other GPU with the speech model (~20 GB). On a single 96 GB GPU start it with
`ROUTER_GPUS=0 ASR_GPU=0 ROUTER_MEM=0.6 bash scripts/backends.sh start`; the span latency figures below
were measured on the two-GPU layout.

On `<ret>` the router picks ONE tool (or answers a knowledge question directly); the tool runs for real —
time, weather, prices, web search, places and routes, SGD-seeded bookings — and its result is the span.
The shipped bank is the 60 tools whose result is something a voice assistant says
([`contextspan/datasets/moshicp/TOOLS.md`](contextspan/datasets/moshicp/TOOLS.md)). Servers, GPU layout,
vLLM flags, latency knobs and the environment variables: [`docs/BACKENDS.md`](docs/BACKENDS.md).

## Run

```bash
# a wav through the model; out.wav is stereo, L = user, R = agent
python main.py infer --input-wav assets/test/question.wav --output-wav out.wav --voice f0 \
    --output-text out.json --user-name Priya --user-city Sydney

# live conversation in the browser (microphone needs https or localhost)
python main.py serve --voice f0 --host 0.0.0.0 --port 8080 --token <value>
```

The page shows what the ASR heard, the agent's words, and every span as it lands with the tool it came
from; name, location and timezone fill the Context DB profile and the model's prefix. Stop offers the
conversation as a stereo wav and a JSON transcript. Wire format: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Training

```bash
python main.py prepare --in-dir data/raw --out-dir data/prepared
python main.py train   --data-dir data/prepared --out-dir runs/ft
```

`prepare` encodes dialogues (JSON + stereo wav) into tensors; `train` splices Context Span blocks at
MoshiRAG-sampled delays and fine-tunes with the selective loss. Data format and the v7 recipe:
[`docs/TRAINING.md`](docs/TRAINING.md).

## Evaluation

`eval/` is the benchmark harness — MoshiRAG RAG suite, Full-Duplex-Bench v1/v1.5/v2/v3, a live-session
benchmark — separate from the runtime and run on the same stack a user talks to. See [`eval/README.md`](eval/README.md).

## Layout

| path | role |
|---|---|
| `main.py` | entry point: `infer` / `serve` / `prepare` / `train` |
| `contextspan/model/` | the speech model: `weights.py`, `engine.py`, `context_span_block.py`, `sequence_convention.py` |
| `contextspan/runtime/` | the live loop: `frame_stream.py` (wav), `websocket_server.py` + `web/` (browser), `user_leveller.py`, `default_persona.py` |
| `contextspan/training/` | `prepare.py`, `finetune.py` |
| `contextspan/duetaspan/` | the backend runtime: tool router, tool bank + MCP servers, LLM-RAG, Context DB, ASR client and server |
| `contextspan/datasets/` | the shipped tool bank, its SQLite world and geo index |
| `contextspan/moshi/` | vendored PersonaPlex fork of `moshi` (third-party) |
| `eval/` | benchmark harness |
| `scripts/` | `backends.sh`, `env.sh`, `prefill_timing.py` |
| `docs/` | `BACKENDS.md`, `PROTOCOL.md`, `TRAINING.md`, `LAYOUT.md` (rename map from the previous layout) |

## License

MIT — see `LICENSE`. `contextspan/moshi/` carries its own license files.
