<div align="center">

  <!-- TODO(seonghyeon): Dueta logo goes here once it exists — 2026-08-20 AI-SLM meeting: derive it from the
       BAZE app colour scheme (violet), reviewers 선영·상현·경수; 8/27: 경수님께 2시간 한도로 요청.
       <img width="240" alt="Dueta" src="assets/figures/logo.png"> -->

  <h1>Context Spanning</h1>
  <h3>A Communication Framework for Full-Duplex Speech Models and External LLM Backends</h3>

  <br>

  <p>
    <b>Seonghyeon Go</b> &nbsp;•&nbsp; <b>Yongwoo Kim</b> &nbsp;•&nbsp; <b>Hyeonjin Cha</b> &nbsp;•&nbsp; <b>Jaeho Shin</b>
  </p>
  <p>
    <b>Mindlogic</b>
  </p>

  <br>

  <!-- TODO(seonghyeon): replace XXXX.XXXXX with the arXiv id once the preprint is up -->
  [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-context--spanning--7b-yellow)](https://huggingface.co/mindlogicinc/context-spanning-7b) [![Base Model](https://img.shields.io/badge/base-PersonaPlex--7B-76b900)](https://huggingface.co/nvidia/personaplex-7b-v1) [![Code License](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE) [![Weights License](https://img.shields.io/badge/weights-PersonaPlex%20license-76b900)](https://huggingface.co/nvidia/personaplex-7b-v1)
  <!-- TODO(seonghyeon): add a Demo / Project Page badge here if one goes live -->

</div>

<p align="center">
  <img src="assets/figures/architecture.png" alt="Context Spanning architecture: the full-duplex frontend streams user speech, agent speech and agent text on one timeline; on the ret token the backend runs ASR, an LLM over the Context Memory DB and a tool, and the result is written back into the stream as a Context Span." width="100%">
</p>

Official PyTorch implementation of **Context Spanning**, a communication framework that lets a
full-duplex speech model call an external LLM backend (retrieval, tool calls, memory) *while it keeps
listening and talking*, and read the result back **as-is** as a block of text tokens in its own context
stream. Code, the released **DuetaSpan-7B** weights, the tool bank and the benchmark harness are all here.

Keywords: full-duplex spoken dialogue, speech-to-speech models, retrieval-augmented generation, tool calling

## Abstract

Full-duplex spoken dialogue models can listen and speak simultaneously like the real-time dynamics of
human conversation. For natural dialogue, the ability to search external information in real-time is also
an important capability. Many models remain trapped in parametric knowledge, leaving them unable to access
real-time information and tool execution. Furthermore, even when LLMs retrieve information, many models are
designed to reason over that information in the latent space rather than as-is, which can result in a loss
of detail. To address this issue, we propose **Context Spanning**, a framework for information exchange
between a full-duplex speech model and an external LLM backend. It feeds the retrieved information to the
speech model as-is, enabling it to reason over the information independently and generate responses. With
this approach, our model achieves high performance on everyday conversation in the Full-Duplex-Benchmark
and strong results on Question Answering tasks, demonstrating its potential.

## How it works

<p align="center">
  <img src="assets/figures/context_span.png" alt="A Context Span on the Moshi token frame: after the ret token the backend's answer is written into the text row between sos and eos, with silence and a sine placeholder in the audio rows, and prefilled in a single KV pass." width="90%">
</p>

A full-duplex speech model calls an external backend while it keeps listening and talking. When the
model emits `<ret>`, the recent user audio is transcribed, the backend (a tool router over a tool bank,
with an LLM for knowledge questions) returns one reference, and that reference is written into the
model's context stream as a masked *Context Span* block at whatever frame it arrives. The frame clock
never waits. Training and inference share one sequence convention (`contextspan/model/sequence_convention.py`).

- **One primitive.** Retrieval, tools and memory all come back through the same `<sos> … <eos>` span, so
  the model reads exact values (time, temperature, prices, distances) instead of a latent summary.
- **No stall.** The span is prefilled in a single KV pass after the acoustic delay; the model only
  updates its cache and keeps stepping at the 80 ms frame clock.
- **Same convention in training and inference.** `<ret>` = 4, span open = 12, span close = 13, silence
  in the agent audio rows and a sine placeholder in the user rows for the span frames.

## Contents

- [Released Weights](#released-weights)
- [Results](#results)
- [Install](#install)
- [Backends](#backends)
- [Run](#run)
- [Training](#training)
- [Evaluation](#evaluation)
- [Layout](#layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Released Weights

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

## Results

Tables 1 and 2 of the paper. Underlined models in the paper are reprinted from the original benchmark
papers (Full-Duplex-Bench) and from MoshiRAG (spoken QA and math); their rows are marked † below.

<!-- TODO(seonghyeon): confirm these are the camera-ready numbers before the repo goes public -->

**Table 1. Full-Duplex-Bench v1 and v3.** Arrows give the direction of better.

| Task | Metric | **Ours** | PersonaPlex† | Moshi† | Gemini† |
|---|---|---:|---:|---:|---:|
| *v1* | | | | | |
| Pause Handling | TOR (candor) ↓ | 0.866 | 0.431 | 0.980 | 0.310 |
| | TOR (synthetic) ↓ | 0.956 | 0.358 | 0.985 | 0.255 |
| Backchannel | TOR ↓ | 0.673 | 0.273 | 1.000 | 0.091 |
| | Freq ↑ | **0.184** | 0.042 | 0.001 | 0.012 |
| | JSD ↓ | 0.700 | 0.662 | 0.957 | 0.896 |
| Smooth Turn-Taking | TOR ↑ | **1.000** | 0.908 | 0.941 | 0.655 |
| | Latency ↓ | **0.145** | 0.170 | 0.265 | 1.301 |
| User Interruption | TOR ↑ | 0.970 | 0.950 | 1.000 | 0.891 |
| | GPT-4o score ↑ | 4.17 | 4.290 | 0.765 | 3.376 |
| | Latency ↓ | 0.771 | 0.240 | 0.257 | 1.183 |
| *v3* | | | | | |
| Tool Use | Pass@1 ↑ | **0.56** | – | – | 0.540 |
| | Tool Selection ↑ | **0.910** | – | – | 0.817 |
| | Arg. Accuracy ↑ | **0.653** | – | – | 0.588 |
| | Resp. Quality ↑ | 0.242 | – | – | 0.718 |

**Table 2. Spoken QA and math reasoning (accuracy, %).** `ref.` = the reference document is provided;
`resp.` = the model's response. Our router is Gemma 4.

| Model | LlamaQ ref. | LlamaQ resp. | WebQ ref. | WebQ resp. | TriviaQA ref. | TriviaQA resp. | HaluEval ref. | HaluEval resp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4-Voice† | | 64.7 | | 32.2 | | 39.1 | | 21.2 |
| STITCH-S† | | 73.3 | | 50.2 | | 50.0 | | – |
| MoshiRAG (Gemma 3)† | 83.0 | 80.3 | 71.5 | 67.2 | 73.7 | 69.6 | 42.0 | 36.3 |
| MoshiRAG (GPT-4.1)† | 87.8 | 80.6 | 77.7 | 68.9 | 86.8 | 78.2 | 61.2 | 51.3 |
| MoshiRAG (Tavily)† | 84.6 | 78.2 | 73.5 | 66.1 | 84.9 | 77.5 | 54.3 | 47.0 |
| Vanilla Moshi† | | 62.3 | | 26.6 | | 22.8 | | 10.5 |
| **Ours (Gemma 4)** | **88.3** | **81.7** | 60.0 | 48.3 | 86.7 | 38.3 | 46.7 | 26.7 |

| Model | AddSub ref. | AddSub resp. | MultiArith ref. | MultiArith resp. | SinglEq ref. | SinglEq resp. | SVAMP ref. | SVAMP resp. | GSM8K ref. | GSM8K resp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4-Voice† | | 59.4 | | 62.0 | | 71.0 | | 4.0 | | 29.0 |
| STITCH-S† | | 81.7 | | 87.9 | | 91.7 | | 72.2 | | 56.7 |
| MoshiRAG (Gemma 3)† | 76.6 | 61.7 | 87.1 | 69.0 | 83.2 | 68.2 | 74.1 | 55.0 | 66.2 | 33.9 |
| MoshiRAG (GPT-4.1)† | 87.9 | 64.8 | 87.1 | 76.0 | 89.6 | 72.9 | 80.5 | 61.1 | 70.8 | 43.2 |
| Vanilla Moshi† | | 8.3 | | 9.8 | | 18.4 | | 9.7 | | 2.1 |
| **Ours (Gemma 4)** | 75.0 | 50.0 | 65.0 | 55.0 | 65.0 | 55.0 | 60.0 | 25.0 | 35.0 | 30.0 |

The step-8000 checkpoint on the Hub was measured again after the paper runs, on the `semi` protocol
(120 items per set); those numbers are in [Released Weights](#released-weights) and the two must not be
mixed. Protocols, arms and scorers: [`eval/README.md`](eval/README.md).

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

## Citation

<!-- TODO(seonghyeon): fill in the arXiv id / venue once known -->

```bibtex
@article{go2026contextspanning,
  title   = {Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends},
  author  = {Go, Seonghyeon and Kim, Yongwoo and Cha, Hyeonjin and Shin, Jaeho},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Acknowledgements

The speech model is fine-tuned from [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1)
(NVIDIA), which builds on [Moshi](https://github.com/kyutai-labs/moshi) and the Mimi codec (Kyutai). The
retrieval pipeline, the data generation pipeline and the RAG-suite protocol follow
[MoshiRAG](https://arxiv.org/abs/2604.12928). Full-Duplex-Bench v1/v3 and their scorers are the
benchmark authors' own. `contextspan/moshi/` is the PersonaPlex fork of Kyutai's `moshi` package (MIT).

## License

Code: MIT — see `LICENSE`. `contextspan/moshi/` carries its own license files. The released weights are
fine-tuned from `nvidia/personaplex-7b-v1` and inherit the
[PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1) (NVIDIA Open Model License;
PersonaPlex builds on `kyutai/moshiko-pytorch-bf16`, CC-BY-4.0). The voice prompts are derived from VCTK
(CC BY 4.0). See the [model card](https://huggingface.co/mindlogicinc/context-spanning-7b).
