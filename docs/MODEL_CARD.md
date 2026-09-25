---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/personaplex-7b-v1
pipeline_tag: audio-to-audio
language:
  - en
library_name: contextspan
tags:
  - full-duplex
  - spoken-dialogue
  - speech-to-speech
  - retrieval
  - retrieval-augmented-generation
  - tool-calling
  - context-spanning
  - moshi
  - personaplex
---

# Context Spanning · DuetaSpan-7B

> [!Note]
> This repository holds the released **DuetaSpan-7B** weights described in
> *Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends*
> (Go, Kim, Cha, Shin — Mindlogic, 2026). The inference and training code lives at
> [mindlogic-ai/Context-Spanning](https://github.com/mindlogic-ai/Context-Spanning); the weights load only
> through that package.

[![Code](https://img.shields.io/badge/GitHub-Context--Spanning-181717?logo=github)](https://github.com/mindlogic-ai/Context-Spanning)

A full-duplex speech model that keeps listening and talking while it calls an external backend. When the
model emits `<ret>`, the recent user audio is transcribed, a tool router (with an LLM for knowledge
questions) returns one reference, and that reference is written into the model's own context stream
**as-is** as a *Context Span* block. The frame clock never waits, and the model reads exact values
(time, weather, prices, distances) instead of a latent summary.

<p align="center">
  <img src="assets/figures/architecture.png" alt="Context Spanning architecture: the frontend streams user audio, agent audio and agent text on one timeline; after the ret token the agent keeps talking (lead portion) while the backend passes the Streaming ASR user text and the Context DB to an LLM, whose tool result is written back as a Context Span (sine wave on user audio, silence on agent audio, sos … eos on agent text) before the body portion." width="100%">
</p>

<!-- assets/figures/architecture.png is uploaded to this model repo together with the card -->

## Files

| file | content |
|---|---|
| `context_spanning_7b.pt` | `{"model": state_dict}` in bf16, loadable with `main.py infer --checkpoint` or fetched automatically |
| `voices/{main,f0,f1,f2,f3,m0,m1,m2,m3}.pt` | voice prompts (`{"codes": LongTensor[8, P]}`, agent-voice Mimi codes; 8 s each) |
| `assets/figures/` | the paper's figures used by this card |

Mimi and the text tokenizer are taken from the PersonaPlex repository at load time; the `moshi` package is
vendored in the code repository (`contextspan/moshi/`).

## Model Overview

- Type: full-duplex speech-to-speech model (Moshi architecture, PersonaPlex fork)
- Base model: [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1), all trainable parameters fine-tuned
- Parameters: 7B
- Audio codec: Mimi, 12.5 Hz, 8 codebooks per channel (user and agent)
- Frame clock: 80 ms per step
- Sequence convention: `<ret>` = 4, span open = 12, span close = 13, text pad = 3
- Released checkpoint: 2026-09-23
- Backends used for the results: router `google/gemma-4-26B-A4B-it` on vLLM, ASR Qwen3-ASR-1.7B
- Language: English

## Benchmark Results

Full runs of the released checkpoint with the released router (Gemma-4-26B-A4B on vLLM) and ASR (Qwen3-ASR-1.7B);
protocols and scorers are in [`benchmark/README.md`](https://github.com/mindlogic-ai/Context-Spanning/blob/main/benchmark/README.md). Rows marked † are reprinted from the
original benchmark papers (Full-Duplex-Bench, PersonaPlex) or from MoshiRAG. Arrows give the direction of better.

**Table 1. Full-Duplex-Bench v1 (727 clips).**

| Task | Metric | **Ours** | PersonaPlex† | Moshi† | Gemini Live 2.5† |
|---|---|---:|---:|---:|---:|
| Pause Handling | TOR, Candor ↓ | 0.824 | 0.431 | 0.980 | 0.310 |
| | TOR, synthetic ↓ | 0.861 | 0.358 | 0.985 | 0.255 |
| Backchannel | TOR ↓ | 0.782 | 0.273 | 1.000 | 0.091 |
| | Frequency (per s) ↑ | 0.150 | 0.042 | 0.001 | 0.012 |
| | JSD ↓ | 0.716 | 0.662 | 0.957 | 0.896 |
| Turn-Taking | TOR ↑ | 0.899 | 0.908 | 0.941 | 0.655 |
| | Latency (s) ↓ | 0.064 | 0.170 | 0.265 | 1.301 |
| User Interruption | TOR ↑ | 0.925 | 0.950 | 1.000 | 0.891 |
| | GPT-4o rating (0-5) ↑ | 3.924 | 4.290 | 0.765 | 3.376 |
| | Latency (s) ↓ | 0.598 | 0.240 | 0.257 | 1.183 |

TOR is the take-over rate, the fraction of clips in which the model takes the turn; JSD is the Jensen-Shannon
divergence from the human backchannel timing distribution.

**Table 2. Full-Duplex-Bench v3 (100 scenarios, 12 mock tools).** MoshiRAG is the released `kyutai/moshika-rag`
checkpoint run with its own inference code, its reference LLM replaced by the same tool router and prompt as Ours
(router LLM Gemma-3-27B-it).

| Task | Metric | **Ours** | MoshiRAG | GPT-Realtime† | Gemini Live 3.1† |
|---|---|---:|---:|---:|---:|
| Tool Use | Tool selection ↑ | 0.855 | 0.738 | 0.876 | 0.817 |
| | Argument accuracy ↑ | 0.567 | 0.440 | 0.680 | 0.588 |
| | Response quality ↑ | 0.411 | 0.255 | 0.792 | 0.718 |
| | Pass rate ↑ | 0.470 | 0.280 | 0.600 | 0.540 |
| Turn-Taking Dynamics | Take-turn rate (%) ↑ | 95.0 | 94.0 | 96.0 | 78.0 |
| | Latency, task completion (s) ↓ | 5.83 | 7.69 | 6.89 | 4.25 |
| | Interruption rate (%) ↓ | 73.7 | 58.5 | 13.5 | 19.2 |
| | Filler rate (%) ↓ | 96.0 | 92.3 | 16.9 | 31.7 |

Tool-use metrics are fractions of the 100 scenarios; rates are percentages; latency is the task-completion
latency in seconds.

**Table 3. Spoken QA and math reasoning (accuracy, %).** `ref.` = the router's reference answer is judged;
`resp.` = the model's spoken response is judged. LlamaQ 300, WebQ 1000, TriviaQA 1000, HaluEval 1000; math
3,822 (AddSub 395, MultiArith 600, SinglEq 508, SVAMP 1000, GSM8K 1319).

| Model | LlamaQ ref. | LlamaQ resp. | WebQ ref. | WebQ resp. | TriviaQA ref. | TriviaQA resp. | HaluEval ref. | HaluEval resp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4-Voice† | | 64.7 | | 32.2 | | 39.1 | | 21.2 |
| STITCH-S† | | 73.3 | | 50.2 | | 50.0 | | – |
| MoshiRAG (Gemma 3)† | 83.0 | 80.3 | 71.5 | 67.2 | 73.7 | 69.6 | 42.0 | 36.3 |
| MoshiRAG (GPT-4.1)† | 87.8 | 80.6 | 77.7 | 68.9 | 86.8 | 78.2 | 61.2 | 51.3 |
| MoshiRAG (Tavily)† | 84.6 | 78.2 | 73.5 | 66.1 | 84.9 | 77.5 | 54.3 | 47.0 |
| Vanilla Moshi† | | 62.3 | | 26.6 | | 22.8 | | 10.5 |
| **Ours (Gemma 4)** | 85.3 | 81.7 | 67.7 | 59.1 | 71.5 | 68.9 | 39.6 | 33.7 |

| Model | AddSub ref. | AddSub resp. | MultiArith ref. | MultiArith resp. | SinglEq ref. | SinglEq resp. | SVAMP ref. | SVAMP resp. | GSM8K ref. | GSM8K resp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4-Voice† | | 59.4 | | 62.0 | | 71.0 | | 4.0 | | 29.0 |
| STITCH-S† | | 81.7 | | 87.9 | | 91.7 | | 72.2 | | 56.7 |
| MoshiRAG (Gemma 3)† | 76.6 | 61.7 | 87.1 | 69.0 | 83.2 | 68.2 | 74.1 | 55.0 | 66.2 | 33.9 |
| MoshiRAG (GPT-4.1)† | 87.9 | 64.8 | 87.1 | 76.0 | 89.6 | 72.9 | 80.5 | 61.1 | 70.8 | 43.2 |
| Vanilla Moshi† | | 8.3 | | 9.8 | | 18.4 | | 9.7 | | 2.1 |
| **Ours (Gemma 4)** | 78.5 | 76.2 | 94.5 | 89.7 | 82.1 | 77.6 | 86.0 | 81.1 | 70.4 | 62.5 |

## Quickstart

The weights load through the `contextspan` package; there is no `transformers` loader.

```bash
git clone https://github.com/mindlogic-ai/Context-Spanning && cd Context-Spanning
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
export HF_TOKEN=<token with access to nvidia/personaplex-7b-v1 and this repo>
```

Start the two backends (router LLM and ASR; two GPUs by default, one 96 GB GPU with the flags in the
repo README), then:

```bash
bash scripts/backends.sh start && source scripts/env.sh

# a wav through the model; out.wav is stereo, L = user, R = agent
python main.py infer --input-wav assets/test/question.wav --output-wav out.wav  \
    --output-text out.json --user-name Priya --user-city Sydney

# live conversation in the browser (microphone needs https or localhost)
python main.py serve --host 0.0.0.0 --port 8080 --token <value>
```

Any OpenAI-compatible server replaces the router and any `POST /transcribe -> {"text"}` endpoint replaces
the ASR. Servers, GPU layout, vLLM flags and environment variables:
[`docs/BACKENDS.md`](https://github.com/mindlogic-ai/Context-Spanning/blob/main/docs/BACKENDS.md).

## License

- Weights (this repository): NVIDIA Open Model License.
- Voice prompts: `main.pt` MIT; `f*`/`m*` CC0 (Kyutai Unmute Voice Donation).
- Code ([mindlogic-ai/Context-Spanning](https://github.com/mindlogic-ai/Context-Spanning)): MIT; `contextspan/moshi/` keeps Kyutai's own license files.

## Acknowledgements

Fine-tuned from [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1) (NVIDIA), which builds on
[Moshi](https://github.com/kyutai-labs/moshi) and the Mimi codec (Kyutai). The benchmark protocols follow MoshiRAG
and Full-Duplex-Bench.
