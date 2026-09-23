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

# Context Spanning-7B

> [!Note]
> This repository holds the **Context Spanning-7B** weights (training step 5,938) described in
> *Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends*
> (Go, Kim, Cha, Shin — Mindlogic, 2026). The inference and training code lives at
> [mindlogic-ai/ContextSpanning](https://github.com/mindlogic-ai/ContextSpanning); the weights load only
> through that package.

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![Code](https://img.shields.io/badge/GitHub-ContextSpanning-181717?logo=github)](https://github.com/mindlogic-ai/ContextSpanning) [![Base Model](https://img.shields.io/badge/base-PersonaPlex--7B-76b900)](https://huggingface.co/nvidia/personaplex-7b-v1) [![Weights License](https://img.shields.io/badge/weights-NVIDIA%20Open%20Model%20License-76b900)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) [![Code License](https://img.shields.io/badge/code-MIT-blue.svg)](https://github.com/mindlogic-ai/ContextSpanning/blob/main/LICENSE)

A full-duplex speech model that keeps listening and talking while it calls an external backend. When the
model emits `<ret>`, the recent user audio is transcribed, a tool router (with an LLM for knowledge
questions) returns one reference, and that reference is written into the model's own context stream
**as-is** as a *Context Span* block. The frame clock never waits, and the model reads exact values
(time, weather, prices, distances) instead of a latent summary.

<p align="center">
  <img src="assets/figures/architecture.png" alt="Context Spanning architecture: the frontend streams user audio, agent audio and agent text on one timeline; after the ret token the agent keeps talking (lead portion) while the backend passes the Streaming ASR user text and the Context DB to an LLM, whose tool result is written back as a Context Span (sine wave on user audio, silence on agent audio, sos … eos on agent text) before the body portion." width="100%">
</p>

<!-- assets/figures/architecture.png is uploaded to this model repo together with the card -->

## Highlights

- **One primitive for retrieval, tools and memory.** Everything the backend returns comes back through the
  same `<sos> … <eos>` span in the text row, so the model verbalises precise values without information loss.
- **No stall.** The span is prefilled in a single KV pass after the acoustic delay; the model only updates
  its cache and keeps stepping at the 80 ms frame clock.
- **Real tools.** The shipped bank is 60 tools whose result is something a voice assistant says — time,
  weather, prices, web search, places and routes, SGD-seeded bookings — plus four MCP servers.
- **Nine released voices** (`f0`-`f3` female, `m0`-`m3` male, plus `seonghyeon`, a team member's own recorded voice) and PersonaPlex-style persona prompts carrying
  the user's name and city.

## Files

| file | content |
|---|---|
| `context_spanning_7b.pt` | `{"model": state_dict}` in bf16, loadable with `main.py infer --checkpoint` or fetched automatically |
| `voices/{f0,f1,f2,f3,m0,m1,m2,m3,seonghyeon}.pt` | voice prompts (`{"codes": LongTensor[8, P]}`, agent-voice Mimi codes; 8 s each) |
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
- Training data: a 2,100-hour stereo corpus of 195,257 synthetic dialogues (38.2 s and 9.9 spoken turns on average): everyday conversation 1,453 h / 102k (SODA, turn-taking and backchannels placed with the Candor corpus), knowledge retrieval 389 h / 71k (Natural Questions, HotpotQA, TriviaQA through the MoshiRAG pipeline), tool use 195 h / 12k (125 tools: 87 from MCPToolBench++, 38 from Google SGD), other scenarios such as abstaining 63 h / 10k. Scripts by Gemma 4 31B, span contents by Gemma-4-26B-A4B, speech by Fish Audio with 5,164 GLOBE speaker prompts (UTMOSv2 > 3.2), numbers verbalised with NeMo text normalization
- Optimisation: every parameter of PersonaPlex-7B, AdamW, context 3,000 frames, lr 2e-6 (temporal transformer) / 4e-6 (depth transformer), one epoch on two NVIDIA RTX Pro 6000 (8 h); recipe in [`docs/TRAINING.md`](https://github.com/mindlogic-ai/ContextSpanning/blob/main/docs/TRAINING.md)
- Released checkpoint: training step 5,938 (2026-09-23)
- Backends: router `google/gemma-4-26B-A4B-it` on vLLM (the default; the spoken-QA table also reports GPT-4.1 as the reference LLM), ASR Qwen3-ASR-1.7B
- Language: English

## Benchmark Results

The tables of the paper. Rows marked † are reprinted from the cited papers (MoshiRAG, PersonaPlex,
Full-Duplex-Bench); everything else was measured with this checkpoint and the benchmark harness of the code
repository. Unless a row says otherwise the backend is Gemma-4-26B-A4B.

**Spoken QA and math reasoning (accuracy, %).** `ref.` = the injected reference judged against the gold
answer; `resp.` = the model's response. Following MoshiRAG's API-backend protocol, the pre-computed reference is
injected a fixed delay after `<ret>`: GPT-4.1 answered in 0.77 s on average in our runs, so the GPT-4.1 row uses a
0.8 s delay (MoshiRAG used 1.5 s). The math sets were unseen during training. Every set whole (LlamaQ 300,
WebQ 1,000, TriviaQA 1,000, HaluEval 1,000, math 3,822); MoshiRAG's judges (gemma-3-27b-it for HaluEval and math,
gpt-4o for the OpenAudioBench sets). Protocol and per-set reports: [`benchmark/README.md`](https://github.com/mindlogic-ai/ContextSpanning/blob/main/benchmark/README.md),
[`benchmark/results/`](https://github.com/mindlogic-ai/ContextSpanning/tree/main/benchmark/results).

| Model | LlamaQ ref. | LlamaQ resp. | WebQ ref. | WebQ resp. | TriviaQA ref. | TriviaQA resp. | HaluEval ref. | HaluEval resp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4-Voice† | | 64.7 | | 32.2 | | 39.1 | | 21.2 |
| STITCH-S† | | 73.3 | | 50.2 | | 50.0 | | – |
| MoshiRAG (Gemma 3)† | 83.0 | 80.3 | 71.5 | 67.2 | 73.7 | 69.6 | 42.0 | 36.3 |
| MoshiRAG (GPT-4.1)† | 87.8 | 80.6 | 77.7 | **68.9** | 86.8 | 78.2 | 61.2 | 51.3 |
| MoshiRAG (Tavily)† | 84.6 | 78.2 | 73.5 | 66.1 | 84.9 | 77.5 | 54.3 | 47.0 |
| Vanilla Moshi† | | 62.3 | | 26.6 | | 22.8 | | 10.5 |
| **Ours (Gemma 4)** | 85.3 | 81.7 | 67.7 | 59.1 | 71.5 | 68.9 | 39.6 | 33.7 |
| **Ours (GPT-4.1)** | 89.9 | **83.3** | 79.4 | 66.7 | 91.1 | **83.8** | 68.3 | **55.5** |

| Model | AddSub ref. | AddSub resp. | MultiArith ref. | MultiArith resp. | SinglEq ref. | SinglEq resp. | SVAMP ref. | SVAMP resp. | GSM8K ref. | GSM8K resp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-4-Voice† | | 59.4 | | 62.0 | | 71.0 | | 4.0 | | 29.0 |
| STITCH-S† | | **81.7** | | 87.9 | | **91.7** | | 72.2 | | 56.7 |
| MoshiRAG (Gemma 3)† | 76.6 | 61.7 | 87.1 | 69.0 | 83.2 | 68.2 | 74.1 | 55.0 | 66.2 | 33.9 |
| MoshiRAG (GPT-4.1)† | 87.9 | 64.8 | 87.1 | 76.0 | 89.6 | 72.9 | 80.5 | 61.1 | 70.8 | 43.2 |
| Vanilla Moshi† | | 8.3 | | 9.8 | | 18.4 | | 9.7 | | 2.1 |
| **Ours (Gemma 4)** | 78.5 | 76.2 | 94.5 | **89.7** | 82.1 | 77.6 | 86.0 | 81.1 | 70.4 | 62.5 |
| **Ours (GPT-4.1)** | 82.6 | 74.7 | 95.2 | **89.7** | 85.9 | 82.2 | 89.2 | **85.4** | 75.7 | **67.4** |

**Full-Duplex-Bench v1.** TOR = take-over rate (the fraction of clips in which the model takes the turn);
backchannel frequency is backchannels per second, JSD the Jensen-Shannon divergence from the human backchannel
timing distribution; latencies in seconds; the interruption response is rated by GPT-4o. Rows marked † are
reprinted from the PersonaPlex paper; Gemini is Gemini Live 2.5.

| Task | Metric | **Ours** | PersonaPlex† | Moshi† | Gemini† |
|---|---|---:|---:|---:|---:|
| Turn-Taking | TOR ↑ | 0.899 | 0.992 | 0.941 | 0.655 |
| | Latency, s ↓ | 0.064 | 0.070 | 0.265 | 1.301 |
| Pause Handling | TOR, Candor ↓ | 0.824 | 0.662 | 0.980 | 0.310 |
| | TOR, synthetic ↓ | 0.861 | 0.584 | 0.985 | 0.255 |
| Backchannel | TOR ↓ | 0.782 | 0.327 | 1.000 | 0.091 |
| | Frequency, per s ↑ | 0.150 | 0.025 | 0.001 | 0.012 |
| | JSD ↓ | 0.716 | 0.649 | 0.957 | 0.896 |
| User Interruption | TOR ↑ | 0.925 | 1.000 | 1.000 | 0.891 |
| | GPT-4o rating, 0–5 ↑ | 3.924 | 4.210 | 0.765 | 3.376 |
| | Latency, s ↓ | 0.598 | 0.400 | 0.257 | 1.183 |

**Full-Duplex-Bench v3 (tool calling under disfluency).** MoshiRAG is the released MoshiRAG checkpoint with the
Gemma-3-27B reference LLM under the same tool router and prompt. Tool selection, argument accuracy, response quality
and pass rate are fractions of the 100 scenarios; take-turn, interruption and filler rates are percentages; latency
is the task-completion time in seconds. Rows marked † are reprinted from the benchmark paper; GPT is GPT-Realtime,
Gemini is Gemini Live 3.1.

| Task | Metric | **Ours** | MoshiRAG | GPT† | Gemini† |
|---|---|---:|---:|---:|---:|
| Tool Use | Tool selection ↑ | 0.855 | 0.738 | 0.876 | 0.817 |
| | Argument accuracy ↑ | 0.567 | 0.440 | 0.680 | 0.588 |
| | Response quality ↑ | 0.411 | 0.255 | 0.792 | 0.718 |
| | Pass rate ↑ | 0.470 | 0.280 | 0.600 | 0.540 |
| Turn-Taking Dynamics | Take-turn rate, % ↑ | 95.0 | 94.0 | 96.0 | 78.0 |
| | Latency, s ↓ | 5.83 | 7.69 | 6.89 | 4.25 |
| | Interruption rate, % ↓ | 73.7 | 58.5 | 13.5 | 19.2 |
| | Filler rate, % ↓ | 96.0 | 92.3 | 16.9 | 31.7 |

**Context Span processing latency.** Span prefill plus one decoding step, which must fit the 80 ms frame budget
(Mimi runs at 12.5 Hz). `n` = span length in frames; `n = 0` is a plain step, as in vanilla Moshi. Latency grows
sub-linearly with the span length.

| n (frames) | 0 | 16 | 64 | 256 | 600 |
|---|---:|---:|---:|---:|---:|
| Latency, ms (mean ± sd) | 34.7 ± 0.4 | 56.1 ± 0.2 | 57.5 ± 0.2 | 62.0 ± 0.3 | 86.1 ± 0.3 |
| P99, ms | 35.2 | 56.3 | 57.7 | 62.3 | 86.3 |

## Quickstart

The weights load through the `contextspan` package; there is no `transformers` loader.

```bash
git clone https://github.com/mindlogic-ai/ContextSpanning && cd ContextSpanning
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
export HF_TOKEN=<token with access to nvidia/personaplex-7b-v1 and this repo>
```

Start the two backends (router LLM and ASR; two GPUs by default, one 96 GB GPU with the flags in the
repo README), then:

```bash
bash scripts/backends.sh start && source scripts/env.sh

# a wav through the model; out.wav is stereo, L = user, R = agent
python main.py infer --input-wav assets/test/question.wav --output-wav out.wav --voice f0 \
    --output-text out.json --user-name Priya --user-city Sydney

# live conversation in the browser (microphone needs https or localhost)
python main.py serve --voice f0 --host 0.0.0.0 --port 8080 --token <value>
```

Any OpenAI-compatible server replaces the router and any `POST /transcribe -> {"text"}` endpoint replaces
the ASR. Servers, GPU layout, vLLM flags and environment variables:
[`docs/BACKENDS.md`](https://github.com/mindlogic-ai/ContextSpanning/blob/main/docs/BACKENDS.md).

## Intended Use and Limitations

- English, spoken conversation with a voice assistant persona. Not trained for other languages.
- The model's factual accuracy is bounded by the backend: it says what the span says. Run it with a
  router and tools you trust.
- Context Spans occupy one text token per frame, so long references spend context; see the paper's
  limitations section.
- Out of scope: text-only use, batch transcription, or any use of the voice prompts to imitate a real
  person. The `f`/`m` voices come from volunteers who released their recordings under CC0 and `seonghyeon` is a team member's own voice, released with consent; do not condition on a real person's voice without
  their consent.

## License

- **Weights (this repository):** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
  inherited from the PersonaPlex weights they are fine-tuned from. (PersonaPlex was initialised from
  `kyutai/moshiko-pytorch-bf16`, CC-BY-4.0; that attribution is carried, it is not the license of this model.)
- **Voice prompts** (`voices/*.pt`) are built from CC0 recordings of the [Kyutai Unmute Voice Donation](https://huggingface.co/kyutai/tts-voices)
  project (volunteers who released their voice under CC0; no attribution or consent condition attaches).
- **Code:** [mindlogic-ai/ContextSpanning](https://github.com/mindlogic-ai/ContextSpanning) is MIT; the
  vendored `contextspan/moshi/` carries Kyutai's own license files.

## Citation

```bibtex
@article{go2026contextspanning,
  title   = {Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends},
  author  = {Go, Seonghyeon and Kim, Yongwoo and Cha, Hyeonjin and Shin, Jaeho},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Acknowledgements

Fine-tuned from [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1) (NVIDIA), which builds
on [Moshi](https://github.com/kyutai-labs/moshi) and the Mimi codec (Kyutai). Retrieval pipeline, data
generation and the RAG-suite protocol follow MoshiRAG (Chien et al., 2026).
