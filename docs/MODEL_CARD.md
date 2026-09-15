---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/personaplex-7b-v1
pipeline_tag: audio-to-audio
language:
  - en
library_name: contextspan
# TODO(jaeho): thumbnail: <absolute URL of assets/figures/architecture.png once the repo is public>
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
> This repository holds the **DuetaSpan v7 (step 8000)** weights described in
> *Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends*
> (Go, Kim, Cha, Shin — Mindlogic, 2026). The inference and training code lives at
> [mindlogic-ai/ContextSpanning](https://github.com/mindlogic-ai/ContextSpanning); the weights load only
> through that package.

<!-- TODO(seonghyeon): replace XXXX.XXXXX with the arXiv id -->
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![Code](https://img.shields.io/badge/GitHub-ContextSpanning-181717?logo=github)](https://github.com/mindlogic-ai/ContextSpanning) [![Base Model](https://img.shields.io/badge/base-PersonaPlex--7B-76b900)](https://huggingface.co/nvidia/personaplex-7b-v1) [![Weights License](https://img.shields.io/badge/weights-NVIDIA%20Open%20Model%20License-76b900)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) [![Code License](https://img.shields.io/badge/code-MIT-blue.svg)](https://github.com/mindlogic-ai/ContextSpanning/blob/main/LICENSE)

A full-duplex speech model that keeps listening and talking while it calls an external backend. When the
model emits `<ret>`, the recent user audio is transcribed, a tool router (with an LLM for knowledge
questions) returns one reference, and that reference is written into the model's own context stream
**as-is** as a *Context Span* block. The frame clock never waits, and the model reads exact values
(time, weather, prices, distances) instead of a latent summary.

<p align="center">
  <img src="assets/figures/architecture.png" alt="Context Spanning architecture: the full-duplex frontend streams user speech, agent speech and agent text on one timeline; on the ret token the backend runs ASR, an LLM over the Context Memory DB and a tool, and the result is written back into the stream as a Context Span." width="100%">
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
- Sequence convention: `<ret>` = 4, span open = 12, span close = 13 (`CS_SPAN_CLOSE_ID`; checkpoints trained before 2026-09-08 used 12 on both sides), text pad = 3
- Training data: `manifest_v6h` — 823,659 dialogues, ~10,009 h indexed; every dialogue passed a frame-level audio QA and a full-text QA
- Training objective: masked cross-entropy on the text row and on the audio codebooks as in PersonaPlex, with the span and prefix columns masked out
- Optimisation: 8-bit AdamW, context 3,000 frames, lr 2e-6 (temporal transformer) / 4e-6 (depth transformer), 32 dialogues per update, 4× NVIDIA RTX Pro 6000
- Released checkpoint: DuetaSpan v7, step 8000 (2026-09-11)
- Backends the numbers were measured with: router `google/gemma-4-26B-A4B-it` on vLLM, ASR Qwen3-ASR-1.7B
- Language: English

## Benchmark Results

Numbers as reported in the paper (Tables 1 and 2). Rows marked † are reprinted from the original benchmark papers
(Full-Duplex-Bench) or from MoshiRAG (spoken QA and math). Arrows give the direction of better.

<!-- TODO(seonghyeon): confirm these are the camera-ready numbers -->

<div style="max-width:960px;margin:0 auto">
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead>
<tr>
<th style="padding:10px 8px;text-align:left;border-bottom:2px solid #1F65FF;color:#1F65FF">Full-Duplex-Bench</th>
<th style="padding:10px 8px;text-align:left;border-bottom:2px solid #1F65FF;color:#1F65FF">Metric</th>
<th style="padding:10px 8px;text-align:center;border-bottom:2px solid #1F65FF;color:#1F65FF;background:rgba(31,101,255,.08)">Ours</th>
<th style="padding:10px 8px;text-align:center;border-bottom:2px solid #1F65FF;color:#1F65FF">PersonaPlex†</th>
<th style="padding:10px 8px;text-align:center;border-bottom:2px solid #1F65FF;color:#1F65FF">Moshi†</th>
<th style="padding:10px 8px;text-align:center;border-bottom:2px solid #1F65FF;color:#1F65FF">Gemini†</th>
</tr>
</thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#1F65FF;background:rgba(31,101,255,.10);border-bottom:1px solid rgba(31,101,255,.2)">v1 — turn-taking</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Pause Handling</td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">TOR (candor) ↓</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.866</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.431</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.980</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.310</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">TOR (synthetic) ↓</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.956</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.358</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.985</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.255</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Backchannel</td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">TOR ↓</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.673</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.273</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">1.000</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.091</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Freq ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)"><strong>0.184</strong></td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.042</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.001</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.012</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">JSD ↓</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.700</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.662</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.957</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.896</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Smooth Turn-Taking</td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">TOR ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)"><strong>1.000</strong></td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.908</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.941</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.655</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Latency ↓</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)"><strong>0.145</strong></td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.170</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.265</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">1.301</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">User Interruption</td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">TOR ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.970</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.950</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">1.000</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.891</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">GPT-4o score ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">4.17</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">4.290</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.765</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">3.376</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Latency ↓</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.771</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.240</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.257</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">1.183</td></tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#1F65FF;background:rgba(31,101,255,.10);border-bottom:1px solid rgba(31,101,255,.2)">v3 — tool calling under disfluency</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Tool Use</td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Pass@1 ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)"><strong>0.56</strong></td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.540</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Tool Selection ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)"><strong>0.910</strong></td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.817</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Arg. Accuracy ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)"><strong>0.653</strong></td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.588</td></tr>
<tr><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)"></td><td style="padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15)">Resp. Quality ↑</td><td style="padding:7px 8px;text-align:center;background:rgba(31,101,255,.06);border-bottom:1px solid rgba(128,128,128,.15)">0.242</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">–</td><td style="padding:7px 8px;text-align:center;border-bottom:1px solid rgba(128,128,128,.15)">0.718</td></tr>
</tbody>
</table>
</div>

**Spoken QA and math reasoning (accuracy, %).** `ref.` = the reference document is provided; `resp.` = the
model's response. Our router is Gemma 4.

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

The step-8000 checkpoint in this repository was measured again after the paper runs on the `semi`
protocol (120 items per set): HaluEvalAudio resp 0.642 / ref 0.725 / P(resp | ref) 0.851, math word
problems (40) P(resp | ref) 1.00, `<ret>` rate 0.925. Protocols and scorers:
[`benchmark/README.md`](https://github.com/mindlogic-ai/ContextSpanning/blob/main/benchmark/README.md).

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
<!-- TODO(seonghyeon): decide gated: true / extra_gated_prompt before the repo goes public -->

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

Fine-tuned from [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1) (NVIDIA), which builds
on [Moshi](https://github.com/kyutai-labs/moshi) and the Mimi codec (Kyutai). Retrieval pipeline, data
generation and the RAG-suite protocol follow MoshiRAG (Chien et al., 2026).
