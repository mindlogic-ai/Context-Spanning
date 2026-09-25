<div align="center">

  <h1>Context Spanning</h1>
  <h3>A Communication Framework for Full-Duplex Speech Models and External LLM Backends</h3>

  <br>

  <p>
    <b>Seonghyeon Go</b> &nbsp;•&nbsp; <b>Yongwoo Kim</b> &nbsp;•&nbsp; <b>Hyeonjin Cha</b> &nbsp;•&nbsp; <b>Jaeho Shin</b>
  </p>
  <p>
    <a href="https://mindlogic.ai">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/figures/mindlogic_logo_white.png">
        <img width="170" alt="Mindlogic" src="assets/figures/mindlogic_logo_blue.png">
      </picture>
    </a>
  </p>

  <br>

  [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-context--spanning--7b-yellow)](https://huggingface.co/mindlogicinc/context-spanning-7b)

</div>

<p align="center">
  <img src="assets/figures/architecture.png" alt="Context Spanning architecture: the frontend streams user audio, agent audio and agent text on one timeline; after the ret token the agent keeps talking (lead portion) while the backend passes the Streaming ASR user text and the Context DB to an LLM, whose tool result is written back as a Context Span (sine wave on user audio, silence on agent audio, sos … eos on agent text) before the body portion." width="100%">
</p>

Official PyTorch implementation of **Context Spanning**, a communication framework that lets a
full-duplex speech model call an external LLM backend (retrieval, tool calls, memory).

Code, the released **DuetaSpan-7B** weights, the tool bank and the benchmark harness are all here.

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
  <img src="assets/figures/context_span.png" alt="A Context Span on the token frame: five rows (user acoustic and semantic, agent acoustic and semantic, agent text); after the ret token the span holds sos … eos on the text row, a sine wave on both user rows and silence on both agent rows, and is read in a single KV prefill pass without taking a step index." width="90%">
</p>

A full-duplex speech model calls an external backend while it keeps listening and talking. When the
model emits `<ret>`, the recent user audio is transcribed, the backend returns reference, and that reference is written into the
model's context stream as a masked *Context Span* block at whatever frame it arrives. 


## Released Weights

[mindlogicinc/context-spanning-7b](https://huggingface.co/mindlogicinc/context-spanning-7b), fine-tuned from
[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1). `main.py infer` and `main.py serve`
fetch the weights and the released voices automatically; a local checkpoint loads with `--checkpoint`.

## Results

Full runs of the released checkpoint with the released router (Gemma-4-26B-A4B on vLLM) and ASR (Qwen3-ASR-1.7B);
protocols and scorers are in [`benchmark/README.md`](benchmark/README.md). Rows marked † are reprinted from the
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

## Backends

Two servers: the router LLM (OpenAI-compatible; default Gemma-4-26B-A4B on vLLM) and an ASR endpoint
(default Qwen3-ASR-1.7B on vLLM via `qwen-asr-serve`, OpenAI audio API; `ASR_BACKEND=transformers` starts
the plain `POST /transcribe` server instead). `scripts/backends.sh` starts both with the settings every
results were produced with; `scripts/env.sh` exports the endpoints.

```bash
pip install -e '.[asr-server]' vllm
bash scripts/backends.sh start        # router on GPU 1, ASR on GPU 0; waits until both answer
source scripts/env.sh
```

## Run

```bash
# a wav through the model; out.wav is stereo, L = user, R = agent
python main.py infer --input-wav assets/test/question.wav --output-wav out.wav  \
    --output-text out.json --user-name Priya --user-city Sydney

# live conversation in the browser (microphone needs https or localhost)
python main.py serve --host 0.0.0.0 --port 8080 --token <value>
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
MoshiRAG-sampled delays and fine-tunes on them. Data format and recipe:
[`docs/TRAINING.md`](docs/TRAINING.md).


## License

The code in this repository is released under the MIT License:

```
MIT License

Copyright (c) 2026 Mindlogic Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`contextspan/moshi/` is Kyutai's Moshi package and keeps its own license files. The released weights follow the NVIDIA Open Model License. Voice prompts: `main` MIT; `f*`/`m*` from CC0 recordings (Kyutai Unmute Voice Donation).
