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
    <a href="https://mindlogic.ai">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/figures/mindlogic_logo_white.png">
        <img width="170" alt="Mindlogic" src="assets/figures/mindlogic_logo_blue.png">
      </picture>
    </a>
  </p>

  <br>

  <!-- TODO(seonghyeon): replace XXXX.XXXXX with the arXiv id once the preprint is up -->
  [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-context--spanning--7b-yellow)](https://huggingface.co/mindlogicinc/context-spanning-7b) [![Base Model](https://img.shields.io/badge/base-PersonaPlex--7B-76b900)](https://huggingface.co/nvidia/personaplex-7b-v1) [![Code License](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE) [![Weights License](https://img.shields.io/badge/weights-NVIDIA%20Open%20Model%20License-76b900)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
  <!-- TODO(seonghyeon): add a Demo / Project Page badge here if one goes live -->

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

[mindlogicinc/context-spanning-7b](https://huggingface.co/mindlogicinc/context-spanning-7b) 
fine-tuned from [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)


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



## Backends

Two servers: the router LLM (OpenAI-compatible; default Gemma-4-26B-A4B on vLLM) and an ASR endpoint
(default Qwen3-ASR-1.7B on vLLM via `qwen-asr-serve`, OpenAI audio API; `ASR_BACKEND=transformers` starts
the plain `POST /transcribe` server instead). `scripts/backends.sh` starts both with the settings every
number here was measured with; `scripts/env.sh` exports the endpoints.

```bash
pip install -e '.[asr-server]' vllm
bash scripts/backends.sh start        # router on GPU 1, ASR on GPU 0; waits until both answer
source scripts/env.sh
```

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
MoshiRAG-sampled delays and fine-tunes with the masked cross-entropy. Data format and the v7 recipe:
[`docs/TRAINING.md`](docs/TRAINING.md).


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


## License

Code: MIT — see `LICENSE`. `contextspan/moshi/` carries its own license files. The released weights are
fine-tuned from `nvidia/personaplex-7b-v1` and are distributed under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
The voice prompts are built from CC0 voice recordings ([Kyutai Unmute Voice Donation](https://huggingface.co/kyutai/tts-voices), volunteers
who released their voice under CC0). See the [model card](https://huggingface.co/mindlogicinc/context-spanning-7b).
