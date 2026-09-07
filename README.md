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
pip install --no-deps "git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi"
pip install -e .
export HF_TOKEN=<token with access to nvidia/personaplex-7b-v1 and the weights repo>
```

`CS_BASE_DIR` / `CS_WEIGHTS_DIR` point at local copies of the base speech-LM checkpoint and of the
weights repo (`context_spanning_7b.pt`, `voices/*.pt`) instead of downloading them.

## Backends

Two OpenAI-compatible HTTP servers; any implementation works (vLLM shown).

```bash
# router LLM
vllm serve google/gemma-3-27b-it --port 8000
export CS_LLM_URL=http://localhost:8000/v1/chat/completions CS_LLM_MODEL=google/gemma-3-27b-it
# ASR for the user question
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --port 8901 --served-model-name qwen3-asr
export CS_ASR_URL=http://localhost:8901/v1/audio/transcriptions CS_ASR_MODEL=qwen3-asr
```

A hosted API works the same way (`CS_LLM_URL=https://api.openai.com/v1/chat/completions`,
`CS_LLM_MODEL`, `CS_LLM_API_KEY`). `retrieve(question, context) -> str` in `contextspan/backend.py`
is the only contract: replace `LLMReferenceBackend` to change where the knowledge comes from.

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
each injected span as it lands; name, location and timezone go to the router as user context, the
**Knowledge** field goes to it as the Context DB. Stop offers the conversation as a stereo wav and a
JSON transcript. "Clone my voice" records 12 s and speaks with that voice.

## Training

```bash
python main.py prepare --in-dir data/raw --out-dir data/prepared
python main.py train   --data-dir data/prepared --out-dir runs/ft
```

`data/raw` holds one JSON per dialogue (turns with `speaker`, `text`, and for retrieval turns
`reference`) next to its stereo wav; `prepare` encodes them and splices the Context Span blocks at
sampled retrieval delays; `train` fine-tunes from the base speech-LM checkpoint (or `--checkpoint`).

## License

MIT — see `LICENSE`.
