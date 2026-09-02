# Context Spanning

Official code for **Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends**.

Context Spanning lets a full-duplex speech model ([PersonaPlex](https://github.com/NVIDIA/personaplex) / Moshi architecture) call an external backend while it keeps listening and talking: the model emits a `<ret>` token when it needs outside knowledge, the backend answers, and the answer is written into the model's context stream as a masked *Context Span* at whatever frame it arrives. Training and inference share one sequence convention, so anything an LLM, a tool, or a search engine returns can be spoken grounded, in real time.

<p align="center"><img src="assets/architecture.png" alt="Context Spanning overview" width="720"></p>

Weights: [mindlogicinc/context-spanning-7b](https://huggingface.co/mindlogicinc/context-spanning-7b) (fine-tuned from `nvidia/personaplex-7b-v1`).

## Install

```bash
pip install "git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi"   # PersonaPlex moshi fork
pip install -e .
export HF_TOKEN=<token with access to nvidia/personaplex-7b-v1 and the weights repo>
```

## Backends

The model talks to two OpenAI-compatible HTTP servers; any implementation works (vLLM shown).

```bash
# reference LLM (writes a <=20-word spoken reference for the transcribed question)
vllm serve google/gemma-3-27b-it --port 8000
export CS_LLM_URL=http://localhost:8000/v1/chat/completions CS_LLM_MODEL=google/gemma-3-27b-it
# ASR for the user question
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --port 8901 --served-model-name qwen3-asr
export CS_ASR_URL=http://localhost:8901/v1/audio/transcriptions CS_ASR_MODEL=qwen3-asr
```

A hosted API works the same way — set `CS_LLM_URL=https://api.openai.com/v1/chat/completions`, `CS_LLM_MODEL=<model>`, `CS_LLM_API_KEY=<key>` (reasoning models such as GPT-5 / Luna are handled automatically).

`retrieve(question) -> str` in `contextspan/backend.py` is the only contract: replace `LLMReferenceBackend` with a search engine, a tool router, or a database to change where the knowledge comes from.

## Inference

```bash
python main.py infer --input-wav assets/test/question.wav --output-wav out.wav \
    --voice assets/voices/f0.pt --text-prompt "You are a helpful and friendly voice assistant."
```

`out.wav` is stereo: left = user, right = agent. The transcript (with `<ret>` / `<span>` markers) is printed and, with `--output-text`, saved with the retrieval events (question as transcribed, reference, arrival time).

Streaming runs at a 1.0x frame clock (12.5 Hz). After `<ret>` the loop waits for the user to finish (`--debounce`, seconds of trailing silence), transcribes the utterance, asks the backend and injects the reference on the next frame. If the user keeps talking within `--reroute` seconds, the utterance is transcribed again and the span is replaced.

## Training

Each dialogue is a JSON file plus a stereo wav (left = user, right = agent):

```json
{"system_prompt": "You are a warm bookshop owner.", "voice": "assets/voices/f0.pt",
 "turns": [
  {"speaker": "agent", "words": [{"w": "Hi", "t": 0.4}, {"w": "there.", "t": 0.7}]},
  {"speaker": "user",  "words": [{"w": "When", "t": 4.1}, {"w": "does", "t": 4.3}, {"w": "it", "t": 4.5}, {"w": "open?", "t": 4.7}]},
  {"speaker": "agent", "ret": true, "reference": "The shop opens at seven thirty AM.",
   "words": [{"w": "Oh,", "t": 5.6}, {"w": "it", "t": 5.9}, {"w": "opens", "t": 6.1}, {"w": "at", "t": 6.3}, {"w": "seven", "t": 6.5}, {"w": "thirty.", "t": 6.8}],
   "body_word_index": 2}
 ]}
```

`ret` marks an agent turn that needs the reference; `body_word_index` is the first word that states retrieved content, so the span is always inserted before it (a random delay after `<ret>`, following the retrieval-delay sampling of MoshiRAG).

```bash
python main.py prepare --in-dir data/raw --out-dir data/prepared
python main.py train --data-dir data/prepared --out-dir runs/ft --steps 1000 --checkpoint context_spanning_7b.pt
```

`prepare` encodes both channels with Mimi and lays the agent text on the text channel at word times. `train` fine-tunes all parameters (lr 2e-6, effective batch = `--accum`, bf16), inserting spans as masked blocks at a freshly sampled delay every step and applying reference dropout (0.2); checkpoints are written every `--ckpt-every` steps as `{"model": state_dict}`, loadable by `infer --checkpoint`.

## Voice prompts

A voice prompt is a short clip of the agent voice encoded with Mimi and saved as `{"codes": LongTensor[8, P]}`; `assets/voices/` ships a few. Training and inference both prepend `voice -> silence -> text prompt -> silence` as a masked prefix.

## License

Code: MIT. Model weights inherit the PersonaPlex model license.
