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
| `contextspan/tools.py` | the tool bank: the values the router cannot know, fetched and rendered as one span sentence |
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

### Tools

The router prompt forbids answering the current time, the current weather or a live value from the
model's own knowledge: those must come from a tool. `contextspan/tools.py` is that catalog. Its
schemas are put in front of the router, a routed call is executed, and the result comes back as one
sentence in the same shape a reference has, because that is what is spliced into the stream.

| tool | provider | needs a key |
|---|---|---|
| `get_time` | `zoneinfo`, local | no network at all |
| `get_weather` | `CS_WEATHER_URL` (default wttr.in) | no |
| `find_places` | `CS_PLACES_URL` (default OpenStreetMap Nominatim) | no |
| `web_search` | `CS_SEARCH_URL` (default Wikipedia) | no |

A tool whose provider is set to an empty string is dropped from the catalog rather than offered and
then failing, so the router can only pick something that can actually run; `CS_TOOLS=get_time,get_weather`
restricts it further. `CS_TOOL_TIMEOUT` (default 6 s) is a hard ceiling, because a span that misses
the response delay is worse than no span. A tool never raises: a dead endpoint means no span and the
conversation continues.

Missing arguments are filled from the user profile the page supplies, so "what time is it" and
"somewhere to eat near me" work without the user naming a city. `find_places` only returns names the
model can pronounce (`CS_PLACES_LANG`, default `en`) and declines otherwise, which means that in
cities where OpenStreetMap carries no English names it will find nothing — point `CS_PLACES_URL` at a
provider with better local coverage if that matters.

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
JSON transcript (`ret` / `question` / `span` / `text` events with timing; `question` is what the ASR heard, so a wrong span can be traced to the ASR or to the backend). "Clone my voice" records 12 s and speaks with that voice.

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
