# Layout

Folders are grouped by role; files are named after the structure they implement.

```
main.py                         infer / serve / prepare / train
contextspan/
  model/                        the speech model
    weights.py                  base, checkpoint and voices: Hub or CS_BASE_DIR / CS_WEIGHTS_DIR
    engine.py                   Engine: persona prefix, step, <ret>, inject_context_span, clone_voice
    context_span_block.py       the Context Span block, read into the KV stream in one forward
    sequence_convention.py      token ids, placeholders, prefix, training-sequence assembly, transcript
  runtime/                      the live loop around the engine
    frame_stream.py             frame-clock loop for a wav: utterance ASR, <ret> handling, span injection
    websocket_server.py         the same loop over a WebSocket for the browser page
    web/                        index.html, mic.js, player.js
    user_leveller.py            user-channel denoise + loudness levelling to the training level
    default_persona.py          the persona the demo starts from
  training/
    prepare.py                  dialogues (JSON + stereo wav) -> .npz
    finetune.py                 span splicing, selective loss, training loop
  duetaspan/                    backend runtime (see its README)
  datasets/moshicp/             tool bank, SQLite world, geo index, TOOLS.md
  moshi/                        vendored PersonaPlex fork of Kyutai's moshi (third-party)
eval/                           benchmark harness (stack.py + rag/ fdb/ live/)
scripts/                        backends.sh, env.sh, prefill_timing.py
docs/                           BACKENDS.md, PROTOCOL.md, TRAINING.md, LAYOUT.md, demo/
```

## Rename map (previous layout -> this one)

| before | after |
|---|---|
| `contextspan/model.py` | `contextspan/model/weights.py` (loading) + `contextspan/model/engine.py` (`Engine`) |
| `contextspan/inject.py` | `contextspan/model/context_span_block.py` |
| `contextspan/spans.py` | `contextspan/model/sequence_convention.py` |
| `main.py: transcript()` | `contextspan/model/sequence_convention.py: transcript()` |
| `contextspan/stream.py` | `contextspan/runtime/frame_stream.py` |
| `contextspan/serve.py` | `contextspan/runtime/websocket_server.py` |
| `contextspan/levelling.py` | `contextspan/runtime/user_leveller.py` |
| `contextspan/personas.py` | `contextspan/runtime/default_persona.py` |
| `contextspan/web/` | `contextspan/runtime/web/` |
| `contextspan/web/PROTOCOL.md` | `docs/PROTOCOL.md` |
| `contextspan/train.py` | `contextspan/training/prepare.py` + `contextspan/training/finetune.py` |
| `eval/common.py` | `eval/stack.py` |
| `tools/prefill_timing.py`, `tools/prefill_passage.txt` | `scripts/` |

Import paths that stay valid: `from contextspan.model import Engine, load_model, load_voice`
(the `model` package re-exports them). Every other old module path was renamed; nothing else is aliased,
so a stale `contextspan.spans` / `contextspan.stream` import fails loudly instead of drifting.
`contextspan/duetaspan/` and `contextspan/datasets/` are unchanged: their module paths appear in
`scripts/backends.sh`, in the `DUETASPAN_*` environment contract and in external tooling.
