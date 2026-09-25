# Layout

Folders are grouped by role; files are named after the structure they implement.

```
main.py                         infer / serve / prepare / train
contextspan/
  model/                        the speech model
    weights.py                  base and checkpoint: Hub or CS_BASE_DIR / CS_WEIGHTS_DIR; voices: Hub or CS_VOICES_DIR
    engine.py                   Engine: persona prefix, step, <ret>, inject_context_span, clone_voice
    context_span_block.py       the Context Span block, read into the KV stream in one forward
    lora.py                     LoRA checkpoints loaded unmerged (adapted Linears rebuilt as trained)
    prefix_prefill.py           the per-connection prefix (voice codes, persona text), read in one forward
    sequence_convention.py      token ids, placeholders, prefix, training-sequence assembly, transcript
  runtime/                      the live loop around the engine
    frame_stream.py             frame-clock loop for a wav: utterance ASR, <ret> handling, span injection
    websocket_server.py         the same loop over a WebSocket for the browser page
    web/                        index.html, mic.js, player.js
    user_leveller.py            user-channel denoise + loudness levelling to the training level
    default_persona.py          the persona the demo starts from
  training/
    prepare.py                  dialogues (JSON + stereo wav) -> .npz
    finetune.py                 span splicing, training loop
  duetaspan/                    backend runtime (see its README)
  datasets/moshicp/             tool bank, SQLite world, geo index, TOOLS.md
  moshi/                        vendored PersonaPlex fork of Kyutai's moshi (third-party)
benchmark/                      benchmark harness: stack.py, run_full.sh, rag/ fdb/ live/ math/
scripts/                        backends.sh, env.sh, prefill_timing.py, prefill_passage.txt
assets/                         figures/ (README), test/question.wav (infer), example/ (a `prepare` input pair)
docs/                           BACKENDS.md, PROTOCOL.md, TRAINING.md, LAYOUT.md, MODEL_CARD.md
```

`from contextspan.model import Engine, load_model, load_voice` is the public import path (the `model`
package re-exports them). The module paths under `contextspan/duetaspan/` and `contextspan/datasets/`
appear in `scripts/backends.sh` and in the `DUETASPAN_*` environment contract.
