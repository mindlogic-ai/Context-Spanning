# Training

```bash
python main.py prepare --in-dir data/raw --out-dir data/prepared
python main.py train   --data-dir data/prepared --out-dir runs/ft [--checkpoint ckpt.pt] [--steps N] [--lr 2e-6]
```

## Data format (`prepare`)

One JSON per dialogue next to a stereo wav with L = user, R = agent, at any sample rate:

```json
{"system_prompt": "You are Kelina, ...", "voice": "voices/f0.pt",
 "turns": [{"speaker": "user",  "words": [{"w": "when", "t": 4.21}, {"w": "does", "t": 4.40}, ...]},
           {"speaker": "agent", "words": [{"w": "it", "t": 6.02}, ...],
            "ret": true, "reference": "The cafe opens at seven thirty AM.", "body_word_index": 3}]}
```

Word times are seconds. An agent turn with `ret` and `reference` is a grounded turn: `<ret>` is placed on
the frame before its first word and the reference becomes its Context Span. `body_word_index` (default:
a third of the turn) marks where the lead-in ends; the span must have arrived by then, so the retrieval
delay is sampled between `<ret>` and that frame. `prepare` encodes both channels with Mimi (12.5 Hz, 8
codebooks each), writes the text row from the word times, and saves one `.npz` per dialogue.

A complete pair to try it on is in `assets/example/`:
`python main.py prepare --in-dir assets/example --out-dir data/prepared`.

## What `train` does

Every step draws `--accum` dialogues (`contextspan/training/finetune.py`):

1. the Context Span block (`[span] reference tokens [span]`, SILENCE/SINE placeholder audio) is spliced
   at a delay after `<ret>` sampled as in MoshiRAG Eq. 3 (`sequence_convention.sample_rag_delay`); the
   acoustic codebooks of the frame before the block MOVE to its last column, so real audio is read exactly
   once around it (`assemble_training_sequence`);
2. the masked PersonaPlex prefix (voice codes, `<system> persona <system>`) is prepended
   (`persona_prefix`);
3. a `--context`-frame window is cut and the model is fine-tuned on its text row and audio codebooks; the
   prefix and span columns are masked.

Checkpoints are written every `--ckpt-every` steps as `{"model": state_dict}` and load with
`python main.py infer --checkpoint`.

## The released weights (Context Spanning-7B, step 5,938)

Every parameter of `nvidia/personaplex-7b-v1` fine-tuned with this sequence convention on a 2,100-hour stereo
corpus of 195,257 synthetic dialogues (38.2 s and 9.9 spoken turns on average): everyday conversation
(1,453 h / 102k dialogues; SODA, with turn-taking and backchannels placed as in the Candor corpus), knowledge
retrieval (389 h / 71k; Natural Questions, HotpotQA and TriviaQA through the MoshiRAG pipeline), tool use
(195 h / 12k; 125 tools, 87 from MCPToolBench++ and 38 from Google SGD, the 65 MCPToolBench++ tasks unsuited to a
voice assistant excluded) and other scenarios such as abstaining (63 h / 10k). Dialogues with retrieval carry
1.53 `<ret>` on average. Scripts were written by Gemma 4 31B and every span content by Gemma-4-26B-A4B, the
backend model; the speech is Fish Audio with 5,164 single-speaker GLOBE prompts kept above a UTMOSv2 score of
3.2. Numbers are fully verbalised with the NeMo text normalizer before tokenization (Moshi's digit splitting
misaligns frames on large numbers). Every dialogue passed a frame-level audio QA (alignment, levels, silence)
and a full-text QA that removed fabricated lookups without `<ret>`. Persona prefixes carry the user's name and
city in the phrasing `persona_text` produces.

AdamW, context 3,000 frames, learning rate 2e-6 for the temporal transformer and 4e-6 for the depth transformer,
one epoch on two NVIDIA RTX Pro 6000 (8 hours). The convention, the placeholders and the delay sampler in this
repository are the ones the corpus was written with, so a checkpoint produced by `main.py train` on new data is
directly comparable.
