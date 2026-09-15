# Live-session benchmark

Every other runner here streams a benchmark's own audio through the library. This one drives the
**browser page** with a fake microphone, one question per session, and scores what the page
rendered — so it measures what actually ships: the ASR, the router, the tool, the deadline, the
injected span, and what the model said with it. A number from here is a number a user could have
produced by talking to the demo.

```bash
python -m benchmark.live.build_clips benchmark/live/cases.json <clips_dir>     # TTS the questions (macOS `say`)
python -m benchmark.live.run http://localhost:8080 <clips_dir> runs/live  # one browser session per case
```

Needs `pip install playwright && playwright install chromium`, and a `python main.py serve` to talk
to. Sessions are sequential because the engine holds one conversation at a time.

## What it scores

| verdict | meaning |
| --- | --- |
| `span` | a span arrived and was injected — retrieval worked at all |
| `late` | it arrived after `RET_DEADLINE_S` and was dropped |
| `answer_ok` | the agent's words carry the expected fact |
| `grounded` | both: a span arrived **and** the answer is right |
| `invented_or_empty` | no span arrived and the answer is wrong or says nothing |
| `span_utilisation` | of the turns where a span did arrive, how many the answer used |

`span_utilisation` is the one specific to this system. Every other metric can be met by a model
answering from its own weights; this one asks whether the Context Span reached the words.

The agent **speaks** its answer and the gold is **written**, so matching is on normalised words with
spoken numerals folded to digits, then again with spaces removed — that is what makes "six fifty-one
P M" match `PM` and "eight thousand eight hundred forty eight" match `8,848`. Keep `expect` to the
one token that decides the answer (a name, a year), not a sentence.

## Cases

`cases.json` is `[{id, question, kind, expect: [...], not_expect: [...]}]`. `kind` is free text and
only groups the summary; the shipped set uses `live` (values a tool must fetch), `knowledge`,
`places` and `chitchat`. `not_expect` catches a confident wrong answer — for "capital of Australia"
it holds Sydney and Melbourne.
