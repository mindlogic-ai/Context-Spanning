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

Needs `pip install -e '.[benchmark]' && playwright install chromium`, and a `python main.py serve` to
talk to. Sessions are sequential because the engine holds one conversation at a time.

## What it scores

Per case (`results.json`):

| field | meaning |
| --- | --- |
| `span` | a span arrived and was injected: retrieval worked at all |
| `late` | it arrived after `CS_RET_DEADLINE_S` and was dropped |
| `answer_ok` | the agent's words carry the expected fact |
| `contradicted` | a span arrived and the agent said one of the case's `not_expect` values |
| `retrieval_s` | seconds from `<ret>` to the span |

Over the set (`summary.json`):

| key | meaning |
| --- | --- |
| `span_rate`, `late_rate` | share of cases with a span, and with a span dropped as late |
| `answer_accuracy` | share of cases answered correctly |
| `grounded` | a span arrived **and** the answer is right |
| `invented` | no span arrived, the agent spoke anyway, and the answer is wrong |
| `contradicted_span` | the answer contradicts the span that was injected |
| `retrieval_s_p50`, `retrieval_s_p90` | retrieval latency |
| `by_kind` | `span_rate` and `answer_accuracy` per case kind |

`grounded` and `invented` are the two specific to this system. Accuracy alone can be met by a model
answering from its own weights; these ask whether the Context Span reached the words, and what the model
does when no span came.

The agent **speaks** its answer and the gold is **written**, so matching is on normalised words with
spoken numerals folded to digits, then again with spaces removed — that is what makes "six fifty-one
P M" match `PM` and "eight thousand eight hundred forty eight" match `8,848`. Keep `expect` to the
one token that decides the answer (a name, a year), not a sentence.

## Cases

`cases.json` is `[{id, question, kind, expect: [...], not_expect: [...]}]`. `kind` is free text and
only groups the summary; the shipped set uses `live` (values a tool must fetch), `knowledge`,
`places` and `chitchat`. `not_expect` catches a confident wrong answer — for "capital of Australia"
it holds Sydney and Melbourne.
