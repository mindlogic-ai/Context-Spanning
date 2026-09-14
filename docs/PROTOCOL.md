# Browser <-> server protocol (`/ws`)

Server: `contextspan/runtime/websocket_server.py`; page: `contextspan/runtime/web/`.
One WebSocket per conversation; the server holds one engine, so a second connection gets `busy`.
Binary frames are float32 PCM at the sample rate given in `ready`, one frame (`frame_size` samples,
80 ms) per message in both directions. Text frames are JSON objects with a `type`.

## Client → server

| message | when | effect |
| --- | --- | --- |
| `{"type":"context", name, location, lat, lon, tz, persona, db}` | before the first frame (and whenever a field changes) | Context DB profile (`db` = notes) and the model's prefix (see below); a change to `persona`, `name` or `location` before the first frame resets the prefix |
| `{"type":"voice", "name":"f1"}` | before the first frame | one of the released voices (`f0`-`f3`, `m0`-`m3`) |
| `{"type":"clone"}` then one binary message (float32 PCM, ≥ 2 s) | before the first frame | the agent speaks with that voice (microphone recording or a decoded audio file) |
| binary frame | while talking | one 80 ms frame of the user; the engine steps once per frame |
| `{"type":"reset"}` | any time | new conversation on the same connection (prefix, Context DB) |

`name` and `location` are not only profile fields: the text given to the model as its prefix is the
persona followed by ` The user's name is {name}.` and ` The user is in {location}.` (each only when
set, in that order, nothing else added). This is the exact phrasing of the training corpus, whose
dialogues with a named user carry `<system> {persona} The user's name is Priya. The user is in Sydney. <system>`;
a prefix that omits the name while the profile has one taught the model to invent one (#33). The `persona`
in `ready` is the persona alone, without these sentences.

## Server → client

| message | meaning |
| --- | --- |
| `ready {sample_rate, frame_size, persona}` | connection accepted; send `context`, then frames |
| `busy` | another conversation holds the engine |
| `voice {name, frames}` / `cloned {frames, seconds}` | voice set |
| binary frame | one 80 ms frame of agent audio |
| `text {delta}` | agent's words as they are produced |
| `user_text {utt, final, t, text}` | what the ASR heard, per utterance: partial (updates) then final |
| `ret` | the model asked for external information |
| `question {text, ms}` | the transcript that was sent to the router (`ms` = ASR latency) |
| `span {text, question, source, frames, seconds, prefill_ms}` | the Context Span that was injected; `source` = `mcp:<tool>` or the router's direct answer; `prefill_ms` = wall clock of the block read |
| `lead {ms, target_ms, rate}` (page-side, transcript only) | once a second: agent audio queued when a chunk arrived, the jitter-measured target, the playback rate in use |
| `underrun {late_ms, target_ms}` (page-side) | the playback queue ran dry; `late_ms` = silence until it refilled |
| `grow {from_ms, target_ms}` (page-side) | output held inside an agent pause to bring the queue up to the target |
| `drain {dropped_ms, excess_ms}` (page-side) | a pause chunk dropped to drain backlog a hole left |
| `cut {cut_ms}` (page-side) | pathological backlog cut to the target (oldest audio removed) |
| `slot {prefill_ms, step_ms, total_ms, over_budget}` | the frame right after a span read: block read + that frame's step against the 80 ms slot |
| `no_span {question}` | the router had nothing to look up: nothing injected, not a failure |
| `late {question, seconds, source}` | the answer arrived after the deadline and was dropped |
| `error {stage, message}` | a real failure (asr / retrieval / persona / voice) |

Turn structure for a chat-style rendering: a `user_text` with `final:true` closes a user utterance;
agent `text` deltas between user utterances belong to the agent's turn (full-duplex: both can be open at
once); `question`/`span`/`slot`/`no_span`/`late` are diagnostics tied to the preceding `ret` and can be hidden
behind a debug switch. Everything a session produced is also in the JSON transcript offered by Stop.
