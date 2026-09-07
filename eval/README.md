# Evaluation

Everything under `eval/` is the benchmark harness. It is separate from the runtime: it imports
`contextspan` as a library and never the other way round. Every runner streams the benchmark audio
through the same stack a user talks to — `Engine` at the 1.0x frame clock, the DuetaSpan backend
(router + tool bank + LLM-RAG), the ASR endpoint — and writes the benchmark's own layout, so the
benchmark's official scorers run on the outputs unchanged. Nothing is scored on text shortcuts.

| Benchmark | What it measures | Runner | Scorer |
| --- | --- | --- | --- |
| MoshiRAG RAG suite: HaluEvalAudio, OpenAudioBench (LlamaQuestions, WebQuestions, TriviaQA), math | knowledge grounding: ref. / resp. accuracy, `<ret>` rate, latency (TTFAT, KD) | `eval.rag.run` | `eval.rag.transcribe` → `eval.rag.score`, `eval.rag.latency` |
| Full-Duplex-Bench v1 / v1.5 | pause handling, smooth turn-taking, backchannel, interruption | `eval.fdb.v1_render` | the benchmark's scorers |
| Full-Duplex-Bench v2 | examiner (GPT Realtime) multi-turn conversation | `eval/fdb/v2/go.sh` + `ours_adapter.js` | the benchmark's scorers |
| Full-Duplex-Bench v3 | tool calling under disfluency: tool selection, argument accuracy, pass rate | `eval.fdb.v3_run` | `eval/fdb/v3_score.sh` (official) |

## Run conditions (every benchmark)

Span latency is part of what is measured, so the machine state is part of the protocol:

- **Warm.** Every model is loaded and has served at least one request before the first scored item:
  the speech engine, the router / RAG LLM, the ASR, the tool servers. The runners prewarm the backend
  and the ASR themselves (`eval/common.py`); start the LLM server well before the run and send it one
  request. A cold first item is a measurement of loading time, not of the system.
- **No parallelism.** One benchmark process at a time, one item at a time, one engine. No sharding
  across GPUs, no concurrent runs of another suite, no other inference job on the GPUs the run uses.
  The router LLM and the ASR each sit alone on their GPU(s). Contention shows up as span latency and
  the numbers are then about the hardware, not the model.
- **Lowest-latency mode.** The servers run with the settings in `scripts/backends.sh` (JSON-mode router
  replies, speculative decoding, prefix caching, bounded decode); the engine steps at the 1.0x frame
  clock; retrieval is never delayed or batched to make a span "arrive on time". The system is measured
  as it is deployed.

Servers: the same router / RAG / ASR servers as the runtime (see the top-level README) plus a judge LLM
for the RAG suite (`JUDGE_LLM_URL`, `JUDGE_LLM_MODEL`; `scripts/env.sh` points it at the router server unless
`JUDGE_MODEL` is set — the paper's RAG-suite judge was google/gemma-4-31B-it on its own server).

## Protocols: full and semi

Two named protocols, so a number is never quoted without its scale.

| | RAG suite | math | FDB v1 / v1.5 | FDB v2 | FDB v3 |
| --- | --- | --- | --- | --- | --- |
| **full** | HaluEvalAudio 1000, LlamaQuestions 300, WebQuestions 1000, TriviaQA 1000 | 100 | all clips (v1 727 + v1.5 overlap set) | 200 tasks | 100 scenarios |
| **semi** | first 120 items of each set (deterministic order) | 100 | — | — | 100 scenarios |

The tool universe differs by benchmark and is part of the protocol: the RAG suite and FDB v1/v1.5/v2 run
with the shipped 60-tool bank plus the four MCP servers; FDB v3 runs with **only** the benchmark's 12
official mock APIs (`MOSHICP_TOOLPACK_ONLY=1`), the product bank being unmounted for the run, because
v3 scores tool names and arguments against exactly those twelve.

`semi` is the checkpoint-comparison protocol: it runs in about an hour per RAG set on one GPU and is
what the checkpoint tables in this repository's history use (`--limit 120`). `full` is the reporting
protocol. Items are taken in dataset order, so a `semi` run is a prefix of the corresponding `full`
run and the two never disagree on an item they share. Every run directory carries its own
`rag_report.json` with `n`, and a result quoted from a `semi` run must say so.

```bash
python -m eval.rag.run halueval <halueval_audio> runs/semi/halueval --limit 120     # semi
python -m eval.rag.run halueval <halueval_audio> runs/full/halueval                 # full
```

## RAG suite

```bash
python -m eval.rag.run halueval        <halueval_audio>  runs/halueval            # arm=router (default)
python -m eval.rag.run halueval        <halueval_audio>  runs/halueval_gold --arm gold
python -m eval.rag.run llama_questions <openaudiobench/eval_datas> runs/llamaq
python -m eval.rag.transcribe runs/halueval        # whisper-large-v3 transcript of the agent (OAB protocol)
python -m eval.rag.score      runs/halueval        # ref / resp accuracy, span rate, ret/inject latency
python -m eval.rag.latency    runs/halueval        # TTFAT / KD / E2EKD (parakeet word timestamps)
```

Data layout. HaluEvalAudio and the math set: a directory with `meta.json` (`[{id, text, answer,
knowledge}]`, `knowledge` = the gold passage for HaluEval) and `audios/<id>.wav`; the spoken questions
are the HaluEvalAudio release (the math questions were synthesised with a TTS from MoshiRAG's Table 10
items). OpenAudioBench: the `eval_datas/<set>/<set>.csv` + `audios/` layout of the OpenAudioBench release.

Arms. `router` (default for HaluEval) hands the gold passage to the router as Context DB text and injects
the router's answer — the deployed path. `gold` injects the passage itself (MoshiRAG Table 8 protocol).
`real` uses the deployed backend with no passage (Table 9). `off` lets `<ret>` fire with nothing arriving.
`ref` accuracy is judged on the span that arrived, `resp` on what the agent said; the judge prompt is
MoshiRAG Table 16 verbatim, with literal containment settling `ref` first.

## Full-Duplex-Bench

v1 / v1.5: `python -m eval.fdb.v1_render <fdb>/v1_v1.5/data <out>` writes `{task}/{id}/output.wav`
(and `clean_output.wav` for overlap tasks) with the benchmark's PersonaPlex prompts, then run the
benchmark's own scoring. v2: `eval/fdb/v2/go.sh` copies our examinee adapter into the benchmark clone
and drives its orchestrator against a running `python main.py serve`; scoring is the benchmark's
(NeMo alignment + Gemini judge). v3: the router's tool universe is the benchmark's own 12 mock APIs
(`eval/fdb/v3_toolpack.py`, mounted with `MOSHICP_EXTRA_TOOLPACK` / `MOSHICP_TOOLPACK_ONLY`) — the
60-tool product bank is out of the run; `python -m eval.fdb.v3_run <v3_data_released>` writes
`result_ours.json` per sample with the official evaluators' keys; `eval/fdb/v3_score.sh` runs them.
Latency metrics of v3 are partial on this path: the official run measures audio timings through
LiveKit, ours records the frame time of each tool call.

## Voice and profile

A prefix without a voice block is outside the training distribution, so every sample is conditioned on
a released voice chosen by a hash of the sample key (`--voice f0` pins one). The Context DB profile is
`Seoul / Asia/Seoul` for every run, as in the paper.
