# Benchmark protocol

This file is the complete list of what a reported number was produced with. Every item is a file in
this repository or a pinned public artifact, so a number can be reproduced from the commit hash alone.
`benchmark/run_full.sh` runs the whole protocol with these settings and nothing else.

## 1. Systems under test

| role | what | where it is pinned |
| --- | --- | --- |
| speech model | DuetaSpan v7 step 8000, fine-tuned from `nvidia/personaplex-7b-v1` | `mindlogicinc/context-spanning-7b` on the Hub, `--checkpoint` |
| voices | released voice prompts, one per sample by a hash of the sample id | `voices/` in the Hub repo, `benchmark/stack.py` |
| Context DB profile | `Seoul / Asia/Seoul`, no name, no facts | `benchmark/stack.py` (`CTX`) |
| persona prompt | `You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.` | `benchmark/rag/run.py` (`PROMPT`); FDB uses the PersonaPlex prompts in `benchmark/fdb/v1_render.py` |
| sampling | audio temperature 0.8, text temperature 0.7 | `benchmark/stack.py` defaults, fixed for every run |
| frame clock | 1.0x, 80 ms frames, one item at a time, one engine per lane | `contextspan/runtime/frame_stream.py` |
| `<ret>` handling | `<ret>` is the model's own token, never forced; the question is the utterance transcript (`CS_RET_CUT_S` 0.4 s cut, `CS_RET_UTT_WAIT_S` 1.0 s); a span later than `CS_RET_DEADLINE_S` 2.5 s after `<ret>` is dropped | `contextspan/runtime/frame_stream.py` |
| ASR of the user | `Qwen/Qwen3-ASR-1.7B` on vLLM (`qwen-asr-serve`, OpenAI audio API), utterance level | `scripts/backends.sh` (`ASR_BACKEND=vllm`) |

## 2. Reference backend (the "router")

The reference LLM is our router model, `google/gemma-4-26B-A4B-it` on vLLM (`--gpu-memory-utilization 0.70`,
`--max-model-len 8192`, `--enable-prefix-caching`). One router serves one lane. Two arms exist and the
report says which one a number comes from:

| arm | prompt | tools | abstain | used for |
| --- | --- | --- | --- | --- |
| `moshirag` | kyutai-labs/moshi-rag `reference_prompt_template.txt`, copied verbatim into `benchmark/rag/moshirag_prompts/` | none | none: the LLM always writes a reference | **the reported RAG numbers** (MoshiRAG Table 9 protocol, LLM reference) |
| `gold` | none: the gold passage of HaluEvalAudio is the reference | none | none | **the reported HaluEval numbers** (MoshiRAG Table 8 protocol, GT reference) |
| `real` | the deployed tool router (`contextspan/duetaspan/runtime/mcp/client.py` `_SYSTEM_PROMPT`) | the shipped 60-tool bank + MCP servers | the router may answer `(no information found)` | deployment checks only, not reported |

`moshirag` arm parameters, all from the moshi-rag repository (`LLMReferenceGenerator`, `LLMClient`,
`RAGManager`): system prompt `You are a helpful assistant.`, the conversation so far as `Human:` /
`moshi:` lines with earlier references interleaved, `Reference:` cue, temperature 1.0, `max_tokens` 512,
stop at the first newline, `rag_timeout` 1.5 s (a timeout injects nothing). `benchmark/rag/moshirag.py`.

## 3. Datasets

| set | items | source | layout |
| --- | --- | --- | --- |
| HaluEvalAudio | 1000 | HaluEvalAudio release (question audio + gold passage) | `meta.json` + `audios/<id>.wav` |
| TriviaQA / WebQuestions / LlamaQuestions | 1000 / 1000 / 300 | OpenAudioBench release | `eval_datas/<set>/<set>.csv` + `audios/` |
| math (AddSub, MultiArith, SingleEq, SVAMP, GSM8K samples) | 100 | MoshiRAG Table 10 items, synthesised with the Kyutai TTS VCTK preset voices | `meta.json` + `audios/` |
| Full-Duplex-Bench v1 | 727 clips, 5 tasks | the benchmark's `v1_v1.5/dataset/data/v1.0` | the benchmark's layout |
| Full-Duplex-Bench v3 | 100 scenarios | `fdb_v3_data_released` | the benchmark's layout |

Items are taken in dataset order; `--limit N` is the first N items; `--shard K/N` is every N-th item
from K (two lanes sharing one set). A `semi` run is the first 120 items and must be labelled as such.

## 4. Judges and metric definitions

RAG suite (`benchmark/rag/score.py --protocol moshirag`), copied from moshi-rag `evaluate/judge`:

| set | judge model | prompt | what is judged |
| --- | --- | --- | --- |
| HaluEvalAudio | `google/gemma-3-27b-it` on vLLM | `SimpleQALLMJudge` | the model's text stream (inner monologue), `<ret>`/`<span>` removed |
| math | `google/gemma-3-27b-it` on vLLM | `MathQALLMJudge` (Yes/No) | same |
| TriviaQA, WebQuestions | `gpt-4o-2024-08-06`, temperature 0 | `TriviaQAJudge` (JSON judgment) | same |
| LlamaQuestions | `gpt-4o-2024-08-06`, temperature 0 | `LLamaQuestionsJudge` | same |

Definitions, as in moshi-rag `evaluate/score.py`:

- `ref` = the injected reference judged against the gold answers; averaged over items whose
  reference text is non-empty (`n_ref_judged`).
- `resp` = the model's text stream judged against the gold answers; averaged over items whose text
  is non-empty (`n_resp_judged`).
- `P(resp|ref)` = `resp` on the items whose reference was judged correct (our headline number).
- `ret_rate` = items with at least one `<ret>`; `span_rate` = items with at least one injected span;
  `inj_lat_s` = seconds from `<ret>` to injection; `resp_acc_all` / `ref_acc_all` = the same
  accuracies over all items (no exclusion) for comparison with earlier reports.

The audio is also transcribed with `openai/whisper-large-v3` (`benchmark.rag.transcribe`) and kept in
the row as `hyp`; it is not what the judges see under this protocol.

Full-Duplex-Bench v1: the benchmark's own scorers (`get_transcript/asr.py`, `evaluation/evaluate.py`);
`user_interruption` uses the benchmark's GPT judge (`gpt-4-turbo`), so `OPENAI_API_KEY` must be set.
Full-Duplex-Bench v3: the benchmark's `evaluate_tool_calls.py` and `evaluate_pass_rate.py` with `--use-llm`
(`gpt-4o`), provider `ours`, tool universe = the benchmark's 12 mock APIs only (`MOSHICP_TOOLPACK_ONLY=1`).

## 5. Machine layout and run conditions

Two lanes, one router per lane, the ASR of a lane on the same GPU as its router or engine, the judge on
its own GPU, no other job on the GPUs a lane uses. Every server has served a request before the first
scored item. `benchmark/run_full.sh` encodes the layout used for the paper (4 x 96 GB):

| GPU | processes |
| --- | --- |
| 0 | router B (`:8006`) |
| 1 | router A (`:8004`), lane A engine |
| 2 | gemma-3-27b-it judge (`:8007`, 0.70), ASR A (`:8990`, 0.10) |
| 3 | lane B engine, ASR B (`:8991`, 0.10) |

Lane A: WebQuestions, math, HaluEval shard 0/2. Lane B: TriviaQA, LlamaQuestions, HaluEval shard 1/2.
FDB v1 and v3 run after the RAG suite on one lane with the same servers.

## 6. Changes of protocol are changes of this file

A number is comparable with another only if both were produced under the same revision of this file.
The bench tree records the commit it ran from (`.git_rev`); the run directory carries `rag_report_moshirag.json`
with `protocol`, `n`, `n_ref_judged`, `n_resp_judged`.
