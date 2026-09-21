# Benchmark

Everything a reported number depends on is in this folder, and there is one method. `run_full.sh` runs
it end to end; a number is comparable with another only if both come from the same revision of this folder.

| Benchmark | Runner | Scorer |
| --- | --- | --- |
| MoshiRAG RAG suite: HaluEvalAudio, OpenAudioBench (TriviaQA, WebQuestions, LlamaQuestions), math | `benchmark.rag.run` | `benchmark.rag.score` (moshi-rag judges), `benchmark.rag.latency` (TTFAT, KD) |
| Full-Duplex-Bench v1 / v1.5 | `benchmark.fdb.v1_render` | the benchmark's own scorers |
| Full-Duplex-Bench v2 | `fdb/v2/go.sh` + `ours_adapter.js` | the benchmark's own scorers |
| Full-Duplex-Bench v3 | `benchmark.fdb.v3_run` | the benchmark's own evaluators (`fdb/v3_score.sh`) |
| Live session (`live/`) | `benchmark.live.run` | built in; a deployment check, not a paper number |

## 1. Systems under test

| role | what | pinned in |
| --- | --- | --- |
| speech model | DuetaSpan v7 step 8000, fine-tuned from `nvidia/personaplex-7b-v1` | `mindlogicinc/context-spanning-7b`, `--checkpoint` |
| voices | released voice prompts, one per sample by a hash of the sample id | `voices/` in the Hub repo, `stack.py` |
| Context DB profile | `Seoul / Asia/Seoul`, no name, no facts | `stack.py` (`CTX`) |
| persona prompt | `You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.` | `rag/run.py`; FDB uses the PersonaPlex prompts in `fdb/v1_render.py` |
| sampling | audio temperature 0.8, text temperature 0.7 | `rag/run.py` |
| frame clock | 1.0x, 80 ms frames, one item at a time, one engine per lane | `contextspan/runtime/frame_stream.py` |
| `<ret>` | the model's own token, never forced. As in moshi-rag `run_inference.py`: 0.5 s after `<ret>` (`stt_wait_time`) the transcript so far is sent to the reference LLM, and the reference is applied whenever it arrives within the 10 s timeout | `rag/run.py` (`ret_fixed_wait_s`, `ret_deadline_s`); the live page keeps its own utterance path |
| ASR of the user | `Qwen/Qwen3-ASR-1.7B` on vLLM (`qwen-asr-serve`), utterance level | `scripts/backends.sh` |

## 2. Reference

The reference that answers a `<ret>` is decided by the set, not by an option:

- **HaluEvalAudio**: the set's gold passage is the reference (MoshiRAG Table 8, GT reference).
- **every other set**: the moshi-rag reference generator (`rag/moshirag.py`) on our router model
  `google/gemma-4-26B-A4B-it` (vLLM, `--gpu-memory-utilization 0.70`, `--max-model-len 8192`,
  `--enable-prefix-caching`; `MOSHIRAG_LLM_URL` and `MOSHIRAG_LLM_MODEL` name the server, and a run
  with another reference LLM is reported as such): kyutai-labs/moshi-rag `reference_prompt_template.txt` copied verbatim
  (`rag/reference_prompt_template.txt`), system prompt `You are a helpful assistant.`, the conversation so
  far as `Human:` / `moshi:` lines with earlier references interleaved, temperature 1.0, 64 tokens, stop at
  the first newline, 10 s timeout (the defaults of moshi-rag `run_inference.py`, its offline evaluation
  driver; a timeout injects nothing). No tools, no abstain clause (MoshiRAG Table 9, LLM reference).

A reference is injected whenever it arrives within the timeout: the runtime's late-span drop
(`CS_RET_DEADLINE_S`) is set to the same 10 s for the benchmark, as `run_inference.py` applies the
reference at trigger step + measured retrieval steps with no cut-off.

The deployed tool router (`contextspan/duetaspan/runtime/mcp`) is not part of the benchmark.

## 3. Datasets

| set | items | source | layout |
| --- | --- | --- | --- |
| HaluEvalAudio | 1000 | HaluEvalAudio release (question audio + gold passage) | `meta.json` + `audios/<id>.wav` |
| TriviaQA / WebQuestions / LlamaQuestions | 1000 / 1000 / 300 | OpenAudioBench release | `eval_datas/<set>/<set>.csv` + `audios/` |
| math | 3822: AddSub 395, MultiArith 600, SingleEq 508, SVAMP 1000, GSM8K 1319 | the five public math word-problem test sets of STITCH / MoshiRAG, spoken with the Kyutai TTS (built by `math/`, below) | `meta.json` + `audios/<id>.wav` |
| Full-Duplex-Bench v1 | 727 clips, 5 tasks | `v1_v1.5/dataset/data/v1.0` | the benchmark's |
| Full-Duplex-Bench v3 | 100 scenarios | `fdb_v3_data_released` | the benchmark's |

Items are taken in dataset order. `--limit N` is the first N items;
`--shard K/N` is every N-th item from K, for two lanes sharing one set.

The math set is built from its public sources on any machine with one GPU (`pip install moshi` for the Kyutai
TTS; the questions come from GitHub, the model and the voices from the Hugging Face Hub):

```bash
python benchmark/math/prepare_questions.py /data/math_audio   # -> questions/<set>.jsonl, 3822 rows
python benchmark/math/build_audio.py /data/math_audio         # -> meta.json + audios/<id>.wav
```

`build_audio.py --limit 100` builds only the first 100 items, a quick subset laid out and noised exactly as
those items are in the full build; `--dry-run` prints the item counts and voice assignment without a GPU, and
an interrupted build resumes. Each question is spoken with a voice from the raw `voice-donations/` recordings of
`kyutai/tts-voices`, chosen by a hash of the item id, and written as 24 kHz stereo with 0.35 s of lead silence,
the speech on the left channel and low-level noise on the right (the HaluEvalAudio layout); the docstring of
`build_audio.py` gives the full recipe.

## 4. Judges and metrics (`rag/score.py`, copied from moshi-rag `evaluate`)

| set | judge | prompt | judged text |
| --- | --- | --- | --- |
| HaluEvalAudio | `google/gemma-3-27b-it` on vLLM, temperature 1.0 | `SimpleQALLMJudge` | the model's text stream, `<ret>`/`<span>` removed |
| math | `google/gemma-3-27b-it` on vLLM, temperature 1.0 | `MathQALLMJudge` (Yes/No) | same |
| TriviaQA, WebQuestions | `gpt-4o-2024-08-06`, temperature 0 | `TriviaQAJudge` (JSON judgment) | same |
| LlamaQuestions | `gpt-4o-2024-08-06`, temperature 0 | `LLamaQuestionsJudge` | same |

- `ref` = the injected reference judged against the gold answers, averaged over items whose reference text
  is non-empty (`n_ref_judged`).
- `resp` = the model's text stream judged against the gold answers, averaged over items whose text is
  non-empty (`n_resp_judged`).
- `P(resp|ref)` = `resp` on the items whose reference was judged correct (our headline number).
- `ret_rate`, `span_rate`, `inj_lat_s` (seconds from `<ret>` to injection); `resp_acc_all` / `ref_acc_all`
  are the same accuracies over all items, for comparison with earlier tables.

Full-Duplex-Bench v1: the benchmark's scorers (`get_transcript/asr.py`, `evaluation/evaluate.py`);
`user_interruption` uses its GPT judge (`gpt-4-turbo`), so `OPENAI_API_KEY` is required. v3: the
benchmark's `evaluate_tool_calls.py` and `evaluate_pass_rate.py` with `--use-llm` (`gpt-4o`), provider
`ours`, tool universe = the benchmark's 12 mock APIs only (`MOSHICP_TOOLPACK_ONLY=1`).

## 5. Machine layout and run conditions

Two lanes, one router per lane, the ASR of a lane next to its engine, the judge on its own GPU, nothing
else on those GPUs, every server warmed with one request before the first scored item. `run_full.sh`
encodes the layout used for the paper (4 x 96 GB):

| GPU | processes |
| --- | --- |
| 0 | router B (`:8006`) |
| 1 | router A (`:8004`), lane A engine |
| 2 | gemma-3-27b-it judge (`:8007`, 0.70), ASR A (`:8990`, 0.10) |
| 3 | lane B engine, ASR B (`:8991`, 0.10) |

Lane A: WebQuestions, math, HaluEval shard 0/2. Lane B: TriviaQA, LlamaQuestions, HaluEval shard 1/2.
FDB v1 and v3 follow on lane B with the same servers.

Running it elsewhere: the environment names only servers and keys.

| variable | meaning |
| --- | --- |
| `MOSHIRAG_LLM_URL`, `MOSHIRAG_LLM_MODEL` | OpenAI-compatible server of the reference LLM and the model id it serves (default `http://localhost:8004`, `google/gemma-4-26B-A4B-it`) |
| `MOSHIRAG_GEMMA_JUDGE_URL` | vLLM server of `google/gemma-3-27b-it` (default `http://localhost:8007`) |
| `OPENAI_API_KEY` | `gpt-4o-2024-08-06` judge, FDB interruption judge, FDB v3 evaluators |
| `MOSHICP_ASR_URL`, `MOSHICP_ASR_MODEL` | the Qwen3-ASR server (`scripts/backends.sh`) |
| `MCP_ROUTER_LLM_URL`, `MCP_ROUTER_LLM_MODEL` | the deployed router, used by the FDB runs only |

```bash
CHECKPOINT=/path/keep_step8000.pt DATA=/data OUT=runs/full OPENAI_API_KEY=... bash benchmark/run_full.sh
python -m benchmark.rag.run web_questions <openaudiobench/eval_datas> runs/webq --checkpoint ckpt.pt   # one set
python -m benchmark.rag.score runs/webq                                                                # -> rag_report.json
```

## 6. Where this differs from moshi-rag's own driver

Stated so the comparison is read correctly. Everything not listed here is the same procedure.

| item | moshi-rag `run_inference.py` | here |
| --- | --- | --- |
| user ASR | Kyutai streaming STT (`LocalSpeechToText`), word level, VAD turn switching | `Qwen/Qwen3-ASR-1.7B`, utterance level; the router sees the whole transcript so far either way |
| reference conditioning | a separate reference encoder summed into the token embeddings over several steps | the span prefilled as text into the model's own text stream |
| end of a sample | after the input, until `max_consecutive_silence_frames` of model silence | a fixed 14 s window after the question |
| clock | the stream pauses during the retrieval call and the measured latency is replayed as frames | the stream runs at 1.0x while the call is in flight |
| speech model | Moshi (MoshiRAG fine-tune), one voice | DuetaSpan v7 8000 (PersonaPlex fine-tune), released voices |

## 7. Live session

`live/` drives the shipped page through Playwright with synthetic TTS clips of twelve hand-written
questions (`live/cases.json`) and matches the answer on the page's text stream. It checks a deployment
end to end; paper numbers come from the suites above. `pip install -e '.[benchmark]' && playwright install chromium`.
Run it on the box or over `ssh -L`, never through a Cloudflare quick tunnel.
