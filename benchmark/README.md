# Benchmark

Everything a reported number depends on is in this folder. `run_full.sh` runs the live protocol end to end;
`rag/run.py --protocol api` is the API-backend protocol of the spoken-QA table. A number is comparable with
another only if both come from the same revision of this folder and the same protocol.

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
| speech model | Context Spanning-7B (training step 5,938), fine-tuned from `nvidia/personaplex-7b-v1` | `mindlogicinc/context-spanning-7b` `context_spanning_7b.pt`, `--checkpoint` |
| voices | released voice prompts, one per sample by a hash of the sample id | `voices/` in the Hub repo, `stack.py` |
| Context DB profile | `Seoul / Asia/Seoul`, no name, no facts | `stack.py` (`CTX`) |
| persona prompt | `You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.` | `rag/run.py`; FDB uses the PersonaPlex prompts in `fdb/v1_render.py` |
| sampling | audio temperature 0.8, text temperature 0.7 | `rag/run.py` |
| frame clock | 1.0x, 80 ms frames, one item at a time, one engine per lane | `contextspan/runtime/frame_stream.py` |
| `<ret>` | the model's own token, never forced. As in moshi-rag `run_inference.py`: 0.5 s after `<ret>` (`stt_wait_time`) the transcript so far is sent to the reference LLM, and the reference is applied whenever it arrives within the 10 s timeout | `rag/run.py` (`ret_fixed_wait_s`, `ret_deadline_s`); the live page keeps its own utterance path |
| ASR of the user | `Qwen/Qwen3-ASR-1.7B` on vLLM (`qwen-asr-serve`), utterance level | `scripts/backends.sh` |

## 2. Reference

The reference that answers a `<ret>` is the same for every set, HaluEvalAudio included (the protocol of the MoshiRAG
main table, where the reference is generated from the question and the gold passage is never shown). There are
two protocols; every table row names the one it came from.

**Live protocol** (`rag/run.py`, the default; `run_full.sh`): the reference LLM is a local server and its latency
is streamed, as moshi-rag `run_inference.py` with a local model.

- **every set**: the moshi-rag reference generator (`rag/moshirag.py`) on our router model
  `google/gemma-4-26B-A4B-it` (vLLM, `--gpu-memory-utilization 0.70`, `--max-model-len 8192`,
  `--enable-prefix-caching`; `MOSHIRAG_LLM_URL` and `MOSHIRAG_LLM_MODEL` name the server, and a run
  with another reference LLM is reported as such): kyutai-labs/moshi-rag `reference_prompt_template.txt` copied verbatim
  (`rag/reference_prompt_template.txt`), system prompt `You are a helpful assistant.`, the conversation so
  far as `Human:` / `moshi:` lines with earlier references interleaved, temperature 1.0, 64 tokens, stop at
  the first newline, 10 s timeout (the defaults of moshi-rag `run_inference.py`, its offline evaluation
  driver; a timeout injects nothing). No tools, no abstain clause.

A reference is injected whenever it arrives within the timeout: the runtime's late-span drop
(`CS_RET_DEADLINE_S`) is set to the same 10 s for the benchmark, as `run_inference.py` applies the
reference at trigger step + measured retrieval steps with no cut-off.

**API-backend protocol** (`rag/run.py --protocol api --reference-delay-s 0.8`; MoshiRAG arXiv 2604.12928
footnote 9: an API reference LLM, a uniform retrieval delay and no timeout): before the stream, the question audio
is transcribed by the same ASR and the reference is generated once per item by the API model (`MOSHIRAG_LLM_MODEL`,
default `gpt-4.1`, through `MOSHIRAG_LLM_URL`, default the OpenAI API with `OPENAI_API_KEY`) with the moshi-rag
server defaults (512 tokens, stop at the first newline; 60 s timeout, retried until it answers). During the stream
the span is injected exactly `--reference-delay-s` after the `<ret>`, replayed as 80 ms frames (0.8 s = 10 frames;
GPT-4.1 answered in 0.77 s on average), the stream waits at that frame for the
reference, nothing is dropped, no partial transcripts are taken and the `<ret>` closes the question utterance. The
stream is not paced to wall time, so several engines can share one GPU without changing a span's frame. The
references are written to `references_<K>.json` next to the items.

The deployed tool router (`contextspan/duetaspan/runtime/mcp`) is not part of the benchmark.

## 3. Datasets

| set | items | source | layout |
| --- | --- | --- | --- |
| HaluEvalAudio | 1000 | HaluEvalAudio release (question audio + gold passage) | `meta.json` + `audios/<id>.wav` |
| TriviaQA / WebQuestions / LlamaQuestions | 1000 / 1000 / 300 | OpenAudioBench release | `eval_datas/<set>/<set>.csv` + `audios/` |
| math | 3822: AddSub 395, MultiArith 600, SingleEq 508, SVAMP 1000, GSM8K 1319 | the five public math word-problem test sets of STITCH / MoshiRAG, spoken with the Kyutai TTS (built by `math/`, below) | `meta.json` + `audios/<id>.wav` |
| Full-Duplex-Bench v1 | 727 clips, 5 tasks | `v1_v1.5/dataset/data/v1.0` | the benchmark's |
| Full-Duplex-Bench v3 | 100 scenarios | `fdb_v3_data_released` | the benchmark's |

Every set is run whole, in dataset order. There is no subset, sample or smoke setting anywhere in this
folder: a number is a full-set number or it is not reported. `--shard K/N` is every N-th item from K, so
that two lanes with their own routers can render one set together.

The math set is built from its public sources on any machine with one GPU (`pip install moshi` for the Kyutai
TTS; the questions come from GitHub, the model and the voices from the Hugging Face Hub):

```bash
python benchmark/math/prepare_questions.py /data/math_audio   # -> questions/<set>.jsonl, 3822 rows
python benchmark/math/build_audio.py /data/math_audio         # -> meta.json + audios/<id>.wav
```

The build is always the whole set: `--dry-run` prints the item counts and voice assignment without a GPU, and
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

Two lanes. A router server serves exactly one lane, never two (a router that batches two lanes changes the
span latency of both), and a GPU that holds a Gemma holds nothing else: no engine, no ASR, no second
model. One engine per render GPU with its ASR beside it, so the 1.0x frame clock is met (three engines on
one 96 GB GPU render at about 0.75x and every span then lands early in frames; `rag/run.py` records
`render_speed_x` per item and warns below 0.98x). Every server is warmed with one request before the
first scored item. `run_full.sh` encodes the layout used for the paper (4 x 96 GB):

| GPU | processes |
| --- | --- |
| 0 | lane A engine, ASR A (`:8990`, 0.10) |
| 1 | router A (`:8004`) |
| 2 | router B (`:8006`) during rendering; gemma-3-27b-it judge (`:8007`, 0.70) after it |
| 3 | lane B engine, ASR B (`:8991`, 0.10) |

Both lanes run all five sets: lane A shard 0/2 with router A, lane B shard 1/2 with router B, so the lanes
finish together whatever the per-item time of a set. When rendering is done router B is replaced by the
judge and scoring starts, while FDB v1 and then v3 render on lane A with router A and the FDB v1 scorers
run on GPU 3.

Running it elsewhere: the environment names only servers and keys.

| variable | meaning |
| --- | --- |
| `MOSHIRAG_LLM_URL`, `MOSHIRAG_LLM_MODEL` | OpenAI-compatible server of the reference LLM and the model id it serves (default `http://localhost:8004`, `google/gemma-4-26B-A4B-it`) |
| `MOSHIRAG_GEMMA_JUDGE_URL` | vLLM server of `google/gemma-3-27b-it` (default `http://localhost:8007`) |
| `OPENAI_API_KEY` | `gpt-4o-2024-08-06` judge, FDB interruption judge, FDB v3 evaluators |
| `MOSHICP_ASR_URL`, `MOSHICP_ASR_MODEL` | the Qwen3-ASR server (`scripts/backends.sh`) |
| `MCP_ROUTER_LLM_URL`, `MCP_ROUTER_LLM_MODEL` | the deployed router, used by the FDB runs only |

```bash
CHECKPOINT=/path/context_spanning_7b.pt DATA=/data OUT=runs/full OPENAI_API_KEY=... bash benchmark/run_full.sh
python -m benchmark.rag.run web_questions <openaudiobench/eval_datas> runs/webq --checkpoint ckpt.pt   # one set, live
python -m benchmark.rag.run web_questions <openaudiobench/eval_datas> runs/webq_api --checkpoint ckpt.pt \
    --protocol api --reference-delay-s 0.8                                                             # api (GPT-4.1)
python -m benchmark.rag.score runs/webq                                                                # -> rag_report.json
```

## 6. Where this differs from moshi-rag's own driver

Stated so the comparison is read correctly. Everything not listed here is the same procedure.

| item | moshi-rag `run_inference.py` | here |
| --- | --- | --- |
| user ASR | Kyutai streaming STT (`LocalSpeechToText`), word level, VAD turn switching | `Qwen/Qwen3-ASR-1.7B`, utterance level; the router sees the whole transcript so far either way |
| reference conditioning | a separate reference encoder summed into the token embeddings over several steps | the span prefilled as text into the model's own text stream |
| end of a sample | after the input, until `max_consecutive_silence_frames` of model silence | a fixed 14 s window after the question |
| clock | the stream pauses during the retrieval call and the measured latency is replayed as frames | live protocol: the stream runs at 1.0x while the call is in flight; api protocol: the stream waits and the delay is replayed as frames, as moshi-rag |
| speech model | Moshi (MoshiRAG fine-tune), one voice | Context Spanning-7B (PersonaPlex fine-tune), released voices |

## 7. Live session

`live/` drives the shipped page through Playwright with synthetic TTS clips of twelve hand-written
questions (`live/cases.json`) and matches the answer on the page's text stream. It checks a deployment
end to end; paper numbers come from the suites above. `pip install -e '.[benchmark]' && playwright install chromium`.
Run it on the box or over `ssh -L`, never through a Cloudflare quick tunnel.

## 8. Results

The released checkpoint's per-set reports are in [`results/`](results/): the summary block of every
`rag_report.json` (n, ref / resp accuracy, P(resp | ref), `<ret>` and span rates, injection latency) and the
per-subset math accuracies, for the API-backend protocol (the paper's GPT-4.1 row). The tables themselves are in
the repository README.
