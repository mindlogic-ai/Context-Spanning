# Results of the released checkpoint

Per-set report of the `benchmark/rag` run with `mindlogicinc/context-spanning-7b` (`context_spanning_7b.pt`,
training step 5,938): API-backend protocol (`rag/run.py --protocol api --reference-delay-s 0.8`, reference LLM
GPT-4.1 injected 0.8 s = 10 frames after `<ret>`), MoshiRAG judges, seed 42424242, every set whole. The file
holds the summary block of every set's `rag_report.json` and, for math, the per-subset accuracies. Accuracies
are fractions; `ref_acc` and `resp_acc` are averaged over the items the judge could judge (`n_ref_judged`,
`n_resp_judged`), `P(resp|ref)` is `resp_acc` over the items whose reference was judged correct.

| file | LlamaQ ref / resp | WebQ | TriviaQA | HaluEval | math (3,822) |
|---|---|---|---|---|---|
| `api_gpt41_0.8s.json` | 89.9 / 83.3 | 79.4 / 66.7 | 91.1 / 83.8 | 68.3 / 55.5 | 84.4 / 78.3 |

Math per subset (ref / resp, %): AddSub 82.6 / 74.7, MultiArith 95.2 / 89.7, SinglEq 85.9 / 82.2,
SVAMP 89.2 / 85.4, GSM8K 75.7 / 67.4.
