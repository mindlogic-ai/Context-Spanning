# Results of the released checkpoint

Per-set reports of `benchmark/rag` runs with `mindlogicinc/context-spanning-7b` (`context_spanning_7b.pt`,
training step 5,938), API-backend protocol (`rag/run.py --protocol api`, reference LLM GPT-4.1, MoshiRAG
judges, seed 42424242, every set whole). Each file holds the summary block of every set's `rag_report.json`
and, for math, the per-subset accuracies. Accuracies are fractions; `ref_acc` and `resp_acc` are averaged over
the items the judge could judge (`n_ref_judged`, `n_resp_judged`), `P(resp|ref)` is `resp_acc` over the items
whose reference was judged correct.

| file | delay after `<ret>` | LlamaQ ref / resp | WebQ | TriviaQA | HaluEval | math (3,822) |
|---|---|---|---|---|---|---|
| `api_gpt41_0.8s.json` (the paper's GPT-4.1 row) | 0.8 s = 10 frames | 89.9 / 83.3 | 79.4 / 66.7 | 91.1 / 83.8 | 68.3 / 55.5 | 84.4 / 78.3 |
| `api_gpt41_1.04s.json` (delay ablation) | 1.04 s = 13 frames | 89.3 / 81.3 | 79.0 / 61.7 | 91.1 / 82.9 | 67.7 / 52.2 | 84.3 / 72.6 |

Math per subset (ref / resp, %), 0.8 s: AddSub 82.6 / 74.7, MultiArith 95.2 / 89.7, SinglEq 85.9 / 82.2,
SVAMP 89.2 / 85.4, GSM8K 75.7 / 67.4. At 1.04 s the same reference columns and lower response columns
(68.6, 86.5, 74.7, 77.8, 62.7): the model commits to its answer about one second after `<ret>`, so a span that
lands later is used less.

The reference (`ref`) columns of the two runs agree within 0.6 points because the references are the same
GPT-4.1 outputs; only the frame they land on differs.
