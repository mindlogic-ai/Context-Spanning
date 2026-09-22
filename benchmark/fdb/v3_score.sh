#!/bin/bash
# Full-Duplex-Bench v3 official scorers on our results (provider=ours by default).
#   FDB_V3_DIR=<clone>/v3  V3_DATA=<released data dir with result_ours.json per sample>  bash benchmark/fdb/v3_score.sh
# evaluate_tool_calls (tool-selection F1, argument accuracy, response quality) and evaluate_pass_rate
# (strict binary) use a gpt-4o judge: export OPENAI_API_KEY. The turn-taking numbers of the paper's Table 2 (take-turn,
# latency, interruption, filler) come from v3_turn_taking.py: parakeet word timestamps on input.wav and on the agent audio
# output_<provider>.wav that v3_run.py writes, then the benchmark's analyze_tool_latency.py.
set -u
: "${FDB_V3_DIR:?set FDB_V3_DIR}"; : "${V3_DATA:?set V3_DATA}"
PROV="${PROV:-ours}"
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$FDB_V3_DIR"
python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$V3_DATA" --provider "$PROV" --output "${PROV}_evaluation_report.json" --use-llm
python evaluate_pass_rate.py  --benchmark benchmark_data_v2.json --results-dir "$V3_DATA" --provider "$PROV" --output "${PROV}_pass_rate_report.json" --use-llm
python "$HERE/v3_turn_taking.py" --data-root "$V3_DATA" --provider "$PROV" --fdb-v3 "$FDB_V3_DIR"
ls -la "$FDB_V3_DIR"/${PROV}_*report*.json
