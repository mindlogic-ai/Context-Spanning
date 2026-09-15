#!/bin/bash
# Full-Duplex-Bench v3 official scorers on our results (provider=ours by default).
#   FDB_V3_DIR=<clone>/v3  V3_DATA=<released data dir with result_ours.json per sample>  bash benchmark/fdb/v3_score.sh
# evaluate_tool_calls (tool-selection F1, argument accuracy, response quality) and evaluate_pass_rate
# (strict binary) use a gpt-4o judge: export OPENAI_API_KEY. analyze_tool_latency is best-effort: this
# token path carries frame-time `timestamp_start` only, not the audio timings the official LiveKit run has.
set -u
: "${FDB_V3_DIR:?set FDB_V3_DIR}"; : "${V3_DATA:?set V3_DATA}"
PROV="${PROV:-ours}"
cd "$FDB_V3_DIR"
python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$V3_DATA" --provider "$PROV" --output "${PROV}_evaluation_report.json" --use-llm
python evaluate_pass_rate.py  --benchmark benchmark_data_v2.json --results-dir "$V3_DATA" --provider "$PROV" --output "${PROV}_pass_rate_report.json" --use-llm
python analyze_tool_latency.py --results-dir "$V3_DATA" --provider "$PROV" --output "${PROV}_latency_report.json" || echo "latency analysis limited (expected on the token path)"
ls -la "$FDB_V3_DIR"/${PROV}_*report*.json
