#!/bin/bash
# Full-Duplex-Bench v3: the official artefacts and scorers on our results (provider=ours by default).
#   FDB_V3_DIR=<clone>/v3  V3_DATA=<dir holding the 100 scenario folders with result_ours.json>  bash benchmark/fdb/v3_score.sh
# 1. v3_artefacts.py: parakeet ASR fields exactly as v3/run_tool_benchmark.py writes them (transcript = ASR of the agent
#    audio, asr_chunks, user_speech_end_rel, audio_agent_speech_start) and the official analyze_tool_latency.py.
# 2. evaluate_tool_calls.py (tool selection, argument accuracy, response quality, interruption) and evaluate_pass_rate.py,
#    both with the gpt-4o judge (export OPENAI_API_KEY).
# 3. v3_summary.py: the table numbers read from those official reports only.
set -u
: "${FDB_V3_DIR:?set FDB_V3_DIR}"; : "${V3_DATA:?set V3_DATA}"
PROV="${PROV:-ours}"
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/v3_artefacts.py" --data-root "$V3_DATA" --provider "$PROV" --fdb-v3 "$FDB_V3_DIR"
( cd "$FDB_V3_DIR" && python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$V3_DATA" --provider "$PROV" --output "$V3_DATA/evaluation_report.json" --use-llm
  python evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$V3_DATA" --provider "$PROV" --output "$V3_DATA/pass_rate_report.json" --use-llm )
python "$HERE/v3_summary.py" --data-root "$V3_DATA" --provider "$PROV" --evaluation "$V3_DATA/evaluation_report.json" --pass-rate "$V3_DATA/pass_rate_report.json" --out "$V3_DATA/summary.json"
