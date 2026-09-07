#!/usr/bin/env bash
# Full-Duplex-Bench v2: GPT Realtime examiner (role A) talks to our live server (role B) through the
# benchmark's orchestrator; our side is eval/fdb/v2/ours_adapter.js (WebRTC 48 kHz PCM16 <-> our
# WebSocket, raw float32 24 kHz, one 80 ms frame per tick so the frame clock never stops).
#   FDB_V2_DIR=<clone>/v2  OURS_WS_URL=ws://localhost:8080/ws  OPENAI_API_KEY=...  bash eval/fdb/v2/go.sh
#   LIMIT=2 / SPLIT=Daily / DURATION=60 / EXAMINER_MODE=fast narrow the run.
# Each task holds a paid GPT Realtime session for DURATION seconds; 200 x 120 s is hours of audio.
set -euo pipefail
: "${FDB_V2_DIR:?set FDB_V2_DIR}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
DURATION="${DURATION:-120}"; EXAMINER_MODE="${EXAMINER_MODE:-slow}"; LIMIT="${LIMIT:-}"; SPLIT="${SPLIT:-}"
OUT="${OUT:-$FDB_V2_DIR/runs/ours_$(date +%m%d_%H%M)}"
export SIGNAL_PORT="${SIGNAL_PORT:-9100}"           # the orchestrator reads it from env only
export OURS_WS_URL="${OURS_WS_URL:-ws://localhost:8080/ws}"
export EXAMINEE_SYSTEM_PROMPT="${EXAMINEE_SYSTEM_PROMPT:-You are a helpful AI assistant. Always speak in English.}"
export DEFAULT_DURATION="$DURATION"
mkdir -p "$OUT" "$FDB_V2_DIR/adapters"
cp "$(dirname "$0")/ours_adapter.js" "$FDB_V2_DIR/adapters/ours_adapter.js"
ARGS=(prompts_staged_200.json --base-out "$OUT" --adapter-b adapters/ours_adapter.js --examiner-mode "$EXAMINER_MODE" --duration "$DURATION")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT"); [[ -n "$SPLIT" ]] && ARGS+=(--split "$SPLIT")
cd "$FDB_V2_DIR"
bash run_dataset.sh "${ARGS[@]}" 2>&1 | tee -a "$OUT/run.log"
echo "[v2] recorded $(find "$OUT" -name B.wav | wc -l) conversations -> $OUT"
echo "[v2] scoring (benchmark's own): bash eval/prepare_evaluation.sh $OUT ; bash eval/run_evaluation.sh --root_dir $OUT ;"
echo "     python scoring/parse.py --root_dir <eval_out> && python scoring/score.py --root_dir <eval_out>"
