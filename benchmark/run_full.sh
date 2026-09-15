#!/bin/bash
# The full benchmark protocol of benchmark/README.md, end to end: servers, the five RAG sets on two lanes,
# scoring with the moshi-rag judges, then Full-Duplex-Bench v1 (all clips, official scorers) and v3.
#
#   CHECKPOINT=/path/keep_step8000.pt DATA=/data OUT=runs/full bash benchmark/run_full.sh
#
# DATA holds halueval_audio/, math_audio/, openaudiobench/eval_datas/, full_duplex_bench/ (the benchmark
# clone with v1_v1.5/ and v3/, plus v3_data/fdb_v3_data_released). The models are the protocol's and are not
# arguments (ASR_MODEL may point at a local copy of Qwen/Qwen3-ASR-1.7B). OPENAI_API_KEY for the
# OpenAudioBench judge, the FDB interruption judge and the v3 evaluators.
# GPU layout (README.md section 5): GPU_ROUTER_A GPU_ROUTER_B GPU_JUDGE GPU_LANE_B; lane A shares GPU_ROUTER_A.
set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT}"; : "${DATA:?set DATA}"; : "${OUT:?set OUT}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
ROUTER_MODEL=google/gemma-4-26B-A4B-it; JUDGE_MODEL=google/gemma-3-27b-it; ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
GPU_ROUTER_A="${GPU_ROUTER_A:-1}"; GPU_ROUTER_B="${GPU_ROUTER_B:-0}"; GPU_JUDGE="${GPU_JUDGE:-2}"; GPU_LANE_B="${GPU_LANE_B:-3}"
PY="${PYTHON:-python}"; FDBPY="${FDB_PYTHON:-$PY}"; LOG="$OUT/logs"; mkdir -p "$OUT" "$LOG"
log(){ echo "$(date '+%F %T') [bench] $*" | tee -a "$LOG/chain.log"; }
serve(){ # $1 gpu  $2 model  $3 port  $4 mem  $5 log
  CUDA_VISIBLE_DEVICES="$1" setsid nohup "$PY" -m vllm.entrypoints.openai.api_server --model "$2" --served-model-name "$2" \
    --port "$3" --tensor-parallel-size 1 --gpu-memory-utilization "$4" --max-model-len 8192 --enable-prefix-caching \
    --trust-remote-code > "$LOG/$5.log" 2>&1 < /dev/null & }
asr(){ CUDA_VISIBLE_DEVICES="$1" setsid nohup qwen-asr-serve "$ASR_MODEL" --port "$2" --gpu-memory-utilization 0.10 \
    --max-model-len 4096 --max-num-seqs 64 --served-model-name qwen3-asr > "$LOG/asr_$2.log" 2>&1 < /dev/null & }
up_llm(){ curl -s -m 5 "localhost:$1/v1/models" 2>/dev/null | grep -q "$2"; }
up_asr(){ [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "localhost:$1/health" 2>/dev/null)" = "200" ]; }
log "start: $CHECKPOINT; router $ROUTER_MODEL; judge $JUDGE_MODEL; ASR $ASR_MODEL; commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
up_llm 8004 "$ROUTER_MODEL" || serve "$GPU_ROUTER_A" "$ROUTER_MODEL" 8004 0.62 routerA
up_llm 8006 "$ROUTER_MODEL" || serve "$GPU_ROUTER_B" "$ROUTER_MODEL" 8006 0.70 routerB
up_llm 8007 "$JUDGE_MODEL"  || serve "$GPU_JUDGE" "$JUDGE_MODEL" 8007 0.70 judge
for i in $(seq 1 120); do up_llm 8004 "$ROUTER_MODEL" && up_llm 8006 "$ROUTER_MODEL" && up_llm 8007 "$JUDGE_MODEL" && break; sleep 10; done
up_asr 8990 || asr "$GPU_JUDGE" 8990; up_asr 8991 || asr "$GPU_LANE_B" 8991
for i in $(seq 1 60); do up_asr 8990 && up_asr 8991 && break; sleep 10; done
up_llm 8004 "$ROUTER_MODEL" && up_llm 8006 "$ROUTER_MODEL" && up_llm 8007 "$JUDGE_MODEL" && up_asr 8990 && up_asr 8991 || { log "servers not ready, see $LOG"; exit 2; }
# one warm request each (README.md section 5)
for p in 8004 8006 8007; do curl -s -m 60 "localhost:$p/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$( [ $p = 8007 ] && echo $JUDGE_MODEL || echo $ROUTER_MODEL)\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" > /dev/null; done
log "servers up and warm"
export MOSHIRAG_GEMMA_JUDGE_URL=http://localhost:8007
export MCP_ROUTER_LLM_API=openai MCP_ROUTER_LLM_MODEL="$ROUTER_MODEL" MOSHICP_MCP=1 MOSHICP_ASR_MODEL=qwen3-asr
R="$OUT/rag"; mkdir -p "$R"
run_set(){ # $1 set  $2 data root  $3 gpu  $4 shard-or-empty
  local sh=""; [ -n "$4" ] && sh="--shard $4"
  CUDA_VISIBLE_DEVICES="$3" "$PY" -m benchmark.rag.run "$1" "$2" "$R/$1" --checkpoint "$CHECKPOINT" $sh \
    > "$LOG/rag_$1${4:+_shard${4%/*}}.log" 2>&1 || log "$1 run failed (gpu $3, shard '$4')"
  log "$1 rendered (gpu $3, shard '$4')"; }
score_set(){ # $1 set
  "$PY" -m benchmark.rag.score "$R/$1" >> "$LOG/rag_$1.log" 2>&1
  log "$1: $("$PY" -c "import json;s=json.load(open('$R/$1/rag_report.json'))['summary'];print({k:s.get(k) for k in ('n','resp_acc','n_resp_judged','ref_acc','n_ref_judged','P(resp|ref)','ret_rate','span_rate','inj_lat_s')})")"; }
( export MOSHIRAG_LLM_URL=http://localhost:8004 MCP_ROUTER_LLM_URL=http://localhost:8004 MOSHICP_ASR_URL=http://localhost:8990/v1/audio/transcriptions
  run_set web_questions "$DATA/openaudiobench/eval_datas" "$GPU_ROUTER_A" ""; score_set web_questions
  run_set math "$DATA/math_audio" "$GPU_ROUTER_A" "";                        score_set math
  run_set halueval "$DATA/halueval_audio" "$GPU_ROUTER_A" 0/2 ) &
A=$!
( export MOSHIRAG_LLM_URL=http://localhost:8006 MCP_ROUTER_LLM_URL=http://localhost:8006 MOSHICP_ASR_URL=http://localhost:8991/v1/audio/transcriptions
  run_set trivia_qa "$DATA/openaudiobench/eval_datas" "$GPU_LANE_B" "";       score_set trivia_qa
  run_set llama_questions "$DATA/openaudiobench/eval_datas" "$GPU_LANE_B" ""; score_set llama_questions
  run_set halueval "$DATA/halueval_audio" "$GPU_LANE_B" 1/2 ) &
B=$!
wait $A $B; score_set halueval
# Full-Duplex-Bench v1 (all clips, official scorers) and v3, on lane B's GPU with router B
export MCP_ROUTER_LLM_URL=http://localhost:8006 MOSHICP_ASR_URL=http://localhost:8991/v1/audio/transcriptions
FDB="$DATA/full_duplex_bench"; V1="$OUT/fdb_v1"; V3="$OUT/fdb_v3"
CUDA_VISIBLE_DEVICES="$GPU_LANE_B" "$PY" -m benchmark.fdb.v1_render "$FDB/v1_v1.5/dataset/data/v1.0" "$V1" --checkpoint "$CHECKPOINT" > "$LOG/fdb_v1_render.log" 2>&1 || log "FDB v1 render failed"
score_task(){ # $1 dir  $2 task  $3 asr task
  ( cd "$FDB/v1_v1.5" && CUDA_VISIBLE_DEVICES="$GPU_LANE_B" "$FDBPY" get_transcript/asr.py --root_dir "$V1/$1" --task "$3" ) >> "$LOG/fdb_v1_score.log" 2>&1
  { echo "## $1 ($2)"; ( cd "$FDB/v1_v1.5/evaluation" && "$FDBPY" evaluate.py --task "$2" --root_dir "$V1/$1" 2>>"$LOG/fdb_v1_score.log" | grep -E "Average|Mean" ); } >> "$OUT/fdb_v1_scores.md"
  log "FDB v1 $1: $(grep -A4 "## $1 " "$OUT/fdb_v1_scores.md" | grep -E "Average|Mean" | tr '\n' ' ')"; }
score_task candor_pause_handling pause_handling default; score_task synthetic_pause_handling pause_handling default
score_task candor_turn_taking smooth_turn_taking default; score_task icc_backchannel backchannel default
score_task synthetic_user_interruption user_interruption user_interruption
rm -rf "$V3" && cp -r "$FDB/v3_data/fdb_v3_data_released" "$V3"
FDB_V3_DIR="$FDB/v3" MOSHICP_TOOLPACK_ONLY=1 CUDA_VISIBLE_DEVICES="$GPU_LANE_B" "$PY" -m benchmark.fdb.v3_run "$V3" --checkpoint "$CHECKPOINT" --provider ours > "$LOG/fdb_v3.log" 2>&1 || log "FDB v3 render failed"
( cd "$FDB/v3" && "$FDBPY" evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$V3" --provider ours --output "$OUT/fdb_v3_evaluation_report.json" --use-llm > "$LOG/fdb_v3_tool_calls.log" 2>&1 \
  && "$FDBPY" evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$V3" --provider ours --output "$OUT/fdb_v3_pass_rate_report.json" --use-llm > "$LOG/fdb_v3_pass_rate.log" 2>&1 ) || log "FDB v3 scoring failed"
log "FDB v3: $(grep -E 'Tool Selection Acc|Argument Acc|Response Qual' "$LOG/fdb_v3_tool_calls.log" | tr -s ' ' | tr '\n' ' ') $(grep -E 'Pass Rate' "$LOG/fdb_v3_pass_rate.log" | tr -s ' ' | tr '\n' ' ')"
log "complete: $OUT"
