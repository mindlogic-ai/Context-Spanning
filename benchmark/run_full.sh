#!/bin/bash
# The full benchmark protocol of benchmark/README.md, end to end: servers, the five RAG sets whole on two lanes,
# scoring with the moshi-rag judges, then Full-Duplex-Bench v1 (all clips, official scorers) and v3 (all scenarios).
# There is no partial run: every set is rendered whole and a lane resumes from what is already rendered.
#
#   CHECKPOINT=/path/context_spanning_7b.pt DATA=/data OUT=runs/full bash benchmark/run_full.sh
#
# DATA holds halueval_audio/ (1000), math_audio/ (3,822, built by benchmark/math/, README.md section 3),
# openaudiobench/eval_datas/ (TriviaQA 1000, WebQuestions 1000, LlamaQuestions 300), full_duplex_bench/ (the benchmark
# clone with v1_v1.5/ and v3/, plus v3_data/fdb_v3_data_released). The models are the protocol's and are not arguments
# (ASR_MODEL may point at a local copy of Qwen/Qwen3-ASR-1.7B). OPENAI_API_KEY for the OpenAudioBench judge, the FDB
# interruption judge and the v3 evaluators.
# GPU layout (README.md section 5): a router GPU holds one router and nothing else, a lane GPU holds one engine and
# its ASR, a router serves one lane. GPU_LANE_A + GPU_ROUTER_A, GPU_LANE_B + GPU_ROUTER_B; the judge takes
# GPU_ROUTER_B once rendering is over.
set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT}"; : "${DATA:?set DATA}"; : "${OUT:?set OUT}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
ROUTER_MODEL=google/gemma-4-26B-A4B-it; JUDGE_MODEL=google/gemma-3-27b-it; ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
GPU_LANE_A="${GPU_LANE_A:-0}"; GPU_ROUTER_A="${GPU_ROUTER_A:-1}"; GPU_ROUTER_B="${GPU_ROUTER_B:-2}"; GPU_LANE_B="${GPU_LANE_B:-3}"
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
warm(){ curl -s -m 120 "localhost:$1/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" > /dev/null; }
stop_port(){ # $1 port: the vLLM server listening there and its workers
  for p in $(ps -eo pid,args | grep -F -- "--port $1" | grep -F api_server | grep -v grep | awk '{print $1}'); do pkill -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done; sleep 20; }
log "start: $CHECKPOINT; router $ROUTER_MODEL; judge $JUDGE_MODEL; ASR $ASR_MODEL; commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
up_llm 8004 "$ROUTER_MODEL" || serve "$GPU_ROUTER_A" "$ROUTER_MODEL" 8004 0.70 routerA
up_llm 8006 "$ROUTER_MODEL" || serve "$GPU_ROUTER_B" "$ROUTER_MODEL" 8006 0.70 routerB
up_asr 8990 || asr "$GPU_LANE_A" 8990; up_asr 8991 || asr "$GPU_LANE_B" 8991
for i in $(seq 1 120); do up_llm 8004 "$ROUTER_MODEL" && up_llm 8006 "$ROUTER_MODEL" && up_asr 8990 && up_asr 8991 && break; sleep 10; done
up_llm 8004 "$ROUTER_MODEL" && up_llm 8006 "$ROUTER_MODEL" && up_asr 8990 && up_asr 8991 || { log "servers not ready, see $LOG"; exit 2; }
warm 8004 "$ROUTER_MODEL"; warm 8006 "$ROUTER_MODEL"
log "routers and ASR up and warm"
export MCP_ROUTER_LLM_MODEL="$ROUTER_MODEL" MOSHIRAG_LLM_MODEL="$ROUTER_MODEL" MOSHICP_ASR_MODEL=qwen3-asr
R="$OUT/rag"; mkdir -p "$R"
SETS="web_questions trivia_qa llama_questions halueval math"
root_of(){ case "$1" in halueval) echo "$DATA/halueval_audio";; math) echo "$DATA/math_audio";; *) echo "$DATA/openaudiobench/eval_datas";; esac; }
run_set(){ # $1 set  $2 gpu  $3 shard K/N
  CUDA_VISIBLE_DEVICES="$2" "$PY" -m benchmark.rag.run "$1" "$(root_of "$1")" "$R/$1" --checkpoint "$CHECKPOINT" --shard "$3" \
    > "$LOG/rag_$1_shard${3%/*}.log" 2>&1 || log "$1 run failed (gpu $2, shard $3)"
  log "$1 shard $3 rendered (gpu $2); slow items: $(grep -c 'WARNING' "$LOG/rag_$1_shard${3%/*}.log" || true)"; }
score_set(){ # $1 set
  "$PY" -m benchmark.rag.score "$R/$1" > "$LOG/rag_$1_score.log" 2>&1 || log "$1 scoring failed"
  log "$1: $("$PY" -c "import json;s=json.load(open('$R/$1/rag_report.json'))['summary'];print({k:s.get(k) for k in ('n','ref_acc','resp_acc','P(resp|ref)','n_ref_correct','ret_rate','span_rate','inj_lat_s')})")"; }
# both lanes render every set: lane A the even items with router A, lane B the odd items with router B
( export MOSHIRAG_LLM_URL=http://localhost:8004 MCP_ROUTER_LLM_URL=http://localhost:8004 MOSHICP_ASR_URL=http://localhost:8990/v1/audio/transcriptions
  for s in $SETS; do run_set "$s" "$GPU_LANE_A" 0/2; done ) &
A=$!
( export MOSHIRAG_LLM_URL=http://localhost:8006 MCP_ROUTER_LLM_URL=http://localhost:8006 MOSHICP_ASR_URL=http://localhost:8991/v1/audio/transcriptions
  for s in $SETS; do run_set "$s" "$GPU_LANE_B" 1/2; done ) &
B=$!
wait $A $B
log "RAG rendering complete; router B -> judge"
stop_port 8006
serve "$GPU_ROUTER_B" "$JUDGE_MODEL" 8007 0.70 judge
for i in $(seq 1 120); do up_llm 8007 "$JUDGE_MODEL" && break; sleep 10; done
up_llm 8007 "$JUDGE_MODEL" || { log "judge not ready, see $LOG"; exit 2; }
warm 8007 "$JUDGE_MODEL"
export MOSHIRAG_GEMMA_JUDGE_URL=http://localhost:8007
( for s in $SETS; do score_set "$s"; done; log "RAG scoring complete" ) &
SCORE=$!
# Full-Duplex-Bench v1 (all clips, official scorers) and v3 (all scenarios) on lane A with router A
export MCP_ROUTER_LLM_URL=http://localhost:8004 MOSHICP_ASR_URL=http://localhost:8990/v1/audio/transcriptions
FDB="$DATA/full_duplex_bench"; V1="$OUT/fdb_v1"; V3="$OUT/fdb_v3"
CUDA_VISIBLE_DEVICES="$GPU_LANE_A" "$PY" -m benchmark.fdb.v1_render "$FDB/v1_v1.5/dataset/data/v1.0" "$V1" --checkpoint "$CHECKPOINT" > "$LOG/fdb_v1_render.log" 2>&1 || log "FDB v1 render failed"
log "FDB v1 rendered"
score_task(){ # $1 dir  $2 task  $3 asr task
  ( cd "$FDB/v1_v1.5" && CUDA_VISIBLE_DEVICES="$GPU_LANE_B" "$FDBPY" get_transcript/asr.py --root_dir "$V1/$1" --task "$3" ) >> "$LOG/fdb_v1_score.log" 2>&1
  { echo "## $1 ($2)"; ( cd "$FDB/v1_v1.5/evaluation" && "$FDBPY" evaluate.py --task "$2" --root_dir "$V1/$1" 2>>"$LOG/fdb_v1_score.log" | grep -E "Average|Mean" ); } >> "$OUT/fdb_v1_scores.md"
  log "FDB v1 $1: $(grep -A4 "## $1 " "$OUT/fdb_v1_scores.md" | grep -E "Average|Mean" | tr '\n' ' ')"; }
( score_task candor_pause_handling pause_handling default; score_task synthetic_pause_handling pause_handling default
  score_task candor_turn_taking smooth_turn_taking default; score_task icc_backchannel backchannel default
  score_task synthetic_user_interruption user_interruption user_interruption; log "FDB v1 scoring complete" ) &
V1SCORE=$!
rm -rf "$V3" && cp -r "$FDB/v3_data/fdb_v3_data_released" "$V3"
FDB_V3_DIR="$FDB/v3" CUDA_VISIBLE_DEVICES="$GPU_LANE_A" "$PY" -m benchmark.fdb.v3_run "$V3" --checkpoint "$CHECKPOINT" --provider ours > "$LOG/fdb_v3.log" 2>&1 || log "FDB v3 render failed"
log "FDB v3 rendered"
( cd "$FDB/v3" && "$FDBPY" evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$V3" --provider ours --output "$OUT/fdb_v3_evaluation_report.json" --use-llm > "$LOG/fdb_v3_tool_calls.log" 2>&1 \
  && "$FDBPY" evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$V3" --provider ours --output "$OUT/fdb_v3_pass_rate_report.json" --use-llm > "$LOG/fdb_v3_pass_rate.log" 2>&1 ) || log "FDB v3 scoring failed"
log "FDB v3: $(grep -E 'Tool Selection Acc|Argument Acc|Response Qual' "$LOG/fdb_v3_tool_calls.log" | tr -s ' ' | tr '\n' ' ') $(grep -E 'Pass Rate' "$LOG/fdb_v3_pass_rate.log" | tr -s ' ' | tr '\n' ' ')"
wait $SCORE $V1SCORE
log "complete: $OUT"
