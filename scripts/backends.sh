#!/usr/bin/env bash
# Start / stop / check the two backend servers with the settings the benchmarks were run with, so a
# fresh machine gets the same span latency without tuning anything:
#   bash scripts/backends.sh start     # router LLM (vLLM) + ASR (Qwen3-ASR), waits until both answer
#   bash scripts/backends.sh status
#   bash scripts/backends.sh stop
# Router model: Gemma-4-26B-A4B (26B MoE, 4B active) — the router's decision is a classification plus a
# few argument strings and its cost is decode, so the 4B-active model answers in roughly a third of the
# time of a dense 27B at the same quality on the tool-selection probes; one 80-96 GB GPU at TP=1.
# The same server also answers the RAG fallback (knowledge questions the tools cannot) and serves as the eval
# judge unless RAG_MODEL points those at a second server. Footprint on one 96 GB GPU with everything on it:
# router ~56 GB (ROUTER_MEM 0.58) + ASR ~5 GB + speech model ~20 GB = ~81 GB.
# GPUs: ROUTER_GPUS (default 1), ASR_GPU (default 0, ~4 GB). ROUTER_MODEL / ROUTER_TP override.
#
# Why these vLLM flags (measured on RTX PRO 6000 Blackwell, 2026-09):
#   --max-model-len 8192           the router prompt carries the tool catalogue (~2.9k tokens at the first
#                                  stage, ~3.5k for the largest tool group) plus the conversation; 4k truncates.
#   --gpu-memory-utilization ROUTER_MEM (0.58, ~56 GB) leaves room on the same GPU for the ASR and the speech model.
#   --speculative-config ngram     the router's replies repeat the prompt (tool names, argument values,
#                                  the sentence it copies from the Context DB): prompt-lookup speculation
#                                  cut the router round trip by ~40% at temperature 0 with no output change.
#   --enable-prefix-caching        every call shares the same 10k-char system prompt + catalogue prefix.
set -euo pipefail
CMD="${1:-status}"
ROUTER_PORT="${ROUTER_PORT:-8004}"; ASR_PORT="${ASR_PORT:-8990}"
ROUTER_MODEL="${ROUTER_MODEL:-google/gemma-4-26B-A4B-it}"
ROUTER_GPUS="${ROUTER_GPUS:-1}"; ROUTER_TP="${ROUTER_TP:-1}"; ASR_GPU="${ASR_GPU:-0}"
ROUTER_MEM="${ROUTER_MEM:-0.58}"          # fraction of the GPU: ~56 GB on a 96 GB card = weights (~52 GB bf16) + KV
# Optional second server for the knowledge jobs (RAG fallback answers, eval judge) so the router model can be
# small: set RAG_MODEL (and RAG_GPUS / RAG_PORT). Unset = the router server does all of it (default).
RAG_MODEL="${RAG_MODEL:-}"; RAG_PORT="${RAG_PORT:-8005}"; RAG_GPUS="${RAG_GPUS:-$ROUTER_GPUS}"; RAG_TP="${RAG_TP:-1}"; RAG_MEM="${RAG_MEM:-0.30}"
LOG="${BACKEND_LOG_DIR:-/tmp/contextspan_backends}"; mkdir -p "$LOG"
PY="${PYTHON:-python}"

router_up() { curl -s -m 5 "http://localhost:${ROUTER_PORT}/v1/models" 2>/dev/null | grep -q "$ROUTER_MODEL"; }
asr_up()    { curl -s -m 5 "http://localhost:${ASR_PORT}/health" 2>/dev/null | grep -q -i ok; }

case "$CMD" in
  start)
    if router_up; then echo "[backends] router already up on :$ROUTER_PORT"; else
      CUDA_VISIBLE_DEVICES="$ROUTER_GPUS" setsid nohup "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$ROUTER_MODEL" --served-model-name "$ROUTER_MODEL" --port "$ROUTER_PORT" \
        --tensor-parallel-size "$ROUTER_TP" --gpu-memory-utilization "$ROUTER_MEM" --max-model-len 8192 \
        --enable-prefix-caching --trust-remote-code \
        --speculative-config '{"method":"ngram","num_speculative_tokens":6,"prompt_lookup_max":5,"prompt_lookup_min":2}' \
        > "$LOG/router.log" 2>&1 < /dev/null &
      echo "[backends] router starting (log $LOG/router.log)"
    fi
    if [ -n "$RAG_MODEL" ]; then
      if curl -s -m 5 "http://localhost:${RAG_PORT}/v1/models" 2>/dev/null | grep -q "$RAG_MODEL"; then echo "[backends] RAG/judge server already up on :$RAG_PORT"; else
        CUDA_VISIBLE_DEVICES="$RAG_GPUS" setsid nohup "$PY" -m vllm.entrypoints.openai.api_server \
          --model "$RAG_MODEL" --served-model-name "$RAG_MODEL" --port "$RAG_PORT" \
          --tensor-parallel-size "$RAG_TP" --gpu-memory-utilization "$RAG_MEM" --max-model-len 8192 \
          --enable-prefix-caching --trust-remote-code > "$LOG/rag.log" 2>&1 < /dev/null &
        echo "[backends] RAG/judge server starting (log $LOG/rag.log)"
      fi
    fi
    if asr_up; then echo "[backends] ASR already up on :$ASR_PORT"; else
      CUDA_VISIBLE_DEVICES="$ASR_GPU" ASR_PORT="$ASR_PORT" setsid nohup "$PY" -m contextspan.duetaspan.runtime.asr_server \
        > "$LOG/asr.log" 2>&1 < /dev/null &
      echo "[backends] ASR starting (log $LOG/asr.log)"
    fi
    for i in $(seq 1 120); do router_up && asr_up && break; sleep 10; done
    router_up && echo "[backends] router ready" || { echo "[backends] router NOT ready — see $LOG/router.log"; exit 1; }
    asr_up && echo "[backends] ASR ready" || { echo "[backends] ASR NOT ready — see $LOG/asr.log"; exit 1; }
    if [ -n "$RAG_MODEL" ]; then for i in $(seq 1 120); do curl -s -m 5 "http://localhost:${RAG_PORT}/v1/models" 2>/dev/null | grep -q "$RAG_MODEL" && break; sleep 10; done; echo "[backends] RAG/judge server $(curl -s -m 5 http://localhost:${RAG_PORT}/v1/models | grep -q "$RAG_MODEL" && echo ready || echo 'NOT ready')"; fi
    echo "[backends] now: source scripts/env.sh" ;;
  stop)
    pkill -f "vllm.entrypoints.openai.api_server --model $ROUTER_MODEL" 2>/dev/null && echo "[backends] router stopped" || true
    [ -n "$RAG_MODEL" ] && pkill -f "vllm.entrypoints.openai.api_server --model $RAG_MODEL" 2>/dev/null && echo "[backends] RAG/judge server stopped" || true
    pkill -f "contextspan.duetaspan.runtime.asr_server" 2>/dev/null && echo "[backends] ASR stopped" || true ;;
  status)
    router_up && echo "router :$ROUTER_PORT up" || echo "router :$ROUTER_PORT down"
    asr_up && echo "ASR :$ASR_PORT up" || echo "ASR :$ASR_PORT down" ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
