# Backend endpoints for the runtime and the benchmark. `source scripts/env.sh` after
# `scripts/backends.sh start`. Override any variable before sourcing.
export ROUTER_PORT="${ROUTER_PORT:-8004}"
export ASR_PORT="${ASR_PORT:-8990}"
export ROUTER_MODEL="${ROUTER_MODEL:-google/gemma-4-26B-A4B-it}"
# tool router (one OpenAI-compatible server serves the router, the question-rewrite LLM and the judge)
export MCP_ROUTER_LLM_URL="http://localhost:${ROUTER_PORT}"
export MCP_ROUTER_LLM_MODEL="$ROUTER_MODEL"
# knowledge jobs: question rewriting + benchmark judge. Same server as the router unless RAG_MODEL is set
# (then scripts/backends.sh starts it on RAG_PORT).
export RAG_MODEL="${RAG_MODEL:-}"; export RAG_PORT="${RAG_PORT:-8005}"
if [ -n "$RAG_MODEL" ]; then
  export MOSHICP_RAG_LLM_URL="http://localhost:${RAG_PORT}/v1/chat/completions"; export MOSHICP_RAG_LLM_MODEL="$RAG_MODEL"
else
  export MOSHICP_RAG_LLM_URL="http://localhost:${ROUTER_PORT}/v1/chat/completions"; export MOSHICP_RAG_LLM_MODEL="$ROUTER_MODEL"
fi
export JUDGE_LLM_URL="$MOSHICP_RAG_LLM_URL"
export JUDGE_LLM_MODEL="${JUDGE_MODEL:-$MOSHICP_RAG_LLM_MODEL}"   # benchmark judge
# ASR endpoint. ASR_BACKEND=vllm (default, qwen-asr-serve): OpenAI audio API, `model` = MOSHICP_ASR_MODEL.
# ASR_BACKEND=transformers: POST /transcribe, multipart wav -> {"text"}.
export ASR_BACKEND="${ASR_BACKEND:-vllm}"
if [ "$ASR_BACKEND" = "vllm" ]; then
  export MOSHICP_ASR_URL="http://localhost:${ASR_PORT}/v1/audio/transcriptions"; export MOSHICP_ASR_MODEL="${MOSHICP_ASR_MODEL:-qwen3-asr}"
else
  export MOSHICP_ASR_URL="http://localhost:${ASR_PORT}/transcribe"
fi
