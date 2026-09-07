# Backend endpoints for the runtime and the evaluation harness. `source scripts/env.sh` after
# `scripts/backends.sh start`. Override any variable before sourcing.
export ROUTER_PORT="${ROUTER_PORT:-8004}"
export ASR_PORT="${ASR_PORT:-8990}"
export ROUTER_MODEL="${ROUTER_MODEL:-google/gemma-4-26B-A4B-it}"
# tool router (one OpenAI-compatible server serves the router, the RAG fallback and the judge)
export MCP_ROUTER_LLM_API=openai
export MCP_ROUTER_LLM_URL="http://localhost:${ROUTER_PORT}"
export MCP_ROUTER_LLM_MODEL="$ROUTER_MODEL"
# knowledge jobs: RAG fallback answers + eval judge. Same server as the router unless RAG_MODEL is set
# (then scripts/backends.sh starts it on RAG_PORT).
export RAG_MODEL="${RAG_MODEL:-}"; export RAG_PORT="${RAG_PORT:-8005}"
if [ -n "$RAG_MODEL" ]; then
  export MOSHICP_RAG_LLM_URL="http://localhost:${RAG_PORT}/v1/chat/completions"; export MOSHICP_RAG_LLM_MODEL="$RAG_MODEL"
else
  export MOSHICP_RAG_LLM_URL="http://localhost:${ROUTER_PORT}/v1/chat/completions"; export MOSHICP_RAG_LLM_MODEL="$ROUTER_MODEL"
fi
export JUDGE_LLM_URL="$MOSHICP_RAG_LLM_URL"
export JUDGE_LLM_MODEL="${JUDGE_MODEL:-$MOSHICP_RAG_LLM_MODEL}"   # eval judge; the paper runs judged with google/gemma-4-31B-it
# ASR endpoint (POST /transcribe, multipart wav -> {"text"})
export MOSHICP_ASR_URL="http://localhost:${ASR_PORT}/transcribe"
