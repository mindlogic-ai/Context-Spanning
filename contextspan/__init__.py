"""Context Spanning: a full-duplex speech model that reads external knowledge as Context Spans.

  model/      the speech model: weights, streaming engine, the Context Span block, sequence convention
  runtime/    the live loop around it: frame-clock stream for a wav, WebSocket server + browser page
  training/   data preparation and fine-tuning
  duetaspan/  the backend runtime: tool router over the tool bank + MCP servers, LLM-RAG, Context DB, ASR client
  datasets/   the shipped tool bank and its SQLite world
  moshi/      vendored PersonaPlex fork of Kyutai's `moshi` (third-party, MIT)
"""
__version__ = "1.0.0"
