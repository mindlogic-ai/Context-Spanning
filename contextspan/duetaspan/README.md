# DuetaSpan runtime (backend)

The backend the speech model talks to on `<ret>`. It is the DuetaSpan runtime, kept under its own
package name because its module paths are part of the deployment contract (`scripts/backends.sh`,
the `DUETASPAN_*` / `MOSHICP_*` environment variables, external tooling).

```
align/asr.py               ASR client: POST /transcribe endpoint first, local whisper fallbacks
common/paths.py            every on-disk location, resolved from DUETASPAN_* variables
runtime/asr_server.py      Qwen3-ASR HTTP server (the endpoint the client prefers)
runtime/backend/
  realtime.py              RealtimeBackend: the retrieve() the engine calls; router first, LLM-RAG second
  retrieve.py              the reference-string contract and the MCP-intent words
  context_db.py            ContextProfile (name, city, timezone, notes) + append-only conversation log
runtime/mcp/
  client.py                MCP client + LLM tool router: two-stage pick, argument fill, one tool per <ret>
  registry.py              one dispatch table for the 125 bank tools (world / maps / fs / live / unsupported)
  world.py, build_world.py SQLite world seeded from Google SGD service results; transactions are real rows
  geo_index.py             local gazetteer + POI index (GeoNames KR, Seoul OSM)
  adapters_*.py            keyless real backends: maps (Nominatim/OSRM/open-meteo), search (DuckDuckGo),
                           media (iTunes), fs (sandboxed), browser (Playwright)
  api_backend.py, gemini_backend.py   single-model routers over hosted APIs (optional)
  cache.py                 disk cache for the free HTTP sources
  coverage.py              call every bank tool once and report
  servers/                 the MCP servers: bank, time, weather, finance, websearch
eval/mcp_toolbank.py       250 real tool calls scored against the world
```

Configuration and the rationale for the defaults: [`docs/BACKENDS.md`](../../docs/BACKENDS.md). Data
(the tool bank, the world, the geo index) ships in `contextspan/datasets/moshicp/`.
