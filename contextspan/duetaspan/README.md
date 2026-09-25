# DuetaSpan runtime (backend)

The backend the speech model talks to on `<ret>`. It is the DuetaSpan runtime, kept under its own
package name because its module paths are part of the deployment contract (`scripts/backends.sh`,
the `DUETASPAN_*` / `MOSHICP_*` environment variables, external tooling).

```
align/asr.py               ASR client: posts audio to the Qwen3-ASR endpoint (vLLM qwen-asr-serve or /transcribe)
common/paths.py            the on-disk locations, resolved from DUETASPAN_* variables
runtime/asr_server.py      Qwen3-ASR plain HTTP server (transformers; scripts/backends.sh ASR_BACKEND=transformers)
runtime/backend/
  realtime.py              RealtimeBackend: the retrieve() the engine calls; the tool router is the single decider
  context_db.py            ContextProfile (name, city, timezone, notes) + append-only conversation log
runtime/mcp/
  client.py                MCP client + LLM tool router: two-stage pick, argument fill, one tool per <ret>
  registry.py              one dispatch table for the bank's tools (world / maps / live / unsupported)
  world.py, build_world.py SQLite world seeded from Google SGD service results; transactions are real rows
  geo_index.py             local gazetteer + POI index (GeoNames KR, Seoul OSM)
  adapters_*.py            keyless real backends: maps (Nominatim/OSRM/open-meteo), search (DuckDuckGo),
                           media (iTunes)
  cache.py                 disk cache for the free HTTP sources
  servers/                 the MCP servers: time, weather, finance, websearch
```

Configuration and the rationale for the defaults: [`docs/BACKENDS.md`](../../docs/BACKENDS.md). Data
(the tool bank, the world, the geo index) ships in `contextspan/datasets/moshicp/`.
