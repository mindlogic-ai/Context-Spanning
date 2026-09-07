"""Live conversation over a WebSocket: browser microphone in, agent audio out.

    python main.py serve [--voice f0] [--host 0.0.0.0 --port 8080 --token ...]

The browser sends one 80 ms frame of float32 PCM at a time and gets one frame back, so the wire
carries the clock the model runs on. When the model emits `<ret>`, the recent user audio is
transcribed, the backend is asked, and the reference is injected on the first frame after it
returns; the stream never stops to wait for it. One engine, one conversation at a time.
"""
import asyncio
import hmac
import json
import logging
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

WEB = Path(__file__).parent / "web"
log = logging.getLogger(__name__)


async def ws_handler(request):
    token = request.app["token"]
    if token and not hmac.compare_digest(request.query.get("token", ""), token):
        raise web.HTTPUnauthorized(text="missing or wrong ?token=")
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    eng, lock = request.app["engine"], request.app["lock"]
    if lock.locked():
        await ws.send_json({"type": "busy"})
        await ws.close()
        return ws
    async with lock:
        persona, voice = request.app["persona"], request.app["voice"]
        eng.reset(); eng.set_persona(persona, voice)
        await ws.send_json({"type": "ready", "sample_rate": int(eng.mimi.sample_rate),
                            "frame_size": eng.frame_size, "persona": persona})
        loop = asyncio.get_running_loop()
        pending = None
        heard, keep = [], int(request.app["listen_s"] * eng.frame_rate)
        said, shown, user, cloning, stepped = [], "", {}, False, 0
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "reset":
                    eng.reset(); eng.set_persona(persona, voice); heard.clear(); stepped = 0
                elif m.get("type") == "clone":          # the next binary message is the recording
                    cloning = True
                elif m.get("type") == "context":
                    user = {k: m[k] for k in ("name", "location", "lat", "lon", "tz", "db") if m.get(k) not in (None, "")}
                    if "persona" in m and m["persona"] != persona:
                        if stepped:
                            await ws.send_json({"type": "error", "stage": "persona",
                                                "message": "already talking; press Stop, then Start to apply it"})
                        else:
                            persona = m["persona"]; eng.reset(); eng.set_persona(persona, voice)
                continue
            if msg.type != WSMsgType.BINARY:
                continue
            if cloning:
                cloning = False
                voice = eng.clone_voice(np.frombuffer(msg.data, dtype=np.float32), int(eng.mimi.sample_rate))
                eng.set_persona(persona, voice); stepped = 0
                await ws.send_json({"type": "cloned", "frames": int(voice.shape[1]),
                                    "seconds": round(voice.shape[1] / eng.frame_rate, 1)})
                continue
            if pending is not None and pending.done():
                ref = pending.result(); pending = None
                if isinstance(ref, dict):
                    await ws.send_json(ref)
                else:
                    n = eng.inject_context_span(ref)
                    await ws.send_json({"type": "span", "text": ref, "frames": n, "seconds": round(n / eng.frame_rate, 2)})
            frame = np.frombuffer(msg.data, dtype=np.float32).copy()
            heard.append(frame)
            if len(heard) > keep:
                del heard[:-keep]
            out = eng.step(frame)
            stepped += 1
            if out["agent_pcm"] is not None:
                await ws.send_bytes(out["agent_pcm"].astype(np.float32).tobytes())
            if out["text_token"] is not None:
                said.append(int(out["text_token"]))
                full = eng.decode_text(said)
                if full != shown:
                    await ws.send_json({"type": "text", "delta": full[len(shown):]}); shown = full
            if out["is_ret"] and pending is None:
                await ws.send_json({"type": "ret"})
                pending = loop.run_in_executor(None, _retrieve, request.app, np.concatenate(heard),
                                               int(eng.mimi.sample_rate), dict(user))
    return ws


def _retrieve(app, clip, sample_rate, user):
    """Transcribe the recent user audio and ask the backend. Returns the reference or an error event."""
    from .backend import NO_INFO
    try:
        question = app["asr"].transcribe(clip, sample_rate)
    except Exception as e:
        return {"type": "error", "stage": "asr", "message": str(e)}
    if not question:
        return {"type": "error", "stage": "asr", "message": "empty transcript"}
    ctx = {k: v for k, v in {"name": user.get("name"), "city": user.get("location"), "timezone": user.get("tz"),
                             "lat": user.get("lat"), "lon": user.get("lon")}.items() if v not in (None, "")}
    if user.get("db"):
        ctx["context_db"] = f"[context db] {user['db']}"
    try:
        ref = app["backend"].retrieve(question, ctx)
    except Exception as e:
        return {"type": "error", "stage": "retrieval", "message": str(e)}
    if not ref or ref == NO_INFO:
        return {"type": "error", "stage": "retrieval", "message": f"{NO_INFO} for: {question}"}
    return ref


async def health(request):
    return web.json_response({"status": "ready", "sample_rate": int(request.app["engine"].mimi.sample_rate)})


async def index(request):
    return web.FileResponse(WEB / "index.html")


def serve(engine, backend, asr, persona, voice, host="127.0.0.1", port=8080, token="", listen_s=12.0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = web.Application()
    app.update(engine=engine, backend=backend, asr=asr, persona=persona, voice=voice, token=token,
               listen_s=listen_s, lock=asyncio.Lock())
    app.add_routes([web.get("/", index), web.get("/health", health), web.get("/ws", ws_handler),
                    web.static("/static", WEB)])
    log.info("open http://%s:%s%s", host, port, "/?token=..." if token else "")
    web.run_app(app, host=host, port=port, print=None)
