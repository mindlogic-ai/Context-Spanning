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
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from .duetaspan.runtime.backend.context_db import ContextDB, ContextProfile
from .stream import retrieve_for_ret

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
        db, events = ContextDB(ContextProfile(persona=persona)), []
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "reset":
                    eng.reset(); eng.set_persona(persona, voice); heard.clear(); stepped = 0
                    db, events = ContextDB(ContextProfile(persona=persona, **user)), []
                elif m.get("type") == "clone":          # the next binary message is the recording
                    cloning = True
                elif m.get("type") == "context":
                    # page fields -> Context DB profile (duetaspan ContextProfile): the "Knowledge" text
                    # is the profile's notes, so the router sees it in the working text.
                    keys = {"name": "name", "location": "city", "lat": "lat", "lon": "lon", "tz": "timezone", "db": "notes"}
                    user = {keys[k]: m[k] for k in keys if m.get(k) not in (None, "")}
                    db = ContextDB(ContextProfile(persona=m.get("persona") or persona, **user))
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
                r = pending.result(); pending = None
                if r.get("error"):
                    await ws.send_json({"type": "error", "stage": r["error"], "message": r["message"]})
                else:
                    n = eng.inject_context_span(r["inject"])
                    events.append(r)
                    await ws.send_json({"type": "span", "text": r["inject"], "question": r["question"],
                                        "source": r["src"], "frames": n, "seconds": round(n / eng.frame_rate, 2)})
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
                notify = lambda ev: asyncio.run_coroutine_threadsafe(ws.send_json(ev), loop)
                pending = loop.run_in_executor(None, _retrieve, request.app, np.concatenate(heard),
                                               int(eng.mimi.sample_rate), dict(user), db, shown, list(events), notify)
    return ws


def _retrieve(app, clip, sample_rate, user, db, said_text, events, notify):
    """One `<ret>` off the frame clock: the DuetaSpan handler (stream.retrieve_for_ret) with the page
    told what the ASR heard as soon as it is known, and question/reference logged once per `<ret>`."""
    t0 = time.monotonic()
    timing = {}

    def on_question(q):
        timing["asr_ms"] = int((time.monotonic() - t0) * 1000)
        log.info("question (%d ms asr): %s", timing["asr_ms"], q)
        if q.strip():
            notify({"type": "question", "text": q, "ms": timing["asr_ms"]})
    try:
        r = retrieve_for_ret(app["backend"], app["asr"], clip, sample_rate, user, db, said_text, events,
                             on_question=on_question)
    except Exception as e:
        log.exception("retrieval failed")
        return {"error": "retrieval", "message": str(e)}
    if r["reference"] is not None:
        log.info("reference (%d ms total, src=%s): %s", int((time.monotonic() - t0) * 1000), r["src"], r["reference"])
    return r


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
