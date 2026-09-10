"""Live conversation over a WebSocket: browser microphone in, agent audio out.

    python main.py serve [--voice f0] [--host 0.0.0.0 --port 8080 --token ...]

The browser sends one 80 ms frame of float32 PCM at a time and gets one frame back, so the wire
carries the clock the model runs on. When the model emits `<ret>`, the recent user audio is
The user's speech is transcribed continuously by utterance (`user_text` events, partial and final); on
`<ret>` the fresh transcript is the question, a sentence still running is waited for (bounded), the
backend is asked, and the reference is injected on the first frame after it returns; a turn that needs
no external knowledge injects nothing (`no_span`). The stream never stops to wait for any of it. One
engine, one conversation at a time.
"""
import asyncio
import hmac
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from .duetaspan.runtime.backend.context_db import ContextDB, ContextProfile
from .levelling import TARGET_LUFS, UserLeveller, load_enhancer, measure_lufs
from .model import load_voice
from .stream import RET_DEADLINE_S, RET_UTT_WAIT_S, UTT_CACHE_S, Utterances, retrieve_for_ret

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
        enh = request.app.get("enhancer")
        leveller = None if request.app.get("raw_user_audio") else \
            UserLeveller(sample_rate=int(eng.mimi.sample_rate), enhancer=enh() if enh else None)
        raw_tail, raw_told = [], False
        db, events = ContextDB(ContextProfile(persona=persona)), []
        # continuous utterance ASR (DuetaSpan live server): `heard` is indexed by absolute frame via `base`
        utts, base, cache, ret_wait, t_ret = Utterances(), 0, {"text": None, "t": -1e9}, None, 0.0
        sr = int(eng.mimi.sample_rate)

        def uslice(f0, f1):
            lo, hi = max(0, f0 - base), max(0, f1 + 1 - base)
            return np.concatenate(heard[lo:hi]) if hi > lo else np.zeros(0, np.float32)

        def kick(i, question=None):
            nonlocal pending, t_ret
            t_ret = time.monotonic()
            notify = lambda ev: asyncio.run_coroutine_threadsafe(ws.send_json(ev), loop)
            pending = loop.run_in_executor(None, _retrieve, request.app, uslice(max(0, i + 1 - keep), i), sr,
                                           dict(user), db, shown, list(events), notify, question)

        async def transcribe_utt(kind, u0, u1):
            nonlocal ret_wait
            text = await loop.run_in_executor(None, lambda: app_asr.transcribe(uslice(u0, u1), sr) or "")
            log.info("heard (%s, %.1fs): %s", kind, (u1 - u0) / eng.frame_rate, text.strip() or "<nothing>")
            if text.strip():
                cache.update(text=text, t=u1 / eng.frame_rate)
                if kind == "final" and text != db.last_user_text():
                    db.add_user_turn(text)
                await ws.send_json({"type": "user_text", "utt": int(u0), "final": kind == "final",
                                    "t": round(u1 / eng.frame_rate, 2), "text": text})
            if kind == "final" and ret_wait is not None:
                waiting, ret_wait = ret_wait, None
                kick(waiting, text.strip() or None)
        app_asr = request.app["asr"]
        lvl_peak = lvl_peak2 = 0.0; said_logged = 0
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "reset":
                    eng.reset(); eng.set_persona(persona, voice); heard.clear(); stepped = 0
                    db, events = ContextDB(ContextProfile(persona=persona, **user)), []
                elif m.get("type") == "clone":          # the next binary message is the recording
                    cloning = True
                elif m.get("type") == "voice":          # one of the released voices, before the conversation
                    if stepped:
                        await ws.send_json({"type": "error", "stage": "voice",
                                            "message": "already talking; press Stop, then choose the voice"})
                    else:
                        name = str(m.get("name") or "f0")
                        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):      # a released voice name, never a path
                            await ws.send_json({"type": "error", "stage": "voice", "message": "unknown voice"}); continue
                        try:
                            voice = load_voice(name)
                            eng.reset(); eng.set_persona(persona, voice)
                            await ws.send_json({"type": "voice", "name": m.get("name"), "frames": int(voice.shape[1])})
                        except Exception as e:
                            await ws.send_json({"type": "error", "stage": "voice", "message": str(e)})
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
                took = time.monotonic() - t_ret
                if r.get("error"):
                    await ws.send_json({"type": "error", "stage": r["error"], "message": r["message"]})
                elif r["inject"] is not None and took > RET_DEADLINE_S:   # too late to be the answer's ground
                    r["inject"] = None; r["late"] = True; events.append(r)
                    await ws.send_json({"type": "late", "question": r["question"], "seconds": round(took, 2),
                                        "source": r["src"]})
                elif r["inject"] is None:      # nothing to ground on: no span, and not a failure
                    events.append(r)
                    await ws.send_json({"type": "no_span", "question": r["question"]})
                else:
                    n = eng.inject_context_span(r["inject"])
                    events.append(r)
                    await ws.send_json({"type": "span", "text": r["inject"], "question": r["question"],
                                        "source": r["src"], "frames": n, "seconds": round(n / eng.frame_rate, 2)})
            frame = np.frombuffer(msg.data, dtype=np.float32).copy()
            # The model is conditioned on this audio and the fine-tune's user channel sits in
            # a narrow band, so a quiet microphone is off-distribution input rather than a
            # merely faint one. The ASR hides that: it keeps transcribing while the model
            # stops reacting (#18). Level it here, before both consumers.
            raw_peak = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
            if leveller is not None:
                raw_tail.append(frame)
                frame = leveller.process(frame)
            lvl_peak = max(lvl_peak, raw_peak); lvl_peak2 = max(lvl_peak2, float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0)
            if stepped and stepped % 63 == 0:      # every ~5 s: loudest frame the user sent, raw and levelled, dBFS RMS
                log.info("user audio: peak frame raw %.1f dBFS, levelled %.1f dBFS, gain x%.1f", 20 * np.log10(max(lvl_peak, 1e-9)),
                         20 * np.log10(max(lvl_peak2, 1e-9)), getattr(leveller, "gain", 1.0))
                lvl_peak = lvl_peak2 = 0.0
                # Once, after ~4 s of raw audio: the loudness the microphone actually delivered,
                # by the same meter the training data was measured with.
                if not raw_told and len(raw_tail) >= 50:
                    raw_told = True
                    lufs = measure_lufs(np.concatenate(raw_tail), int(eng.mimi.sample_rate))
                    if np.isfinite(lufs):
                        await ws.send_json({"type": "mic_level", "lufs": round(lufs, 1),
                                            "target": TARGET_LUFS,
                                            "quiet": bool(lufs < TARGET_LUFS - 10)})
                    raw_tail.clear()
            heard.append(frame)
            if len(heard) > 2 * keep:
                del heard[:keep]; base += keep
            for kind, u0, u1 in utts.feed(stepped, frame):
                asyncio.ensure_future(transcribe_utt(kind, u0, u1))
            if ret_wait is not None and (stepped - ret_wait) / eng.frame_rate >= RET_UTT_WAIT_S:
                waiting, ret_wait = ret_wait, None
                kick(waiting)                           # the sentence did not end in time: transcribe the window
            out = eng.step(frame)
            stepped += 1
            if out["agent_pcm"] is not None:
                await ws.send_bytes(out["agent_pcm"].astype(np.float32).tobytes())
            if out["text_token"] is not None:
                said.append(int(out["text_token"]))
                full = eng.decode_text(said)
                if full != shown:
                    await ws.send_json({"type": "text", "delta": full[len(shown):]}); shown = full
                    if full[-1:] in ".?!" and len(full) > said_logged:
                        log.info("agent: %s", full[said_logged:].strip()); said_logged = len(full)
            if out["is_ret"] and pending is None and ret_wait is None:
                await ws.send_json({"type": "ret"})
                i = stepped - 1
                if cache["text"] and (i / eng.frame_rate - cache["t"]) < UTT_CACHE_S and not utts.speaking:
                    kick(i, cache["text"])              # the sentence just ended: its transcript is the question
                elif utts.speaking:
                    ret_wait = i                        # mid-sentence: wait for the utterance to end (bounded)
                else:
                    kick(i)                             # no utterance around: the fixed window
    return ws


def _retrieve(app, clip, sample_rate, user, db, said_text, events, notify, question=None):
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
                             on_question=on_question, question=question)
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


def serve(engine, backend, asr, persona, voice, host="127.0.0.1", port=8080, token="",
          listen_s=12.0, raw_user_audio=False, enhance=""):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = web.Application()
    app.update(engine=engine, backend=backend, asr=asr, persona=persona, voice=voice, token=token,
               listen_s=listen_s, raw_user_audio=raw_user_audio,
               enhancer=load_enhancer(enhance) if enhance else None, lock=asyncio.Lock())
    app.add_routes([web.get("/", index), web.get("/health", health), web.get("/ws", ws_handler),
                    web.static("/static", WEB)])
    log.info("open http://%s:%s%s", host, port, "/?token=..." if token else "")
    web.run_app(app, host=host, port=port, print=None)
