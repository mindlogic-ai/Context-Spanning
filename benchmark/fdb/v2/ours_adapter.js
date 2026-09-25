#!/usr/bin/env node
/**
 * ours_adapter.js (Role A/B) — Context Spanning examinee adapter for FDB-v2.
 *
 * Bridges orchestrator (WebRTC 48 kHz PCM16, 10 ms sink events) <->
 * `python main.py serve` WS (raw Float32 mono 24 kHz, engine consumes 1920-sample
 * frames = 80 ms @ 24 kHz).
 *
 * Mirrors adapters/moshi_adapter.js structure. Differences:
 *  - No Opus/Ogg: the server speaks raw f32 PCM over WS (binary), JSON control (text).
 *  - Handshake: send {"type":"start","persona":...}; wait for {"type":"ready"}.
 *  - Uplink pacer ships one 80 ms frame per tick, silence when the buffer is
 *    empty — the frame clock NEVER stops (server steps the engine only on
 *    received frames, so continuous uplink == continuous realtime clock).
 */

require('dotenv').config();
const WebSocket = require('ws');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Audio configuration
const ORCH_SR = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;
const OURS_SR = parseInt(process.env.OURS_SAMPLE_RATE, 10) || 24000;
const ORCH_10MS_SMP = Math.floor(ORCH_SR * 0.01);      // 480
const ORCH_10MS_BYTES = ORCH_10MS_SMP * 2;             // 960
const OURS_FRAME_SMP = parseInt(process.env.OURS_FRAME_SAMPLES, 10) || 1920; // 80 ms @ 24k
const FRAME_MS = Math.round(OURS_FRAME_SMP / OURS_SR * 1000);                // 80
const UPLINK_48K_BYTES = OURS_FRAME_SMP * 2 /*24k->48k*/ * 2 /*i16*/;        // 7680

const DEFAULT_OURS_URL = process.env.OURS_WS_URL || 'ws://localhost:8988/ws';
const DEFAULT_PERSONA = process.env.OURS_PERSONA
  || process.env.EXAMINEE_SYSTEM_PROMPT
  || 'You are a helpful AI assistant. Always speak in English.';

// ── PCM utilities (identical math to moshi_adapter.js) ──────────────────────
function i16ToF32(buf) {
  const out = new Float32Array(buf.length / 2);
  for (let i = 0, j = 0; i < buf.length; i += 2, j++)
    out[j] = Math.max(-1, Math.min(1, buf.readInt16LE(i) / 32768));
  return out;
}
function f32ToI16(f) {
  const out = Buffer.alloc(f.length * 2);
  for (let i = 0; i < f.length; i++)
    out.writeInt16LE((Math.max(-1, Math.min(1, f[i])) * 32768) | 0, i * 2);
  return out;
}
function ds48kTo24k(f) {
  const n = f.length & ~1;
  const out = new Float32Array(n / 2);
  for (let i = 0, j = 0; i < n; i += 2, j++) out[j] = 0.5 * (f[i] + f[i + 1]);
  return out;
}
function us24kTo48k(f) {
  if (f.length === 0) return new Float32Array(0);
  const out = new Float32Array(f.length * 2);
  for (let i = 0, j = 0; i < f.length - 1; i++, j += 2) {
    out[j] = f[i];
    out[j + 1] = 0.5 * (f[i] + f[i + 1]);
  }
  out[out.length - 2] = f[f.length - 1];
  out[out.length - 1] = f[f.length - 1];
  return out;
}

// ── OursClient: WS leg to `python main.py serve` ────────────────────────────
// Usable standalone (smoke tests) or from the adapter CLI below.
class OursClient {
  /**
   * @param {object} o {url, persona, voicePreset, city, tz, log}
   * Callbacks: onAgent24k(Float32Array), onText(obj), onError(msg)
   */
  constructor(o = {}) {
    this.url = o.url || DEFAULT_OURS_URL;
    this.persona = o.persona || DEFAULT_PERSONA;
    this.voicePreset = o.voicePreset || process.env.OURS_VOICE_PRESET || null;
    this.city = o.city || process.env.OURS_CITY || null;
    this.tz = o.tz || process.env.OURS_TZ || null;
    this.log = o.log !== false;
    this.ws = null;
    this.ready = false;
    this.uplinkBuf48k = Buffer.alloc(0);
    this.pacer = null;
    this.framesSent = 0;
    this.framesRecv = 0;
    this.samplesRecv = 0;
    this.onAgent24k = o.onAgent24k || (() => {});
    this.onText = o.onText || (() => {});
    this.onError = o.onError || ((m) => { throw new Error(m); });
  }

  async connect(readyTimeoutMs = 60000) {
    this.ws = new WebSocket(this.url);
    await new Promise((res, rej) => {
      this.ws.once('open', res);
      this.ws.once('error', rej);
    });
    this.ws.on('message', (msg, isBinary) => this._onMessage(msg, isBinary));
    // Handshake: start -> ready (persona/voice load can take seconds)
    const start = { type: 'start', persona: this.persona };
    if (this.voicePreset) start.voice_preset = this.voicePreset;
    if (this.city) start.city = this.city;
    if (this.tz) start.tz = this.tz;
    this.ws.send(JSON.stringify(start));
    await new Promise((res, rej) => {
      const to = setTimeout(() => rej(new Error(`no 'ready' within ${readyTimeoutMs}ms`)), readyTimeoutMs);
      this._readyResolve = () => { clearTimeout(to); res(); };
      this._readyReject = (e) => { clearTimeout(to); rej(e); };
    });
    // Drop audio buffered while persona was loading: stream starts NOW.
    this.uplinkBuf48k = Buffer.alloc(0);
  }

  _onMessage(msg, isBinary) {
    if (isBinary) {
      // Raw f32 mono 24 kHz agent PCM frame
      const buf = Buffer.isBuffer(msg) ? msg : Buffer.from(msg);
      const n = Math.floor(buf.length / 4);
      if (n === 0) return;
      const f24 = new Float32Array(n);
      for (let i = 0; i < n; i++) f24[i] = buf.readFloatLE(i * 4);
      this.framesRecv += 1;
      this.samplesRecv += n;
      this.onAgent24k(f24);
      return;
    }
    let m;
    try { m = JSON.parse(msg.toString('utf8')); } catch { return; }
    if (m.type === 'ready') {
      this.ready = true;
      if (this.log) console.log(`[Ours] ready: persona_ms=${m.persona_ms} voice=${m.voice}`);
      if (this._readyResolve) this._readyResolve();
    } else if (m.type === 'error') {
      if (this.log) console.error(`[Ours] server error: ${m.msg}`);
      if (this._readyReject) this._readyReject(new Error(m.msg));
      this.onError(m.msg);
    } else {
      this.onText(m); // {"type":"text"|"event"|"user_text",...}
    }
  }

  /** Append PCM16LE mono @48k bytes (from orchestrator sink). */
  pushUplink48k(buf) {
    this.uplinkBuf48k = Buffer.concat([this.uplinkBuf48k, buf]);
  }

  /** Real-time frame clock: one 80 ms frame per tick, silence when starved. */
  startPacer() {
    if (this.pacer) return;
    this.pacer = setInterval(() => {
      if (!this.ready || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      let chunk;
      const src = this.uplinkBuf48k;
      if (src.length >= UPLINK_48K_BYTES) {
        chunk = src.slice(0, UPLINK_48K_BYTES);
        this.uplinkBuf48k = src.slice(UPLINK_48K_BYTES);
      } else if (src.length > 0) {
        chunk = Buffer.alloc(UPLINK_48K_BYTES);
        src.copy(chunk, 0, 0, src.length);
        this.uplinkBuf48k = Buffer.alloc(0);
      } else {
        chunk = Buffer.alloc(UPLINK_48K_BYTES); // silence — clock never stops
      }
      const f24 = ds48kTo24k(i16ToF32(chunk)); // 1920 f32 @24k
      this.ws.send(Buffer.from(f24.buffer, f24.byteOffset, f24.byteLength));
      this.framesSent += 1;
    }, FRAME_MS);
  }

  close() {
    if (this.pacer) { clearInterval(this.pacer); this.pacer = null; }
    try {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.close();
    } catch {}
  }
}

module.exports = { OursClient, i16ToF32, f32ToI16, ds48kTo24k, us24kTo48k,
                   ORCH_SR, OURS_SR, OURS_FRAME_SMP, UPLINK_48K_BYTES };

// ── Adapter CLI (orchestrator leg mirrors moshi_adapter.js) ─────────────────
if (require.main === module) {
  const wrtc = require('wrtc');
  const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;

  const argv = yargs(hideBin(process.argv))
    .scriptName('ours_adapter')
    .usage('$0 --role <A|B> --signalUrl <ws://host:port/signal> [--oursUrl <ws://host:8988/ws>] [opts]')
    .option('role', { type: 'string', choices: ['A', 'B'], demandOption: true })
    .option('signalUrl', { type: 'string', demandOption: true })
    .option('oursUrl', { type: 'string', default: DEFAULT_OURS_URL })
    .option('persona', { type: 'string', default: DEFAULT_PERSONA })
    .option('voicePreset', { type: 'string', default: process.env.OURS_VOICE_PRESET || '' })
    .option('log', { type: 'boolean', default: true })
    .help().argv;
  // NB: single_conversation.sh also passes --moshiUrl/--tokenServer to non-GPT
  // adapters; yargs is non-strict so they are accepted and ignored.

  const ROLE = argv.role;
  const LOG = argv.log;
  let orchPC, orchSource, orchSink, orchWS, client;
  let downlink48kBuf = Buffer.alloc(0); // i16 @48k awaiting 10 ms slicing

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  async function connectOrchestrator() {
    orchPC = new wrtc.RTCPeerConnection();
    orchSource = new RTCAudioSource({ sampleRate: ORCH_SR, channelCount: 1 });
    orchPC.addTrack(orchSource.createTrack());

    orchPC.ontrack = ({ track }) => {
      if (track.kind !== 'audio') return;
      if (orchSink) try { orchSink.stop(); } catch {}
      orchSink = new RTCAudioSink(track, { sampleRate: ORCH_SR, channelCount: 1, bitDepth: 16 });
      orchSink.ondata = ({ samples }) => {
        const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
        client.pushUplink48k(view);
      };
    };

    orchWS = new WebSocket(argv.signalUrl);
    orchWS.on('error', (err) => {
      console.error(`[Ours-${ROLE}] Orchestrator WS error:`, err && err.message || err);
      process.exit(2);
    });
    await new Promise((res) => orchWS.once('open', res));

    orchPC.onicecandidate = ({ candidate }) => {
      if (!candidate) return;
      orchWS.send(JSON.stringify({ role: ROLE, type: 'candidate', candidate: {
        candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex
      }}));
    };

    const offer = await orchPC.createOffer();
    await orchPC.setLocalDescription(offer);
    orchWS.send(JSON.stringify({ role: ROLE, type: 'offer', sdp: offer.sdp }));

    orchWS.on('message', async (msg) => {
      const data = JSON.parse(msg);
      if (data.type === 'answer' && data.role === ROLE) {
        await orchPC.setRemoteDescription({ type: 'answer', sdp: data.sdp });
        if (LOG) console.log(`[Ours-${ROLE}] Orchestrator answer applied`);
      } else if (data.type === 'candidate' && data.role === ROLE) {
        try { await orchPC.addIceCandidate(data.candidate); } catch {}
      }
    });
  }

  function onAgent24k(f24) {
    // 24k f32 -> 48k i16 -> exact 10 ms chunks into WebRTC source.
    const i16b = f32ToI16(us24kTo48k(f24));
    downlink48kBuf = Buffer.concat([downlink48kBuf, i16b]);
    while (downlink48kBuf.length >= ORCH_10MS_BYTES) {
      const ab = new ArrayBuffer(ORCH_10MS_BYTES);
      new Uint8Array(ab).set(downlink48kBuf.subarray(0, ORCH_10MS_BYTES));
      downlink48kBuf = downlink48kBuf.slice(ORCH_10MS_BYTES);
      orchSource.onData({ samples: new Int16Array(ab), sampleRate: ORCH_SR,
                          bitsPerSample: 16, channelCount: 1 });
    }
  }

  async function shutdown() {
    if (LOG) console.log(`[Ours-${ROLE}] Shutting down… sent=${client ? client.framesSent : 0} recv=${client ? client.framesRecv : 0}`);
    try { if (client) client.close(); } catch {}
    try { if (orchSink) orchSink.stop(); } catch {}
    try { if (orchPC) orchPC.close(); } catch {}
    try { if (orchWS && orchWS.readyState === WebSocket.OPEN) orchWS.close(); } catch {}
    process.exit(0);
  }

  (async function main() {
    if (LOG) console.log(`[Ours-${ROLE}] Start (signal=${argv.signalUrl}) ours=${argv.oursUrl} frame=${OURS_FRAME_SMP}smp/${FRAME_MS}ms`);
    client = new OursClient({
      url: argv.oursUrl,
      persona: argv.persona,
      voicePreset: argv.voicePreset || null,
      onAgent24k,
      onText: (m) => {
        if (!LOG) return;
        if (m.type === 'event') console.log(`[Ours-${ROLE}] [EVENT ${m.kind}] t=${m.t} ${JSON.stringify(m.ref || m.payload || '').slice(0, 160)}`);
        else if (m.type === 'text') console.log(`[Ours-${ROLE}] [TEXT] ${String(m.text).slice(-160)}`);
      },
      onError: (msg) => { console.error(`[Ours-${ROLE}] FATAL server error: ${msg}`); process.exit(3); },
    });
    await connectOrchestrator();
    await client.connect();
    client.startPacer(); // frame clock starts and never stops
  })().catch((e) => { console.error(`[Ours-${ROLE}] startup failed:`, e && e.message || e); process.exit(2); });
}
