// Agent playback on the audio thread. The main thread posts one 80 ms chunk per server frame; this
// processor keeps a queue and reads it directly in process(), so there is no scheduling margin and a
// busy tab cannot delay a chunk that is already here.
//
// Lead policy (all in samples; ms in comments at 24 kHz):
//   target   = worst late arrival in the last WINDOW s + MARGIN, floored at FLOOR, capped at CAP.
//              A chunk is expected one frame after the previous one; arriving later is lateness the
//              queue had to cover. The target rises the moment jitter is seen and falls when the
//              window slides past it. FLOOR 10 ms and the frame-plus-FLOOR start rule are what the
//              Moshi reference worklet uses at its initial setting (80 ms frame + 10 ms partial).
//   start    : play `target` after the first chunk arrives (Moshi's 'partial buffer' rule): on time,
//              the queue then holds exactly `target` when the next chunk lands. Fade the first quantum in.
//   underrun : queue empty mid-stream: fade out, report how long the silence lasted, re-buffer to
//              frame + target and fade back in.
//   grow     : if less than the target is queued (the target rose, or arrivals are running late) and
//              the head chunk is a pause, hold output (silence inside a pause) until the target is
//              queued again, instead of waiting for a hole.
//   drain    : excess E = queued - (frame + target) - SLACK. While E > 0 a pause chunk at the head
//              is dropped (E falls 80 ms per 80 ms of pause) and speech is read at
//              r = min(RATE_MAX, 1 + E / TAU) by linear interpolation, so dE/dt = -E / TAU: an
//              exponential return to the target, no overshoot, r -> 1 as E -> 0.
//   overflow : if the queue ever exceeds frame + target + HARD (pathological burst), the oldest audio
//              is cut to frame + target, as the Moshi worklet does; reported as `cut`.
class AgentPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.frame = o.frameSize;                              // 1920
    const ms = (m) => Math.round(sampleRate * m / 1000);
    this.FLOOR = ms(10); this.CAP = ms(350); this.MARGIN = ms(10); this.WINDOW = 30;
    this.SLACK = ms(80); this.HARD = ms(1000); this.TAU = 2.0; this.RATE_MAX = 1.06; this.QUIET = 0.005;
    this.reset();
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "reset") { this.reset(); return; }
      this.arrive(new Float32Array(e.data));
    };
  }
  reset() {
    this.q = []; this.pos = 0; this.started = false; this.fadeIn = false; this.growing = false;
    this.target = this.FLOOR; this.expectAt = 0; this.late = []; this.silenceFrom = -1;
    this.logAt = 0; this.rate = 1; this.startAt = -1;
  }
  queued() { let n = -Math.floor(this.pos); for (const c of this.q) n += c.pcm.length; return n; }
  arrive(pcm) {
    const now = currentTime;
    // measured jitter -> target
    const lateS = this.expectAt ? Math.max(0, now - this.expectAt) : 0;
    this.expectAt = Math.max(now, this.expectAt) + this.frame / sampleRate;
    this.late.push([now, lateS]);
    while (this.late.length && this.late[0][0] < now - this.WINDOW) this.late.shift();
    let worst = 0; for (const [, l] of this.late) if (l > worst) worst = l;
    this.target = Math.min(this.CAP, Math.max(this.FLOOR, Math.round(worst * sampleRate) + this.MARGIN));
    let e = 0; for (let i = 0; i < pcm.length; i++) e += pcm[i] * pcm[i];
    const quiet = Math.sqrt(e / pcm.length) < this.QUIET;
    const before = this.queued();
    if (!this.started && !this.q.length) this.startAt = now + this.target / sampleRate;
    this.q.push({ pcm, quiet });
    // overflow: cut the oldest audio back to frame + target
    let over = this.queued() - (this.frame + this.target + this.HARD);
    if (over > 0) {
      const cutFrom = this.queued();
      while (this.queued() > this.frame + this.target && this.q.length > 1) { this.q.shift(); this.pos = 0; }
      this.port.postMessage({ type: "cut", cut_ms: this.ms(cutFrom - this.queued()) });
    }
    if (now >= this.logAt) {
      this.logAt = now + 1;
      this.port.postMessage({ type: "lead", ms: this.ms(before), target_ms: this.ms(this.target), rate: +this.rate.toFixed(3) });
    }
  }
  ms(n) { return +(n * 1000 / sampleRate).toFixed(1); }
  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    const now = currentTime;
    const need = this.frame + this.target;
    if (!this.started) {
      if (this.q.length && this.startAt >= 0 && now >= this.startAt) {
        this.started = true; this.fadeIn = true;
        if (this.silenceFrom >= 0) {
          this.port.postMessage({ type: "underrun", late_ms: +((now - this.silenceFrom) * 1000).toFixed(0), target_ms: this.ms(this.target) });
          this.silenceFrom = -1;
        }
      } else { out.fill(0); return true; }
    }
    // grow inside a pause: the target rose above what is queued
    if (this.q.length && this.q[0].quiet && this.queued() < this.target) {
      if (!this.growing) { this.growing = true; this.port.postMessage({ type: "grow", from_ms: this.ms(this.queued()), target_ms: this.ms(this.target) }); }
      out.fill(0); return true;
    }
    this.growing = false;
    // drain
    let E = this.queued() - need - this.SLACK;
    while (E > 0 && this.q.length > 1 && this.q[0].quiet) {
      this.port.postMessage({ type: "drain", dropped_ms: this.ms(this.q[0].pcm.length - Math.floor(this.pos)), excess_ms: this.ms(E) });
      this.q.shift(); this.pos = 0; E = this.queued() - need - this.SLACK;
    }
    this.rate = E > 0 ? Math.min(this.RATE_MAX, 1 + (E / sampleRate) / this.TAU) : 1;
    // read with linear interpolation at this.rate; pos is the fractional read position in the head chunk
    let i = 0;
    while (i < out.length && this.q.length) {
      const c = this.q[0].pcm, k = Math.floor(this.pos), f = this.pos - k;
      if (k + 1 >= c.length) { this.pos = Math.max(0, this.pos - c.length); this.q.shift(); continue; }
      out[i++] = c[k] * (1 - f) + c[k + 1] * f;
      this.pos += this.rate;
    }
    if (this.fadeIn) { this.fadeIn = false; for (let j = 0; j < i; j++) out[j] *= j / i; }
    if (i < out.length) {                                    // ran dry: fade out, re-buffer
      for (let j = 0; j < i; j++) out[j] *= (i - j) / i;
      for (let j = i; j < out.length; j++) out[j] = 0;
      this.started = false; this.silenceFrom = now + i / sampleRate; this.pos = 0; this.startAt = -1;
    }
    return true;
  }
}
registerProcessor("agent-player", AgentPlayer);
