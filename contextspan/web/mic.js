// Capture on the audio thread and hand up exactly one model frame at a time. A ScriptProcessor
// would do this on the main thread, where a busy tab drops frames and the server hears gaps.
class MicFrames extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.frame = options.processorOptions.frameSize;
    this.buf = new Float32Array(0);
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    const merged = new Float32Array(this.buf.length + ch.length);
    merged.set(this.buf); merged.set(ch, this.buf.length);
    let off = 0;
    while (merged.length - off >= this.frame) {
      const f = merged.slice(off, off + this.frame);
      this.port.postMessage(f, [f.buffer]);
      off += this.frame;
    }
    this.buf = merged.slice(off);
    return true;
  }
}
registerProcessor("mic-frames", MicFrames);
