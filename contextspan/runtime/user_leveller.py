"""Bring the user channel to the loudness the fine-tune was trained on, using the same meter that
already levels the agent's voice.

Two libraries do the work; this module only feeds them one frame at a time.

  pyloudnorm   ITU-R BS.1770-4 loudness. `normalize_audio` in the vendored moshi already uses it
               to put the agent's voice prompt at -24 LUFS, so measuring the user channel with the
               same meter and the same target means both channels are on one scale — the scale
               the training data was on (user channel median -24.0 LUFS).
  noisereduce  spectral gating (Sainburg). Runs before the meter, so a quiet microphone's room is
               taken out before anything is lifted rather than being lifted with the voice.

The loudness is the standard short-term measurement — the meter over the last 3 s of audio —
turned into a gain toward the target, smoothed so it cannot pump. Nothing here looks ahead: both
libraries are handed only audio that has already arrived, and the frame that goes in is the frame
that comes out on the same clock.

`--raw-user-audio` bypasses all of it. `--enhance module:callable` replaces the denoiser with
another one (RNNoise, DeepFilterNet, an in-house enhancer): anything with `process(frame) -> frame`
at the engine's sample rate, stateful, no lookahead.
"""
from __future__ import annotations

import importlib
import os
import warnings

import numpy as np
import noisereduce as nr
import pyloudnorm as pyln

TARGET_LUFS = float(os.environ.get("CS_USER_TARGET_LUFS", "-24.0"))
WINDOW_S = 3.0          # EBU R128 short-term loudness window
MAX_GAIN_DB = 30.0
MIN_GAIN_DB = -24.0


class NoiseGate:
    """noisereduce over the last second, returning only the newest frame of it."""

    def __init__(self, sample_rate: int, seconds: float = 1.0, prop_decrease: float = 0.8):
        self.sr, self.prop = sample_rate, prop_decrease
        self.keep = int(sample_rate * seconds)
        self.buf = np.zeros(0, np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, frame])[-self.keep:]
        if self.buf.size < self.sr // 2:            # not enough context for a noise estimate yet
            return frame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nr.reduce_noise(y=self.buf, sr=self.sr, stationary=True,
                                    prop_decrease=self.prop, n_fft=1024)
        return clean[-frame.size:].astype(np.float32)


class UserLeveller:
    """`process(frame) -> frame`: denoise, meter, gain toward the target."""

    def __init__(self, sample_rate: int = 24000, target_lufs: float = TARGET_LUFS,
                 enhancer=None, up_s: float = 0.1, down_s: float = 0.1, frame_s: float = 0.08):
        self.sr = sample_rate
        self.target = target_lufs
        self.meter = pyln.Meter(sample_rate)                       # BS.1770-4, K-weighted
        self.enh = enhancer if enhancer is not None else NoiseGate(sample_rate)
        self.window = np.zeros(0, np.float32)
        self.keep = int(sample_rate * WINDOW_S)
        self.gain = 1.0
        self.a_up = float(np.exp(-frame_s / up_s))      # slow: a hot start must not jump
        self.a_down = float(np.exp(-frame_s / down_s))  # fast: coming down protects the peak
        self.lufs = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        x = self.enh.process(x)
        self.window = np.concatenate([self.window, x])[-self.keep:]
        if self.window.size >= int(self.sr * 0.4):                # the meter's minimum block
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lufs = self.meter.integrated_loudness(self.window)
            if np.isfinite(lufs) and lufs > -70.0:                # -inf / very low = silence: hold
                self.lufs = float(lufs)
                want = float(np.clip(10.0 ** ((self.target - lufs) / 20.0),
                                     10.0 ** (MIN_GAIN_DB / 20.0), 10.0 ** (MAX_GAIN_DB / 20.0)))
                a = self.a_up if want > self.gain else self.a_down
                self.gain = a * self.gain + (1.0 - a) * want
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak * self.gain > 0.95:                               # never clip: pull down now
            self.gain = 0.95 / peak
        return np.clip(x * self.gain, -0.99, 0.99).astype(np.float32)

    def report(self) -> dict:
        return {"lufs": None if self.lufs is None else round(self.lufs, 1),
                "gain_db": round(20.0 * np.log10(self.gain + 1e-12), 1), "target_lufs": self.target}


def measure_lufs(pcm: np.ndarray, sample_rate: int) -> float:
    """Whole-clip loudness with the same meter, for the page's warning and for tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pyln.Meter(sample_rate).integrated_loudness(np.asarray(pcm, dtype=np.float32)))


def load_enhancer(spec: str):
    """`module:callable` -> factory for an object with `process(frame) -> frame`."""
    mod, _, name = spec.partition(":")
    if not mod or not name:
        raise ValueError(f"--enhance wants module:callable, got {spec!r}")
    return getattr(importlib.import_module(mod), name)
