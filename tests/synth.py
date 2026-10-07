"""Signal generators for the test suite. Every fixture is built here, seeded; no audio file is committed."""

import wave

import numpy as np

SR = 48000


def noise(seconds: float, seed: int = 0, rms_dbfs: float = -20.0, sr: int = SR) -> np.ndarray:
    """White noise at the given RMS level."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * sr))
    return (x / np.sqrt((x * x).mean()) * 10 ** (rms_dbfs / 20)).astype(np.float32)


def sine(seconds: float, freq: float, amp: float, phase: float = 0.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def ar1_envelope(
    n: int, seed: int = 0, rho: float = 0.7, depth_db: float = 6.0, knot_s: float = 0.1, sr: int = SR
) -> np.ndarray:
    """A stationary, slowly wandering gain (linear), about depth_db peak to peak, with short memory.

    AR(1) with rho=0.7 at 0.1 s knots: the correlation at a 0.5 s lag is 0.7**5 = 0.17, so the
    envelope has texture but no repeat for the loop check to find.
    """
    rng = np.random.default_rng(seed)
    knots = int(n / (knot_s * sr)) + 2
    walk = np.zeros(knots)
    for i in range(1, knots):
        walk[i] = rho * walk[i - 1] + rng.standard_normal()
    walk = (walk - walk.mean()) / (np.abs(walk).max() or 1.0) * (depth_db / 2)
    t = np.linspace(0, knots - 1, n)
    return (10 ** (np.interp(t, np.arange(knots), walk) / 20)).astype(np.float32)


def tile(segment: np.ndarray, times: int, crossfade: float = 0.0, sr: int = SR) -> np.ndarray:
    """Repeat a segment, optionally with an equal-power crossfade at each join.

    With a crossfade of k samples each repeat adds L - k, so the period is L - k.
    """
    if crossfade <= 0:
        return np.tile(segment, times).astype(np.float32)
    k = int(crossfade * sr)
    ramp = np.linspace(0, np.pi / 2, k)
    fade_in, fade_out = np.sin(ramp), np.cos(ramp)
    out = segment.astype(np.float64)
    for _ in range(times - 1):
        joined = out[-k:] * fade_out + segment[:k] * fade_in
        out = np.concatenate([out[:-k], joined, segment[k:]])
    return out.astype(np.float32)


def step_gain(n: int, at_s: float, db: float, sr: int = SR) -> np.ndarray:
    """A hard level change at at_s: gain 1 before, 10**(db/20) after."""
    g = np.ones(n, dtype=np.float32)
    g[int(at_s * sr):] = 10 ** (db / 20)
    return g


def swell_gain(n: int, centre_s: float, width_s: float, db: float, sr: int = SR) -> np.ndarray:
    """A raised-cosine swell of +db peaking at centre_s, width_s wide, returning to 1."""
    t = np.arange(n) / sr
    u = (t - (centre_s - width_s / 2)) / width_s
    bump = np.where((u >= 0) & (u <= 1), (1 - np.cos(2 * np.pi * u)) / 2, 0.0)
    return (10 ** (bump * db / 20)).astype(np.float32)


def fade_edges(x: np.ndarray, seconds: float, sr: int = SR) -> np.ndarray:
    k = int(seconds * sr)
    y = x.astype(np.float32).copy()
    y[:k] *= np.linspace(0, 1, k, dtype=np.float32)
    y[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return y


def gated(x: np.ndarray, runs: list[tuple[float, float]], sr: int = SR) -> np.ndarray:
    """Keep x only inside the (start, end) runs, silence elsewhere. Speech-shaped timing, noise content."""
    y = np.zeros_like(x, dtype=np.float32)
    for a, b in runs:
        y[int(a * sr):int(b * sr)] = x[int(a * sr):int(b * sr)]
    return y


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** (db / 20)).astype(np.float32)


def write_wav(path, x: np.ndarray, sr: int = SR) -> None:
    """16-bit mono WAV, the one format every ffmpeg build and the stdlib both read."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path) -> tuple[np.ndarray, int]:
    """Read back a 16-bit mono WAV as float32 in [-1, 1]. Used as a stand-in for decode() in CLI tests."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767), sr
