"""Signal generators for the test suite. Every fixture is built here, seeded; no audio file is committed."""

import wave

import numpy as np

SR = 48000


def _n(seconds: float, sr: int) -> int:
    """Sample count for a duration, rounded, so 0.29 s is 13920 samples and not 13919."""
    return int(round(seconds * sr))


def noise(seconds: float, seed: int = 0, rms_dbfs: float = -20.0, sr: int = SR) -> np.ndarray:
    """White noise at the given RMS level.

    Pass distinct seeds for independent signals; two calls with the same seed are identical.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(_n(seconds, sr))
    return (x / np.sqrt((x * x).mean()) * 10 ** (rms_dbfs / 20)).astype(np.float32)


def sine(seconds: float, freq: float, amp: float, phase: float = 0.0, sr: int = SR) -> np.ndarray:
    t = np.arange(_n(seconds, sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def ar1_envelope(
    n: int, seed: int = 0, rho: float = 0.7, depth_db: float = 6.0, knot_s: float = 0.1, sr: int = SR
) -> np.ndarray:
    """A stationary, slowly wandering gain (linear), about depth_db peak to peak, with short memory.

    n is a sample count: ar1_envelope(len(x)).

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
    k = _n(crossfade, sr)
    if k == 0:
        return np.tile(segment, times).astype(np.float32)
    if k >= len(segment):
        raise ValueError("crossfade must be shorter than the segment")
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
    g[_n(at_s, sr):] = 10 ** (db / 20)
    return g


def swell_gain(n: int, centre_s: float, width_s: float, db: float, sr: int = SR) -> np.ndarray:
    """A raised-cosine swell of +db peaking at centre_s, width_s wide, returning to 1."""
    t = np.arange(n) / sr
    u = (t - (centre_s - width_s / 2)) / width_s
    bump = np.where((u >= 0) & (u <= 1), (1 - np.cos(2 * np.pi * u)) / 2, 0.0)
    return (10 ** (bump * db / 20)).astype(np.float32)


def fade_edges(x: np.ndarray, seconds: float, sr: int = SR) -> np.ndarray:
    k = _n(seconds, sr)
    if k == 0:
        return x.astype(np.float32)
    if 2 * k > len(x):
        raise ValueError("fade longer than the signal")
    y = x.astype(np.float32).copy()
    y[:k] *= np.linspace(0, 1, k, dtype=np.float32)
    y[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return y


def gated(x: np.ndarray, runs: list[tuple[float, float]], sr: int = SR) -> np.ndarray:
    """Keep x only inside the (start, end) runs, silence elsewhere. Speech-shaped timing, noise content."""
    y = np.zeros_like(x, dtype=np.float32)
    for a, b in runs:
        y[_n(a, sr):_n(b, sr)] = x[_n(a, sr):_n(b, sr)]
    return y


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return (x * 10 ** (db / 20)).astype(np.float32)


def textured(seconds: float, seed: int = 0, sr: int = SR) -> np.ndarray:
    """White noise under an AR(1) gain: bed-like texture with no repeat."""
    x = noise(seconds, seed, sr=sr)
    return (x * ar1_envelope(len(x), seed, sr=sr)).astype(np.float32)


def write_wav(path, x: np.ndarray, sr: int = SR, clip: bool = False) -> None:
    """16-bit WAV, the one format every ffmpeg build and the stdlib both read: mono for `(n,)`,
    interleaved channels for `(n, ch)`.

    Raises if |x| exceeds 1.0 unless clip=True: a clipped fixture reads back flatter and quieter,
    which shows up as a wrong loudness number, not as a fixture error.
    """
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 1.0 and not clip:
        raise ValueError(f"signal peaks at {peak:.2f}; lower it or pass clip=True")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if np.ndim(x) == 1 else x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.rint(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path) -> tuple[np.ndarray, int]:
    """Read back a 16-bit WAV as float32 in [-1, 1], `(n,)` mono or `(n, ch)`. Used as a stand-in for
    decode() in CLI tests."""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, "read_wav expects 16-bit samples"
        sr, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767
    return (x if channels == 1 else x.reshape(-1, channels)), sr


def fade(x: np.ndarray, seconds: float = 0.01, sr: int = SR) -> np.ndarray:
    """Raised-cosine fade at both ends.

    A hard cut at the file edge cannot then add its own inter-sample overshoot.
    """
    n = _n(seconds, sr)
    ramp = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    y = x.astype(np.float32).copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def irregular_read(seconds: float, seed: int = 0) -> list[tuple[float, float]]:
    """Speech-like timing with no regular cadence: runs of 0.6-3 s, pauses of 0.95-1.5 s, seeded."""
    rng = np.random.default_rng(seed)
    runs, t = [], 0.5 + rng.uniform(0.0, 1.0)
    while True:
        end = t + rng.uniform(0.6, 3.0)
        if end > seconds - 0.5:
            return runs
        runs.append((round(t, 3), round(end, 3)))
        t = end + rng.uniform(0.95, 1.5)
