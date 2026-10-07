"""The three checks. Pure functions of a sample array and a rate; nothing here touches files or ffmpeg."""

from dataclasses import dataclass

import numpy as np

from .loudness import HOP, momentary

SHORTEST_LAG = 0.5  # seconds; under this the 400 ms momentary window correlates neighbouring frames by itself
PEAK_TOLERANCE = 0.05  # a loop correlates at every multiple of its period; the fundamental is the shortest
                       # peak within this of the strongest, so a half-period shoulder cannot name the repeat


@dataclass(frozen=True)
class LoopResult:
    """score: autocorrelation at period_s, 1.0 being an exact repeat. When nothing reached the
    threshold, score is the best value in the flaggable range and period_s is None."""

    score: float
    period_s: float | None
    threshold: float
    min_period_s: float
    passed: bool
    notes: tuple[str, ...] = ()


def _autocorrelation(env: np.ndarray) -> np.ndarray:
    """Unbiased, normalised autocorrelation of a detrended envelope, by FFT. A perfect repeat scores 1.0."""
    t = np.arange(len(env))
    env = env - np.polyval(np.polyfit(t, env, 1), t)
    n = len(env)
    size = 1
    while size < 2 * n:
        size <<= 1
    spectrum = np.fft.rfft(env, size)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    ac = ac / (n - np.arange(n))
    return ac / ac[0] if ac[0] > 0 else np.zeros(n)


def loop_score(
    samples: np.ndarray, sr: int, *, min_period: float = 6.0, threshold: float = 0.75
) -> LoopResult:
    """Does the bed repeat itself?

    Finds the fundamental repeat: among the lags (0.5 s or more) where the loudness envelope's
    autocorrelation peaks at or above `threshold`, the shortest one within PEAK_TOLERANCE of the
    strongest. A fundamental shorter than `min_period` is music-like (a bar) and becomes a note; at
    or above it, the file is a loop and fails. With no peak at the threshold, the score is the best
    value in the flaggable range and the file passes.
    """
    duration = len(samples) / sr
    if duration < 2 * min_period + 1:
        return LoopResult(0.0, None, threshold, min_period, True,
                          (f"too short to test for a repeat longer than {min_period:g}s",))
    ac = _autocorrelation(momentary(samples, sr))
    n = len(ac)
    first, lo, hi = int(round(SHORTEST_LAG / HOP)), int(round(min_period / HOP)), n // 2
    # a plain scan: a 10-minute bed is 3,000 lags, nothing next to the K-weighting
    peaks = [k for k in range(first, hi)
             if ac[k] >= threshold and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return LoopResult(min(float(ac[lo:hi].max()), 1.0), None, threshold, min_period, True)
    best = max(ac[p] for p in peaks)
    k = next(p for p in peaks if ac[p] >= best - PEAK_TOLERANCE)
    period = round(k * HOP, 2)
    passed = k < lo
    notes = ()
    if passed:
        notes = (f"repeats every {period:.1f}s, under --min-period {min_period:g}s, not flagged",)
    return LoopResult(min(float(ac[k]), 1.0), period, threshold, min_period, passed, notes)
