"""The three checks. Pure functions of a sample array and a rate; nothing here touches files or ffmpeg."""

from dataclasses import dataclass

import numpy as np

from .loudness import HOP, momentary

SHORTEST_LAG = 0.5  # seconds; under this the 400 ms momentary window correlates neighbouring frames by itself


@dataclass(frozen=True)
class LoopResult:
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

    Finds the shortest lag (0.5 s or more) at which the loudness envelope's autocorrelation peaks at or
    above `threshold`: the fundamental repeat. A fundamental shorter than `min_period` is music-like
    (a bar) and becomes a note; at or above it, the file is a loop and fails. With no peak above the
    threshold, the score is the best value in the flaggable range and the file passes.
    """
    duration = len(samples) / sr
    if duration < 2 * min_period + 1:
        return LoopResult(
            0.0, None, threshold, min_period, True,
            (f"too short to test for a repeat longer than {min_period:g}s",),
        )
    ac = _autocorrelation(momentary(samples, sr))
    n = len(ac)
    first, lo, hi = int(round(SHORTEST_LAG / HOP)), int(round(min_period / HOP)), n // 2
    peaks = [
        k for k in range(first, hi)
        if ac[k] >= threshold and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]
    ]
    if peaks:
        k = peaks[0]
        period = round(k * HOP, 2)
        if k < lo:
            note = f"repeats every {period:.1f}s, under --min-period {min_period:g}s, not flagged"
            return LoopResult(float(ac[k]), period, threshold, min_period, True, (note,))
        return LoopResult(float(ac[k]), period, threshold, min_period, False)
    k = lo + int(np.argmax(ac[lo:hi]))
    return LoopResult(float(ac[k]), round(k * HOP, 2), threshold, min_period, True)
