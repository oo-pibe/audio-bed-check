"""The three checks. Pure functions of a sample array and a rate; nothing here touches files or ffmpeg."""

from dataclasses import dataclass

import numpy as np

from .loudness import HOP, block_loudness, k_weight, momentary, true_peak

# seconds; under this the 400 ms momentary window correlates neighbouring frames by itself
LOOP_SHORTEST_LAG = 0.5
# a loop correlates at every multiple of its period; the fundamental is the shortest peak within
# this of the strongest, so a half-period peak cannot name the repeat
LOOP_PEAK_TOLERANCE = 0.05


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
    autocorrelation peaks at or above `threshold`, the shortest one within LOOP_PEAK_TOLERANCE of the
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
    first, lo, hi = int(round(LOOP_SHORTEST_LAG / HOP)), int(round(min_period / HOP)), n // 2
    # a plain scan: a 10-minute bed is 3,000 lags, nothing next to the K-weighting
    peaks = [k for k in range(first, hi)
             if ac[k] >= threshold and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return LoopResult(min(float(ac[lo:hi].max()), 1.0), None, threshold, min_period, True)
    best = max(ac[p] for p in peaks)
    k = next(p for p in peaks if ac[p] >= best - LOOP_PEAK_TOLERANCE)
    period = round(k * HOP, 2)
    passed = k < lo
    notes = ()
    if passed:
        notes = (f"repeats every {period:.1f}s, under --min-period {min_period:g}s, not flagged",)
    return LoopResult(min(float(ac[k]), 1.0), period, threshold, min_period, passed, notes)


STEP_BLOCK = 0.5  # seconds; the scale a listener hears as "the sound changed" rather than as texture
STEP_SIDE = 4     # blocks (2 s) averaged on each side of a boundary for the sustained step


@dataclass(frozen=True)
class StepsResult:
    """step_db: largest change in mean level between the 2 s before and after a half-second boundary,
    at step_at_s (seconds into the file); the only field that decides `passed`. range_db (max minus
    min) and transient_db (largest change between adjacent half-second blocks) are reported, never
    failed. step_at_s is None when the file is too short to measure. Values are full precision; the
    CLI rounds."""

    step_db: float
    step_at_s: float | None
    range_db: float
    transient_db: float
    peak_dbtp: float
    max_step: float
    passed: bool
    notes: tuple[str, ...] = ()


def level_steps(samples: np.ndarray, sr: int, *, max_step: float = 6.0, edge: float = 0.75) -> StepsResult:
    """Does the level lurch?

    Only the sustained step fails: the mean of the 2 s after a boundary against the 2 s before. A hard
    join shifts the level and it stays shifted; a crowd surge spikes and comes back, and that is the
    recording's character, so range and transient are reported, never failed. A step within the
    dropped edge plus the run-up (about 2.75 s of either end at the defaults) is not seen.
    """
    if edge < 0:
        raise ValueError("edge must be >= 0")
    env = block_loudness(k_weight(samples, sr), sr, STEP_BLOCK, STEP_BLOCK)
    skip = int(np.ceil(edge / STEP_BLOCK))
    env = env[skip:len(env) - skip] if skip else env
    peak = true_peak(samples, sr)
    if len(env) < 2 * STEP_SIDE + 1:
        need = (2 * STEP_SIDE + 1 + 2 * skip) * STEP_BLOCK
        return StepsResult(0.0, None, 0.0, 0.0, peak, max_step, True,
                           (f"too short to measure level steps (needs at least {need:g}s)",))
    range_db = float(env.max() - env.min())
    transient = float(np.abs(np.diff(env)).max())
    step, at = 0.0, None
    # a plain scan: a 10-minute bed is 1,200 boundaries, nothing next to the K-weighting
    for i in range(STEP_SIDE, len(env) - STEP_SIDE + 1):
        d = abs(float(env[i:i + STEP_SIDE].mean() - env[i - STEP_SIDE:i].mean()))
        if d > step:
            step, at = d, (skip + i) * STEP_BLOCK
    return StepsResult(step, at, range_db, transient, peak, max_step, step <= max_step)
