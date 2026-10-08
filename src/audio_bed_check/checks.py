"""The three checks. Pure functions of a sample array and a rate; nothing here touches files or ffmpeg."""

import math
from dataclasses import dataclass

import numpy as np

from .loudness import HOP, block_loudness, k_weight, momentary, true_peak

# seconds; under this the 400 ms momentary window correlates neighbouring frames by itself
LOOP_SHORTEST_LAG = 0.5
# a loop correlates at every multiple of its period; the fundamental is the shortest peak within
# this of the strongest, so a half-period peak cannot name the repeat
LOOP_PEAK_TOLERANCE = 0.05
# frames (5 s at the 100 ms hop); the least overlap the extended-lag test correlates over
LOOP_MIN_OVERLAP = 50
# a clip repeated once correlates at ~1.0 over its overlap; short overlaps of unlooped beds reach 0.74
# by chance (worst of 600 generated beds), so lags past half the file need this much
LOOP_EXTENDED_THRESHOLD = 0.95
LOOP_EXTENDED_NOTE = "found by the extended-lag test: the clip was repeated once to fill the file"


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


def _detrend(env: np.ndarray) -> np.ndarray:
    """The envelope with its linear trend removed, so a long fade does not correlate at every lag."""
    t = np.arange(len(env))
    return env - np.polyval(np.polyfit(t, env, 1), t)


def _autocorrelation(env: np.ndarray) -> np.ndarray:
    """Unbiased, normalised autocorrelation of an already detrended envelope, by FFT. A perfect repeat
    scores 1.0."""
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
    strongest, searched up to half the file. A fundamental shorter than `min_period` is music-like (a
    bar) and becomes a note; at or above it, the file is a loop and fails. With no peak at the
    threshold, the lags from half the file to the file length minus LOOP_MIN_OVERLAP (or minus
    `min_period`, if longer) are tested by the Pearson correlation of the frame-to-frame changes in
    the two overlapping stretches of envelope: a clip repeated once to fill the file scores about 1.0
    there, and at or above LOOP_EXTENDED_THRESHOLD the file fails. Otherwise the score is the best
    autocorrelation in the flaggable range and the file passes.
    """
    if min_period <= 0:
        raise ValueError("min_period must be positive")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be between 0 (exclusive) and 1")
    duration = len(samples) / sr
    if duration < 2 * min_period + 1:
        return LoopResult(0.0, None, threshold, min_period, True,
                          (f"too short to test for a repeat longer than {min_period:g}s",))
    env = momentary(samples, sr)
    if np.ptp(env) < 0.01:   # silence or a constant tone; a steady bed still varies by tenths of a dB
        return LoopResult(0.0, None, threshold, min_period, True,
                          ("level is constant; nothing to correlate",))
    env = _detrend(env)
    ac = _autocorrelation(env)
    n = len(ac)
    first, lo, hi = int(round(LOOP_SHORTEST_LAG / HOP)), int(round(min_period / HOP)), n // 2
    # a plain scan: a 10-minute bed is 3,000 lags, nothing next to the K-weighting
    peaks = [k for k in range(first, hi)
             if ac[k] >= threshold and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        extended = range(hi, n - max(lo, LOOP_MIN_OVERLAP) + 1)
        if extended:
            # frame-to-frame changes, not levels: a step halfway would otherwise leave two matching
            # ramps after detrending and read as a repeat. A copied clip copies its fine texture.
            # nan_to_num: a stretch with no variation at all has no correlation, so it cannot win
            d = np.diff(env)
            r, k = max((float(np.nan_to_num(np.corrcoef(d[:n - 1 - k], d[k:])[0, 1], nan=-1.0)), k)
                       for k in extended)
            if r >= LOOP_EXTENDED_THRESHOLD and k >= lo:
                return LoopResult(min(r, 1.0), round(k * HOP, 2), threshold, min_period, False,
                                  (LOOP_EXTENDED_NOTE,))
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
    failed. step_at_s is None when the file is too short to measure or the level never changes.
    Values are full precision; the CLI rounds."""

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
    dropped edge plus the run-up is not seen as a full step. A join under 3 s from either end at the
    defaults reads smaller than it is, and inside the dropped edge it is not seen at all.
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
    boundaries = range(STEP_SIDE, len(env) - STEP_SIDE + 1)
    # a plain scan: a 10-minute bed is 1,200 boundaries, nothing next to the K-weighting
    steps = [abs(float(env[i:i + STEP_SIDE].mean() - env[i - STEP_SIDE:i].mean())) for i in boundaries]
    if max(steps) == 0.0:   # digital silence: every boundary ties, so no place is the worst
        return StepsResult(0.0, None, range_db, transient, peak, max_step, True,
                           ("no level change anywhere",))
    worst = int(np.argmax(steps))
    step, at = steps[worst], (skip + boundaries[worst]) * STEP_BLOCK
    return StepsResult(step, at, range_db, transient, peak, max_step, step <= max_step)


SEP_GATE_HOP = 0.02   # seconds; RMS resolution for finding speech in the voiceover
SEP_MIN_RUN = 0.25    # seconds; shortest stretch above the gate that counts as speech
SEP_GUARD = 0.25      # seconds trimmed from each end of a gap, so tails of speech do not leak in
SEP_MIN_GAP = 0.4     # seconds; shortest usable bed-only window
SEP_BLOCK = 0.05      # seconds; fine enough that a block never straddles the edge of a 250 ms run


class NoSpeechError(ValueError):
    """The separation check could not run: no speech windows, no gaps between them, or (with an
    offset) no window inside the mix. The CLI reports it as exit 2, not as a failing mix."""


@dataclass(frozen=True)
class SeparationResult:
    """separation_lu: speech_lkfs minus bed_lkfs, where speech_lkfs is the mean over speech runs of
    each run's 90th-percentile 50 ms K-weighted level in the mix and bed_lkfs the same over the
    bed-only gaps; the only field that decides `passed`. runs/gaps count the windows found in the
    voiceover, before any fall outside the mix. warnings (named so because they are actionable, unlike
    the other checks' notes: a hot peak, windows dropped by the offset) never fail. Values are full
    precision; the CLI rounds."""

    separation_lu: float
    speech_lkfs: float
    bed_lkfs: float
    runs: int
    gaps: int
    peak_dbtp: float
    min_lu: float
    passed: bool
    warnings: tuple[str, ...] = ()


# speech_runs and gaps_between are module-public for the tests; they are not re-exported by the package.
def speech_runs(vo: np.ndarray, sr: int, gate_dbfs: float = -44.0) -> list[tuple[float, float]]:
    """(start, end) seconds of every stretch of the voiceover above the gate for at least SEP_MIN_RUN.

    Mono or stereo: the gate reads the RMS over every sample of every channel in a hop, a power sum, so
    a voice on one channel or in anti-phase still counts and identical channels read as mono.
    """
    x = np.asarray(vo, dtype=np.float64)
    h = int(round(SEP_GATE_HOP * sr))
    blocks = len(x) // h
    if blocks == 0:
        return []
    rms = np.sqrt((x[:blocks * h] ** 2).reshape(blocks, -1).mean(axis=1))
    with np.errstate(divide="ignore"):
        loud = 20 * np.log10(rms) > gate_dbfs
    runs, start = [], None
    # a plain scan: a 10-minute voiceover is 30,000 hops, nothing next to the K-weighting
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if (i - start) * SEP_GATE_HOP >= SEP_MIN_RUN:
                runs.append((round(start * SEP_GATE_HOP, 2), round(i * SEP_GATE_HOP, 2)))
            start = None
    if start is not None and (blocks - start) * SEP_GATE_HOP >= SEP_MIN_RUN:
        runs.append((round(start * SEP_GATE_HOP, 2), round(blocks * SEP_GATE_HOP, 2)))
    return runs


def gaps_between(runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Bed-only windows: the space between consecutive runs, trimmed by SEP_GUARD,
    kept if SEP_MIN_GAP or longer."""
    gaps = []
    for (_, end), (start, _) in zip(runs, runs[1:], strict=False):
        a, b = end + SEP_GUARD, start - SEP_GUARD
        if b - a >= SEP_MIN_GAP:
            gaps.append((round(a, 2), round(b, 2)))
    return gaps


def _window_level(blocks: np.ndarray, t0: float, t1: float) -> float | None:
    """90th percentile of the blocks fully inside [t0, t1),
    or None when the window is not wholly in the mix."""
    i0 = math.ceil(t0 / SEP_BLOCK - 1e-9)
    i1 = math.floor(t1 / SEP_BLOCK + 1e-9)
    if i0 < 0 or i1 > len(blocks) or i1 <= i0:
        return None
    return float(np.percentile(blocks[i0:i1], 90))


def separation(vo: np.ndarray, mix: np.ndarray, sr: int, *, min_lu: float = 10.0, gate_dbfs: float = -44.0,
               vo_offset: float = 0.0) -> SeparationResult:
    """Is the voice on top of the bed?

    Speech windows come from the voiceover on its own; each is measured in the MIX (voice plus bed)
    and compared with the bed-only gaps between them. The 90th percentile per window, so a breath
    inside a run does not drag the speech figure down. The result is what a listener hears, which is
    also what WCAG G56 describes, not the ratio of the two stems.
    """
    runs = speech_runs(vo, sr, gate_dbfs)
    if not runs:
        raise NoSpeechError(f"no speech found in the voiceover above {gate_dbfs:g} dBFS")
    gaps = gaps_between(runs)
    if not gaps:
        raise NoSpeechError(f"the voiceover has no gap of {SEP_MIN_GAP:g}s or more between speech runs, "
                            "so there is no bed-only window to compare against")
    blocks = block_loudness(k_weight(mix, sr), sr, SEP_BLOCK, SEP_BLOCK)
    speech = [v for v in (_window_level(blocks, a + vo_offset, b + vo_offset) for a, b in runs)
              if v is not None]
    bed = [v for v in (_window_level(blocks, a + vo_offset, b + vo_offset) for a, b in gaps) if v is not None]
    if not speech or not bed:
        raise NoSpeechError("the voiceover's speech windows fall outside the mix; check --vo-offset")
    speech_lkfs, bed_lkfs = float(np.mean(speech)), float(np.mean(bed))
    sep = speech_lkfs - bed_lkfs
    peak = true_peak(mix, sr)
    warnings: tuple[str, ...] = ()
    dropped = len(runs) + len(gaps) - len(speech) - len(bed)
    if dropped:
        warnings += (f"{dropped} window(s) fall outside the mix; check --vo-offset",)
    if peak > -1.0:
        warnings += ("mix true peak above -1 dBTP",)
    return SeparationResult(sep, speech_lkfs, bed_lkfs, len(runs), len(gaps), peak, min_lu, sep >= min_lu,
                            warnings)
