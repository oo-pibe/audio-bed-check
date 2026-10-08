"""The three checks. Pure functions of a sample array and a rate; nothing here touches files or ffmpeg."""

import math
from dataclasses import dataclass

import numpy as np

from .loudness import HOP, WINDOW, block_loudness, k_weight, true_peak

# seconds; under this the 400 ms momentary window correlates neighbouring frames by itself
LOOP_SHORTEST_LAG = 0.5
# a loop correlates at every multiple of its period; the fundamental is the shortest peak within
# this of the strongest, so a half-period peak cannot name the repeat
LOOP_PEAK_TOLERANCE = 0.05
LOOP_WINDOW = 100      # frames (10 s) of envelope changes correlated at a time
LOOP_MIN_WINDOW = 30   # frames; files under 30 s use a third of their length, never less than this
LOOP_STRIDE = 5        # frames between window starts


@dataclass(frozen=True)
class LoopResult:
    """score: Pearson correlation of the envelope's frame-to-frame changes at period_s, over the
    best-matching 10 s window; 1.0 is an exact repeat. When nothing reached the threshold, score is
    the strongest local maximum at lags of min_period_s or more and period_s is None."""

    score: float
    period_s: float | None
    threshold: float
    min_period_s: float
    passed: bool
    notes: tuple[str, ...] = ()


def _weighted(samples: np.ndarray, sr: int, weighted: np.ndarray | None) -> np.ndarray:
    """The caller's K-weighted signal, or k_weight(samples, sr) when there is none."""
    if weighted is None:
        return k_weight(samples, sr)
    if np.shape(weighted) != np.shape(samples):
        raise ValueError("weighted must be k_weight(samples, sr): same shape as samples")
    return weighted


def _lag_scores(d: np.ndarray, window: int, first: int, last: int) -> np.ndarray:
    """For each lag k in first..last, the best Pearson correlation of d[s:s+window] with
    d[s+k:s+k+window] over window starts s every LOOP_STRIDE frames. Window sums come from cumulative
    sums, so one lag is O(len(d)) in numpy. A window with no variation cannot win (scores -1)."""
    m = len(d)
    c1 = np.concatenate([[0.0], np.cumsum(d)])
    c2 = np.concatenate([[0.0], np.cumsum(d * d)])
    scores = np.full(last - first + 1, -1.0)
    # a loop over lags, all numpy inside: measured 0.3 s for a 10-minute bed, 4 s for an hour
    for i, k in enumerate(range(first, last + 1)):
        s = np.arange(0, m - k - window + 1, LOOP_STRIDE)
        cp = np.concatenate([[0.0], np.cumsum(d[:m - k] * d[k:])])
        sa, sb = c1[s + window] - c1[s], c1[s + k + window] - c1[s + k]
        va = c2[s + window] - c2[s] - sa * sa / window
        vb = c2[s + k + window] - c2[s + k] - sb * sb / window
        cov = cp[s + window] - cp[s] - sa * sb / window
        ok = (va > 1e-12) & (vb > 1e-12)
        if ok.any():
            scores[i] = float((cov[ok] / np.sqrt(va[ok] * vb[ok])).max())
    return scores


def loop_score(
    samples: np.ndarray, sr: int, *, min_period: float = 6.0, threshold: float = 0.9,
    weighted: np.ndarray | None = None,
) -> LoopResult:
    """Does the bed repeat itself?

    Works on d, the frame-to-frame changes of the momentary loudness (dB, 100 ms hop). A copied clip
    copies its fine texture, so d repeats exactly where the audio repeats, while a fade or a gain
    change moves the level and barely touches d. For every lag from 0.5 s to the file length minus
    one window, the score is the best Pearson correlation of a 10 s window of d with the window that
    many frames later (window starts every 0.5 s; a third of the file, at least 3 s, under 30 s).
    So a repeat is found even when only part of the file loops, and the longest repeat that can be
    found is the file length minus the window (about 10.5 s less than the file). Lags resolve in
    0.1 s steps.

    The fundamental is the shortest lag whose score is a local maximum at or above `threshold` and
    within LOOP_PEAK_TOLERANCE of the strongest such peak. Under `min_period` it is music-like (a
    bar) and becomes a note; at or above it, the file is a loop and fails. With no peak at the
    threshold, the score is the strongest local maximum at lags of `min_period` or more and the file
    passes.

    `weighted` is the K-weighted signal (`k_weight(samples, sr)`) if the caller has it already, so a
    file is filtered once for several checks.
    """
    if min_period <= 0:
        raise ValueError("min_period must be positive")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be between 0 (exclusive) and 1")
    too_short = LoopResult(0.0, None, threshold, min_period, True,
                           (f"too short to test for a repeat longer than {min_period:g}s",))
    if len(samples) / sr < 2 * min_period + 1:
        return too_short
    env = block_loudness(_weighted(samples, sr, weighted), sr, WINDOW, HOP)
    if np.ptp(env) < 0.01:   # silence or a constant tone; a steady bed still varies by tenths of a dB
        return LoopResult(0.0, None, threshold, min_period, True,
                          ("level is constant; nothing to correlate",))
    d = np.diff(env)
    window = min(LOOP_WINDOW, max(LOOP_MIN_WINDOW, len(env) // 3))
    first, lo = int(round(LOOP_SHORTEST_LAG / HOP)), int(round(min_period / HOP))
    last = len(d) - window
    if last < max(lo, first + 2):
        return too_short
    scores = _lag_scores(d, window, first, last)
    lags = np.arange(first, last + 1)
    # local maxima; never the first lag, whose only neighbour is on one side, and the last lag only
    # if it is at least its neighbour, since a repeat can sit right at the end of the range
    is_peak = np.zeros(len(scores), dtype=bool)
    is_peak[1:-1] = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    is_peak[-1] = scores[-1] >= scores[-2]
    peaks = is_peak & (scores >= threshold)
    if not peaks.any():
        flaggable = is_peak & (lags >= lo)
        score = float(scores[flaggable].max()) if flaggable.any() else 0.0
        return LoopResult(min(score, 1.0), None, threshold, min_period, True)
    best = scores[peaks].max()
    i = int(np.flatnonzero(peaks & (scores >= best - LOOP_PEAK_TOLERANCE))[0])
    k = int(lags[i])
    period = round(k * HOP, 2)
    passed = k < lo
    notes = ()
    if passed:
        notes = (f"repeats every {period:.1f}s, under --min-period {min_period:g}s, not flagged",)
    return LoopResult(min(float(scores[i]), 1.0), period, threshold, min_period, passed, notes)


STEP_BLOCK = 0.5  # seconds; the scale a listener hears as "the sound changed" rather than as texture
STEP_HOP = 0.1    # seconds between blocks, so a join is never more than 0.05 s from a boundary
STEP_SIDE = 20    # frames (2 s) averaged on each side of a boundary for the sustained step
STEP_GAP = 5      # frames (0.5 s) skipped at the boundary, so no block on either side straddles the join
STEP_TRANSIENT = int(round(STEP_BLOCK / STEP_HOP))   # frames between the two blocks a transient compares
STEP_FLOOR_LKFS = -70.0   # blocks under this count as silence for the step: noise this low cannot lurch


@dataclass(frozen=True)
class StepsResult:
    """step_db: largest change in mean level between the 2 s before a join and the 2 s after it, at
    step_at_s (seconds into the file); the only field that decides `passed`. range_db (max minus
    min) and transient_db (largest change between half-second blocks 0.5 s apart) are reported, never
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


def level_steps(samples: np.ndarray, sr: int, *, max_step: float = 6.0, edge: float = 0.75,
                weighted: np.ndarray | None = None) -> StepsResult:
    """Does the level lurch?

    Half-second K-weighted blocks every 0.1 s. At every boundary the sustained step is the mean of
    the 2 s of blocks after a 0.5 s transition gap against the 2 s of blocks before it; the gap means
    no block on either side straddles a join, so the reading does not depend on where the join falls
    between blocks. A change held about 2 s or longer therefore fails even if it comes back.

    Only the sustained step fails. A hard join shifts the level and it stays shifted; a crowd surge
    spikes and comes back, and that is the recording's character, so range and transient are
    reported, never failed. The first and last `edge` seconds are dropped (rounded up to whole 0.1 s
    frames); a join less than about 2.5 s inside them reads smaller than it is, and inside them it is
    not seen at all. Blocks under -70 LKFS count as -70 for the step, so a file of near silence
    cannot fail on its own noise.

    `weighted` is the K-weighted signal (`k_weight(samples, sr)`) if the caller has it already.
    """
    if edge < 0:
        raise ValueError("edge must be >= 0")
    env = block_loudness(_weighted(samples, sr, weighted), sr, STEP_BLOCK, STEP_HOP)
    skip = math.ceil(edge / STEP_HOP - 1e-9)
    env = env[skip:len(env) - skip] if skip else env
    peak = true_peak(samples, sr)
    if len(env) < 2 * STEP_SIDE + STEP_GAP:
        frames = 2 * STEP_SIDE + STEP_GAP + 2 * skip
        need = round(STEP_BLOCK + (frames - 1) * STEP_HOP, 2)
        return StepsResult(0.0, None, 0.0, 0.0, peak, max_step, True,
                           (f"too short to measure level steps (needs at least {need:g}s)",))
    range_db = float(env.max() - env.min())
    transient = float(np.abs(env[STEP_TRANSIENT:] - env[:-STEP_TRANSIENT]).max())
    floored = np.maximum(env, STEP_FLOOR_LKFS)
    c = np.concatenate([[0.0], np.cumsum(floored)])
    i = np.arange(STEP_SIDE, len(env) - STEP_GAP - STEP_SIDE + 1)
    before = (c[i] - c[i - STEP_SIDE]) / STEP_SIDE
    after = (c[i + STEP_GAP + STEP_SIDE] - c[i + STEP_GAP]) / STEP_SIDE
    steps = np.abs(after - before)
    if steps.max() < 1e-9:   # digital silence: every boundary ties, so no place is the worst
        return StepsResult(0.0, None, range_db, transient, peak, max_step, True,
                           ("no level change anywhere",))
    worst = int(np.argmax(steps))
    step = float(steps[worst])
    # the join sits between the end of the last block before the gap and the start of the first after
    last_before_ends = (skip + i[worst] - 1) * STEP_HOP + STEP_BLOCK
    first_after_starts = (skip + i[worst] + STEP_GAP) * STEP_HOP
    at = round(float(last_before_ends + first_after_starts) / 2, 2)
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
