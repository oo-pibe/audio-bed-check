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
    the strongest local maximum at lags of min_period_s or more, never below 0.0, and period_s is None."""

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
    cum_d = np.concatenate([[0.0], np.cumsum(d)])
    cum_d2 = np.concatenate([[0.0], np.cumsum(d * d)])
    scores = np.full(last - first + 1, -1.0)
    # a loop over lags, all numpy inside: measured 0.3 s for a 10-minute bed, 4 s for an hour
    for i, k in enumerate(range(first, last + 1)):
        s = np.arange(0, m - k - window + 1, LOOP_STRIDE)
        cum_prod = np.concatenate([[0.0], np.cumsum(d[:m - k] * d[k:])])
        sum_a, sum_b = cum_d[s + window] - cum_d[s], cum_d[s + k + window] - cum_d[s + k]
        ss_a = cum_d2[s + window] - cum_d2[s] - sum_a * sum_a / window
        ss_b = cum_d2[s + k + window] - cum_d2[s + k] - sum_b * sum_b / window
        cross = cum_prod[s + window] - cum_prod[s] - sum_a * sum_b / window
        ok = (ss_a > 1e-12) & (ss_b > 1e-12)
        if ok.any():
            scores[i] = float((cross[ok] / np.sqrt(ss_a[ok] * ss_b[ok])).max())
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
    So a repeat is found even when only part of the file loops, but only when the repeating stretch is
    at least one period plus the window long: a 6 s clip played twice is not seen. The longest repeat
    that can be found is the file length minus the window (about 10.5 s less than the file). Lags
    resolve in 0.1 s steps.

    The fundamental is the shortest lag whose score is a local maximum at or above `threshold` and
    within LOOP_PEAK_TOLERANCE of the strongest such peak. Under `min_period` it is music-like (a
    bar) and becomes a note; at or above it, the file is a loop and fails. With no peak at the
    threshold, the score is the strongest local maximum at lags of `min_period` or more (0.0 when
    there is none, or it is negative) and the file passes.

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
    # if it is at least its neighbour, since a repeat can sit right at the end of the range. A lag
    # where no window had any variation scores -1; a run of those is flat, not a peak
    is_peak = np.zeros(len(scores), dtype=bool)
    is_peak[1:-1] = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    is_peak[-1] = scores[-1] >= scores[-2]
    is_peak &= scores > -1.0
    peaks = is_peak & (scores >= threshold)
    if not peaks.any():
        flaggable = is_peak & (lags >= lo)
        score = float(scores[flaggable].max()) if flaggable.any() else 0.0
        return LoopResult(min(max(score, 0.0), 1.0), None, threshold, min_period, True)
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
SILENT_DBTP = -60.0       # a file peaking at or under this is reported as silent; the verdict is unchanged
SILENT_NOTE = "the file is silent (peak at or under -60 dBTP)"


@dataclass(frozen=True)
class StepsResult:
    """step_db: largest change in mean level between the 2 s before a join and the 2 s after it, at
    step_at_s (seconds into the file, rounded to 0.01 s); the only field that decides `passed`.
    range_db (max minus min) and transient_db (largest change between half-second blocks 0.5 s apart)
    are reported, never failed. step_at_s is None when the file is too short to measure or the level
    never changes.
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
    cannot fail on its own noise. A file peaking at or under -60 dBTP carries SILENT_NOTE as its last
    note, which the CLI prints as a warning; it does not change `passed`.

    `weighted` is the K-weighted signal (`k_weight(samples, sr)`) if the caller has it already.
    """
    if edge < 0:
        raise ValueError("edge must be >= 0")
    env = block_loudness(_weighted(samples, sr, weighted), sr, STEP_BLOCK, STEP_HOP)
    skip = math.ceil(edge / STEP_HOP - 1e-9)
    env = env[skip:len(env) - skip] if skip else env
    peak = true_peak(samples, sr)
    silent = (SILENT_NOTE,) if peak <= SILENT_DBTP else ()
    if len(env) < 2 * STEP_SIDE + STEP_GAP:
        frames = 2 * STEP_SIDE + STEP_GAP + 2 * skip
        need = round(STEP_BLOCK + (frames - 1) * STEP_HOP, 2)
        return StepsResult(0.0, None, 0.0, 0.0, peak, max_step, True,
                           (f"too short to measure level steps (needs at least {need:g}s)",) + silent)
    range_db = float(env.max() - env.min())
    transient = float(np.abs(env[STEP_TRANSIENT:] - env[:-STEP_TRANSIENT]).max())
    floored = np.maximum(env, STEP_FLOOR_LKFS)
    c = np.concatenate([[0.0], np.cumsum(floored)])
    i = np.arange(STEP_SIDE, len(env) - STEP_GAP - STEP_SIDE + 1)
    before = (c[i] - c[i - STEP_SIDE]) / STEP_SIDE
    after = (c[i + STEP_GAP + STEP_SIDE] - c[i + STEP_GAP]) / STEP_SIDE
    steps = np.abs(after - before)
    if steps.max() < 1e-9:   # every boundary ties, so no place is the worst
        # the floor flattened a varying file; a file that really is flat (digital zeros) says so
        floored_flat = env.max() <= STEP_FLOOR_LKFS and np.ptp(env) > 1e-9
        note = ("every block is under -70 LKFS; nothing to measure" if floored_flat
                else "no level change anywhere")
        return StepsResult(0.0, None, range_db, transient, peak, max_step, True, (note,) + silent)
    worst = int(np.argmax(steps))
    step = float(steps[worst])
    # the join sits between the end of the last block before the gap and the start of the first after
    last_before_ends = (skip + i[worst] - 1) * STEP_HOP + STEP_BLOCK
    first_after_starts = (skip + i[worst] + STEP_GAP) * STEP_HOP
    at = round(float(last_before_ends + first_after_starts) / 2, 2)
    return StepsResult(step, at, range_db, transient, peak, max_step, step <= max_step, silent)


SEP_GATE_HOP = 0.02   # seconds; RMS resolution for finding speech in the voiceover
SEP_MIN_RUN = 0.25    # seconds; shortest stretch above the gate that counts as speech
SEP_GUARD = 0.25      # seconds trimmed from each end of a gap, so tails of speech do not leak in
SEP_MIN_GAP = 0.4     # seconds; shortest usable bed-only window
SEP_BLOCK = 0.05      # seconds; fine enough that a block never straddles the edge of a 250 ms run
SEP_SILENT_LKFS = -70.0   # bed-only windows at or under this are silence: the mix holds no bed
SEP_ALIGN_TOLERANCE = 0.2  # seconds; an estimated start further than this from vo_offset is named
# a read with regular pauses matches the mix almost equally at several lags; the given offset stands
# when its match is within this fraction of the best. The fixture that needed it tied at 0.01%, so
# 2% is a generous margin
SEP_ALIGN_TIE = 0.02
# the estimate is trusted only when, at the lag found, the voiceover's level inside its speech runs
# and the mix's level at the same moments correlate at least this well (Pearson). A voice 10 dB under
# a textured bed read 0.32 or less, at wrong lags up to 14 s away; a voice level with the bed read
# 0.40 to 0.73 and one 10 dB over it 0.99
SEP_ALIGN_MIN_R = 0.5


class NoSpeechError(ValueError):
    """The separation check could not run: no speech windows, no gaps between them, or (with an
    offset) no window inside the mix. The CLI reports it as exit 2, not as a failing mix."""


@dataclass(frozen=True)
class SeparationResult:
    """separation_lu: speech_lkfs minus bed_lkfs, where speech_lkfs is the mean over speech runs of
    each run's 90th-percentile 50 ms K-weighted level in the mix and bed_lkfs the same over the
    bed-only gaps; the only field that decides `passed`. runs/gaps count the windows found in the
    voiceover, before any fall outside the mix. estimated_offset_s is where the voiceover's envelope
    best matches the mix's (0.1 s resolution), None when either is under 0.4 s or the match is too
    weak to trust (a voice buried under the bed). warnings (named so
    because they are actionable, unlike the other checks' notes: a hot peak, windows dropped by the
    offset, an offset that disagrees with the estimate) never fail. Values are full precision; the
    CLI rounds."""

    separation_lu: float
    speech_lkfs: float
    bed_lkfs: float
    runs: int
    gaps: int
    peak_dbtp: float
    min_lu: float
    passed: bool
    estimated_offset_s: float | None = None
    warnings: tuple[str, ...] = ()


# speech_runs and gaps_between are module-public for the tests; they are not re-exported by the package.
def speech_runs(vo: np.ndarray, sr: int, gate_dbfs: float = -44.0) -> list[tuple[float, float]]:
    """(start, end) seconds of every stretch of the voiceover above the gate for at least SEP_MIN_RUN.

    Mono or stereo: the gate reads the RMS over every sample of every channel in a hop, the mean power
    over channels (identical channels read as mono; a voice on one channel reads 3 dB lower), so a
    voice on one channel or in anti-phase still counts. Run lengths are compared in whole hops.
    """
    x = np.asarray(vo, dtype=np.float64)
    h = int(round(SEP_GATE_HOP * sr))
    blocks = len(x) // h
    if blocks == 0:
        return []
    rms = np.sqrt((x[:blocks * h] ** 2).reshape(blocks, -1).mean(axis=1))
    with np.errstate(divide="ignore"):
        loud = 20 * np.log10(rms) > gate_dbfs
    shortest = math.ceil(SEP_MIN_RUN / SEP_GATE_HOP - 1e-9)   # hops; 0.25 s is 12.5, so 13
    runs, start = [], None
    # a plain scan: a 10-minute voiceover is 30,000 hops, nothing next to the K-weighting
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= shortest:
                runs.append((round(start * SEP_GATE_HOP, 2), round(i * SEP_GATE_HOP, 2)))
            start = None
    if start is not None and blocks - start >= shortest:
        runs.append((round(start * SEP_GATE_HOP, 2), round(blocks * SEP_GATE_HOP, 2)))
    return runs


def gaps_between(runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Bed-only windows: the space between consecutive runs, trimmed by SEP_GUARD, kept if SEP_MIN_GAP
    or longer less one gate hop, so a 0.9 s pause is kept wherever it falls on the 20 ms grid."""
    gaps = []
    for (_, end), (start, _) in zip(runs, runs[1:], strict=False):
        a, b = end + SEP_GUARD, start - SEP_GUARD
        # one gate hop of slack: off the 20 ms grid a run's end rounds up a hop and the next start
        # rounds down, so a 0.9 s pause measures 0.88 s
        if b - a >= SEP_MIN_GAP - SEP_GATE_HOP - 1e-9:
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


def _estimate_offset(vo: np.ndarray, mix_weighted: np.ndarray, sr: int, runs: list[tuple[float, float]],
                     vo_offset: float = 0.0) -> float | None:
    """Where the voiceover seems to start in the mix, seconds, at 0.1 s resolution; None when either
    file is shorter than one 400 ms window, or when the match is too weak to trust.

    `mix_weighted` is the mix already K-weighted. Cross-correlates the momentary envelopes (LKFS,
    floored at -70 and with their means removed). The voiceover's is kept only inside its speech runs
    and zero elsewhere, so the match is the shape of the read: louder in the mix where the voice is,
    and the read's own texture. Lags run over plus or minus the mix's length. A read with regular
    pauses matches at several lags almost equally, so when the match at `vo_offset` is within
    SEP_ALIGN_TIE (2%) of the best, `vo_offset` (to the nearest 0.1 s) is the estimate.

    The estimate stands only when, at that lag, the voiceover's envelope inside its runs and the mix's
    at the same moments correlate at SEP_ALIGN_MIN_R (0.5) or more. A voice buried under a textured
    bed otherwise lands on whatever lag the bed's texture happens to match.
    """
    v_raw = np.maximum(block_loudness(k_weight(vo, sr), sr, WINDOW, HOP), SEP_SILENT_LKFS)
    m_raw = np.maximum(block_loudness(mix_weighted, sr, WINDOW, HOP), SEP_SILENT_LKFS)
    if len(v_raw) == 0 or len(m_raw) == 0:
        return None
    centres = np.arange(len(v_raw)) * HOP + WINDOW / 2
    inside = np.zeros(len(v_raw), dtype=bool)
    for a, b in runs:
        inside |= (centres >= a) & (centres < b)
    v = np.where(inside, v_raw - v_raw.mean(), 0.0)
    m = m_raw - m_raw.mean()
    corr = np.correlate(m, v, mode="full")   # index k is lag k - (len(v) - 1): m[j + lag] against v[j]
    lags = np.arange(len(corr)) - (len(v) - 1)
    keep = np.abs(lags) <= len(m)
    corr, lags = corr[keep], lags[keep]
    best = int(np.argmax(corr))
    given = int(round(vo_offset / HOP)) - int(lags[0])
    if 0 <= given < len(corr) and corr[given] >= corr[best] - SEP_ALIGN_TIE * abs(corr[best]):
        best = given
    lag = int(lags[best])
    j = np.flatnonzero(inside)
    j = j[(j + lag >= 0) & (j + lag < len(m_raw))]
    if len(j) < 3:
        return None
    a, b = v_raw[j], m_raw[j + lag]
    if np.ptp(a) < 1e-9 or np.ptp(b) < 1e-9:
        return None
    if float(np.corrcoef(a, b)[0, 1]) < SEP_ALIGN_MIN_R:
        return None
    return round(lag * HOP, 2)


def separation(vo: np.ndarray, mix: np.ndarray, sr: int, *, min_lu: float = 10.0, gate_dbfs: float = -44.0,
               vo_offset: float = 0.0) -> SeparationResult:
    """Is the voice on top of the bed?

    Speech windows come from the voiceover on its own; each is measured in the MIX (voice plus bed)
    and compared with the bed-only gaps between them. The 90th percentile per window, so a breath
    inside a run does not drag the speech figure down. The result is what a listener hears, which is
    also what WCAG G56 describes, not the ratio of the two stems.

    Bed-only windows at or under -70 LKFS mean the mix holds no bed (the voiceover given as the mix,
    usually) and raise NoSpeechError. The voiceover's start in the mix is estimated from the two
    envelopes; when the estimate is trusted and more than 0.2 s from `vo_offset` a warning names it.
    """
    runs = speech_runs(vo, sr, gate_dbfs)
    if not runs:
        raise NoSpeechError(f"no speech found in the voiceover above {gate_dbfs:g} dBFS")
    gaps = gaps_between(runs)
    if not gaps:
        raise NoSpeechError(
            f"the voiceover has no pause of {SEP_MIN_GAP + 2 * SEP_GUARD:g}s or more between speech runs "
            f"({SEP_MIN_GAP:g}s after a {SEP_GUARD:g}s guard at each end); either --vo is the mix rather "
            "than the voiceover alone, or its noise floor is above --gate")
    weighted = k_weight(mix, sr)   # once: the windows and the offset estimate both read it
    blocks = block_loudness(weighted, sr, SEP_BLOCK, SEP_BLOCK)
    speech = [v for v in (_window_level(blocks, a + vo_offset, b + vo_offset) for a, b in runs)
              if v is not None]
    bed = [v for v in (_window_level(blocks, a + vo_offset, b + vo_offset) for a, b in gaps) if v is not None]
    if not speech or not bed:
        raise NoSpeechError("the voiceover's speech windows fall outside the mix; check --vo-offset")
    speech_lkfs, bed_lkfs = float(np.mean(speech)), float(np.mean(bed))
    if bed_lkfs <= SEP_SILENT_LKFS:
        raise NoSpeechError("the bed-only windows are silent in the mix; is MIX the rendered mix?")
    sep = speech_lkfs - bed_lkfs
    peak = true_peak(mix, sr)
    estimate = _estimate_offset(vo, weighted, sr, runs, vo_offset)
    warnings: tuple[str, ...] = ()
    dropped = len(runs) + len(gaps) - len(speech) - len(bed)
    if dropped:
        warnings += (f"{dropped} window(s) fall outside the mix; check --vo-offset",)
    if estimate is not None and abs(estimate - vo_offset) > SEP_ALIGN_TOLERANCE + 1e-9:
        warnings += (f"the voiceover seems to start at {estimate:.1f}s in the mix; check --vo-offset",)
    if peak > -1.0:
        warnings += ("mix true peak above -1 dBTP",)
    return SeparationResult(sep, speech_lkfs, bed_lkfs, len(runs), len(gaps), peak, min_lu, sep >= min_lu,
                            estimate, warnings)
