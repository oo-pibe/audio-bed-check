// The three checks. Pure functions of channel arrays and a rate; nothing here touches files or
// ffmpeg. A port of the Python package's checks.py: same constants, same algorithm, same messages.

import { NoSpeechError, ValueError } from "./errors.js";
import { HOP, WINDOW, blockLoudness, channels, kWeight, truePeak } from "./loudness.js";
import { mean, percentile, pyFixed, pyG, pyRound, pyRoundInt } from "./pymath.js";

// seconds; under this the 400 ms momentary window correlates neighbouring frames by itself
export const LOOP_SHORTEST_LAG = 0.5;
// a loop correlates at every multiple of its period; the fundamental is the shortest peak within
// this of the strongest, so a half-period peak cannot name the repeat
export const LOOP_PEAK_TOLERANCE = 0.05;
export const LOOP_WINDOW = 100; // frames (10 s) of envelope changes correlated at a time
export const LOOP_MIN_WINDOW = 30; // frames; files under 30 s use a third of their length, never less
export const LOOP_STRIDE = 5; // frames between window starts

function weightedOf(samples, sr, weighted) {
  if (weighted == null) return kWeight(samples, sr);
  const x = channels(samples);
  const w = channels(weighted);
  if (w.length !== x.length || w[0].length !== x[0].length) {
    throw new ValueError("weighted must be kWeight(samples, sr): same shape as samples");
  }
  return weighted;
}

function ptp(a) {
  let lo = Infinity, hi = -Infinity;
  for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return a.length ? hi - lo : 0;
}

/**
 * For each lag k in first..last, the best Pearson correlation of d[s:s+window] with
 * d[s+k:s+k+window] over window starts s every LOOP_STRIDE frames. A window with no variation
 * cannot win (scores -1).
 */
function lagScores(d, window, first, last) {
  const m = d.length;
  const cumD = new Float64Array(m + 1);
  const cumD2 = new Float64Array(m + 1);
  for (let i = 0; i < m; i++) {
    cumD[i + 1] = cumD[i] + d[i];
    cumD2[i + 1] = cumD2[i] + d[i] * d[i];
  }
  const scores = new Float64Array(last - first + 1).fill(-1);
  const cumProd = new Float64Array(m + 1);
  for (let k = first; k <= last; k++) {
    for (let i = 0; i < m - k; i++) cumProd[i + 1] = cumProd[i] + d[i] * d[i + k];
    let best = -Infinity, any = false;
    for (let s = 0; s <= m - k - window; s += LOOP_STRIDE) {
      const sumA = cumD[s + window] - cumD[s];
      const sumB = cumD[s + k + window] - cumD[s + k];
      const ssA = cumD2[s + window] - cumD2[s] - (sumA * sumA) / window;
      const ssB = cumD2[s + k + window] - cumD2[s + k] - (sumB * sumB) / window;
      if (!(ssA > 1e-12 && ssB > 1e-12)) continue;
      const cross = cumProd[s + window] - cumProd[s] - (sumA * sumB) / window;
      const r = cross / Math.sqrt(ssA * ssB);
      any = true;
      if (r > best) best = r;
    }
    if (any) scores[k - first] = best;
  }
  return scores;
}

function loopResult(score, periodS, threshold, minPeriodS, passed, notes = []) {
  return { score, period_s: periodS, threshold, min_period_s: minPeriodS, passed, notes };
}

/**
 * Does the bed repeat itself? See the Python loop_score docstring for the full method; in short:
 * the best 10 s-window Pearson correlation of the momentary envelope's frame-to-frame changes at
 * every lag from 0.5 s, the fundamental taken as the shortest strong peak.
 */
export function loopScore(samples, sr, { minPeriod = 6.0, threshold = 0.9, weighted = null } = {}) {
  if (!(minPeriod > 0)) throw new ValueError("min_period must be positive");
  if (!(threshold > 0 && threshold <= 1)) {
    throw new ValueError("threshold must be between 0 (exclusive) and 1");
  }
  const x = channels(samples);
  const tooShort = loopResult(0.0, null, threshold, minPeriod, true, [
    `too short to test for a repeat longer than ${pyG(minPeriod)}s`,
  ]);
  if (x[0].length / sr < 2 * minPeriod + 1) return tooShort;
  const env = blockLoudness(weightedOf(samples, sr, weighted), sr, WINDOW, HOP);
  if (ptp(env) < 0.01) {
    // silence or a constant tone; a steady bed still varies by tenths of a dB
    return loopResult(0.0, null, threshold, minPeriod, true, ["level is constant; nothing to correlate"]);
  }
  const d = new Float64Array(env.length - 1);
  for (let i = 0; i < d.length; i++) d[i] = env[i + 1] - env[i];
  const window = Math.min(LOOP_WINDOW, Math.max(LOOP_MIN_WINDOW, Math.floor(env.length / 3)));
  const first = pyRoundInt(LOOP_SHORTEST_LAG / HOP);
  const lo = pyRoundInt(minPeriod / HOP);
  const last = d.length - window;
  if (last < Math.max(lo, first + 2)) return tooShort;
  const scores = lagScores(d, window, first, last);
  const n = scores.length;
  // local maxima; never the first lag, and the last only if it is at least its neighbour. A lag
  // where no window had any variation scores -1; a run of those is flat, not a peak
  const isPeak = new Array(n).fill(false);
  for (let i = 1; i < n - 1; i++) isPeak[i] = scores[i] >= scores[i - 1] && scores[i] >= scores[i + 1];
  isPeak[n - 1] = scores[n - 1] >= scores[n - 2];
  for (let i = 0; i < n; i++) isPeak[i] = isPeak[i] && scores[i] > -1.0;
  const peaks = isPeak.map((p, i) => p && scores[i] >= threshold);
  if (!peaks.some(Boolean)) {
    let score = 0.0, found = false;
    for (let i = 0; i < n; i++) {
      if (isPeak[i] && first + i >= lo && (!found || scores[i] > score)) { score = scores[i]; found = true; }
    }
    if (!found) score = 0.0;
    return loopResult(Math.min(Math.max(score, 0.0), 1.0), null, threshold, minPeriod, true);
  }
  let best = -Infinity;
  for (let i = 0; i < n; i++) if (peaks[i] && scores[i] > best) best = scores[i];
  let i = 0;
  while (!(peaks[i] && scores[i] >= best - LOOP_PEAK_TOLERANCE)) i++;
  const k = first + i;
  const period = pyRound(k * HOP, 2);
  const passed = k < lo;
  const notes = passed
    ? [`repeats every ${pyFixed(period, 1)}s, under --min-period ${pyG(minPeriod)}s, not flagged`]
    : [];
  return loopResult(Math.min(scores[i], 1.0), period, threshold, minPeriod, passed, notes);
}

export const STEP_BLOCK = 0.5; // seconds; the scale a listener hears as "the sound changed"
export const STEP_HOP = 0.1; // seconds between blocks, so a join is never more than 0.05 s from a boundary
export const STEP_SIDE = 20; // frames (2 s) averaged on each side of a boundary for the sustained step
export const STEP_GAP = 5; // frames (0.5 s) skipped at the boundary, so no block straddles the join
export const STEP_TRANSIENT = pyRoundInt(STEP_BLOCK / STEP_HOP);
export const STEP_FLOOR_LKFS = -70.0; // blocks under this count as silence for the step
export const STEP_TIE_DB = 1e-6; // steps this close to the largest count as tied
export const SILENT_DBTP = -60.0; // a file peaking at or under this is reported as silent
export const SILENT_NOTE = "the file is silent (peak at or under -60 dBTP)";

function stepsResult(stepDb, stepAtS, rangeDb, transientDb, peakDbtp, maxStep, passed, notes = []) {
  return {
    step_db: stepDb, step_at_s: stepAtS, range_db: rangeDb, transient_db: transientDb,
    peak_dbtp: peakDbtp, max_step: maxStep, passed, notes,
  };
}

/**
 * Does the level lurch? Half-second K-weighted blocks every 0.1 s; at every boundary the mean of
 * the 2 s of blocks after a 0.5 s gap against the 2 s before it. Only that sustained step fails;
 * range and transient are reported, never failed.
 */
export function levelSteps(samples, sr, { maxStep = 6.0, edge = 0.75, weighted = null } = {}) {
  if (!(edge >= 0)) throw new ValueError("edge must be >= 0");
  const full = blockLoudness(weightedOf(samples, sr, weighted), sr, STEP_BLOCK, STEP_HOP);
  const skip = Math.ceil(edge / STEP_HOP - 1e-9);
  const env = skip ? full.subarray(skip, Math.max(skip, full.length - skip)) : full;
  const peak = truePeak(samples, sr);
  const silent = peak <= SILENT_DBTP ? [SILENT_NOTE] : [];
  if (env.length < 2 * STEP_SIDE + STEP_GAP) {
    const frames = 2 * STEP_SIDE + STEP_GAP + 2 * skip;
    const need = pyRound(STEP_BLOCK + (frames - 1) * STEP_HOP, 2);
    return stepsResult(0.0, null, 0.0, 0.0, peak, maxStep, true, [
      `too short to measure level steps (needs at least ${pyG(need)}s)`, ...silent,
    ]);
  }
  let hi = -Infinity, lo = Infinity;
  for (const v of env) { if (v > hi) hi = v; if (v < lo) lo = v; }
  const rangeDb = hi - lo;
  let transient = 0;
  for (let i = STEP_TRANSIENT; i < env.length; i++) {
    transient = Math.max(transient, Math.abs(env[i] - env[i - STEP_TRANSIENT]));
  }
  const c = new Float64Array(env.length + 1);
  for (let i = 0; i < env.length; i++) c[i + 1] = c[i] + Math.max(env[i], STEP_FLOOR_LKFS);
  const first = STEP_SIDE;
  const steps = new Float64Array(env.length - STEP_GAP - STEP_SIDE - first + 1);
  let step = -Infinity;
  for (let k = 0; k < steps.length; k++) {
    const i = first + k;
    const before = (c[i] - c[i - STEP_SIDE]) / STEP_SIDE;
    const after = (c[i + STEP_GAP + STEP_SIDE] - c[i + STEP_GAP]) / STEP_SIDE;
    steps[k] = Math.abs(after - before);
    if (steps[k] > step) step = steps[k];
  }
  // a looped bed has the same step at every repeat, equal to rounding noise; report the earliest
  // within STEP_TIE_DB of the largest, so the position does not depend on that noise
  let worstAt = first;
  while (steps[worstAt - first] < step - STEP_TIE_DB) worstAt++;
  if (step < 1e-9) {
    // every boundary ties, so no place is the worst; the floor flattened a varying file, or the
    // file really is flat (digital zeros)
    const flooredFlat = hi <= STEP_FLOOR_LKFS && rangeDb > 1e-9;
    const note = flooredFlat ? "every block is under -70 LKFS; nothing to measure" : "no level change anywhere";
    return stepsResult(0.0, null, rangeDb, transient, peak, maxStep, true, [note, ...silent]);
  }
  // the join sits between the end of the last block before the gap and the start of the first after
  const lastBeforeEnds = (skip + worstAt - 1) * STEP_HOP + STEP_BLOCK;
  const firstAfterStarts = (skip + worstAt + STEP_GAP) * STEP_HOP;
  const at = pyRound((lastBeforeEnds + firstAfterStarts) / 2, 2);
  return stepsResult(step, at, rangeDb, transient, peak, maxStep, step <= maxStep, silent);
}

export const SEP_GATE_HOP = 0.02; // seconds; RMS resolution for finding speech in the voiceover
export const SEP_MIN_RUN = 0.25; // seconds; shortest stretch above the gate that counts as speech
export const SEP_GUARD = 0.25; // seconds trimmed from each end of a gap
export const SEP_MIN_GAP = 0.4; // seconds; shortest usable bed-only window
export const SEP_BLOCK = 0.05; // seconds; a block never straddles the edge of a 250 ms run
export const SEP_SILENT_LKFS = -70.0; // bed-only windows at or under this are silence
export const SEP_ALIGN_TOLERANCE = 0.2; // seconds; an estimated start further than this is named
export const SEP_ALIGN_TIE = 0.02; // the given offset stands when within 2% of the best match
export const SEP_ALIGN_MIN_R = 0.5; // the estimate is trusted only at this Pearson r or more

/** [start, end] seconds of every stretch of the voiceover above the gate for at least SEP_MIN_RUN. */
export function speechRuns(vo, sr, gateDbfs = -44.0) {
  const x = channels(vo);
  const h = pyRoundInt(SEP_GATE_HOP * sr);
  const blocks = Math.floor(x[0].length / h);
  if (blocks === 0) return [];
  const shortest = Math.ceil(SEP_MIN_RUN / SEP_GATE_HOP - 1e-9); // hops; 0.25 s is 12.5, so 13
  const runs = [];
  let start = null;
  for (let b = 0; b < blocks; b++) {
    let s = 0;
    for (const ch of x) for (let i = b * h; i < (b + 1) * h; i++) s += ch[i] * ch[i];
    const rms = Math.sqrt(s / (h * x.length));
    const on = rms > 0 && 20 * Math.log10(rms) > gateDbfs;
    if (on && start === null) start = b;
    else if (!on && start !== null) {
      if (b - start >= shortest) runs.push([pyRound(start * SEP_GATE_HOP, 2), pyRound(b * SEP_GATE_HOP, 2)]);
      start = null;
    }
  }
  if (start !== null && blocks - start >= shortest) {
    runs.push([pyRound(start * SEP_GATE_HOP, 2), pyRound(blocks * SEP_GATE_HOP, 2)]);
  }
  return runs;
}

/** Bed-only windows: the space between consecutive runs, trimmed by SEP_GUARD, kept if long enough. */
export function gapsBetween(runs) {
  const gaps = [];
  for (let i = 0; i + 1 < runs.length; i++) {
    const a = runs[i][1] + SEP_GUARD;
    const b = runs[i + 1][0] - SEP_GUARD;
    // one gate hop of slack: off the 20 ms grid a 0.9 s pause measures 0.88 s
    if (b - a >= SEP_MIN_GAP - SEP_GATE_HOP - 1e-9) gaps.push([pyRound(a, 2), pyRound(b, 2)]);
  }
  return gaps;
}

function windowLevel(blocks, t0, t1) {
  const i0 = Math.ceil(t0 / SEP_BLOCK - 1e-9);
  const i1 = Math.floor(t1 / SEP_BLOCK + 1e-9);
  if (i0 < 0 || i1 > blocks.length || i1 <= i0) return null;
  return percentile(blocks.subarray(i0, i1), 90);
}

function pearson(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  return sab / Math.sqrt(saa * sbb);
}

/** Where the voiceover seems to start in the mix, 0.1 s resolution; null when it cannot be trusted. */
function estimateOffset(vo, mixWeighted, sr, runs, voOffset = 0.0) {
  const vRaw = blockLoudness(kWeight(vo, sr), sr, WINDOW, HOP).map((v) => Math.max(v, SEP_SILENT_LKFS));
  const mRaw = blockLoudness(mixWeighted, sr, WINDOW, HOP).map((v) => Math.max(v, SEP_SILENT_LKFS));
  if (vRaw.length === 0 || mRaw.length === 0) return null;
  const inside = new Array(vRaw.length).fill(false);
  for (let i = 0; i < vRaw.length; i++) {
    const centre = i * HOP + WINDOW / 2;
    for (const [a, b] of runs) if (centre >= a && centre < b) { inside[i] = true; break; }
  }
  const vMean = mean(vRaw), mMean = mean(mRaw);
  const v = vRaw.map((x, i) => (inside[i] ? x - vMean : 0.0));
  const m = mRaw.map((x) => x - mMean);
  const Lv = v.length, Lm = m.length;
  // lag runs over -(Lv-1)..(Lm-1), kept where |lag| <= Lm: m[j + lag] against v[j]
  const lag0 = Math.max(-(Lv - 1), -Lm);
  const lag1 = Math.min(Lm - 1, Lm);
  const corr = new Float64Array(lag1 - lag0 + 1);
  for (let lag = lag0; lag <= lag1; lag++) {
    let s = 0;
    const j0 = Math.max(0, -lag), j1 = Math.min(Lv, Lm - lag);
    for (let j = j0; j < j1; j++) s += m[j + lag] * v[j];
    corr[lag - lag0] = s;
  }
  let best = 0;
  for (let i = 1; i < corr.length; i++) if (corr[i] > corr[best]) best = i;
  const given = pyRoundInt(voOffset / HOP) - lag0;
  if (given >= 0 && given < corr.length && corr[given] >= corr[best] - SEP_ALIGN_TIE * Math.abs(corr[best])) {
    best = given;
  }
  const lag = lag0 + best;
  const a = [], b = [];
  for (let j = 0; j < Lv; j++) {
    if (inside[j] && j + lag >= 0 && j + lag < mRaw.length) { a.push(vRaw[j]); b.push(mRaw[j + lag]); }
  }
  if (a.length < 3) return null;
  if (ptp(a) < 1e-9 || ptp(b) < 1e-9) return null;
  if (pearson(a, b) < SEP_ALIGN_MIN_R) return null;
  return pyRound(lag * HOP, 2);
}

/**
 * Is the voice on top of the bed? Speech windows come from the voiceover on its own; each is
 * measured in the MIX and compared with the bed-only gaps between them (90th percentile of 50 ms
 * K-weighted blocks per window).
 */
export function separation(vo, mix, sr, { minLu = 10.0, gateDbfs = -44.0, voOffset = 0.0 } = {}) {
  const runs = speechRuns(vo, sr, gateDbfs);
  if (runs.length === 0) {
    throw new NoSpeechError(`no speech found in the voiceover above ${pyG(gateDbfs)} dBFS`);
  }
  const gaps = gapsBetween(runs);
  if (gaps.length === 0) {
    throw new NoSpeechError(
      `the voiceover has no pause of ${pyG(SEP_MIN_GAP + 2 * SEP_GUARD)}s or more between speech runs ` +
      `(${pyG(SEP_MIN_GAP)}s after a ${pyG(SEP_GUARD)}s guard at each end); either --vo is the mix rather ` +
      "than the voiceover alone, or its noise floor is above --gate",
    );
  }
  const weighted = kWeight(mix, sr); // once: the windows and the offset estimate both read it
  const blocks = blockLoudness(weighted, sr, SEP_BLOCK, SEP_BLOCK);
  const level = ([a, b]) => windowLevel(blocks, a + voOffset, b + voOffset);
  const speech = runs.map(level).filter((v) => v !== null);
  const bed = gaps.map(level).filter((v) => v !== null);
  if (speech.length === 0 || bed.length === 0) {
    throw new NoSpeechError("the voiceover's speech windows fall outside the mix; check --vo-offset");
  }
  const speechLkfs = mean(speech);
  const bedLkfs = mean(bed);
  if (bedLkfs <= SEP_SILENT_LKFS) {
    throw new NoSpeechError("the bed-only windows are silent in the mix; is MIX the rendered mix?");
  }
  const sep = speechLkfs - bedLkfs;
  const peak = truePeak(mix, sr);
  const estimate = estimateOffset(vo, weighted, sr, runs, voOffset);
  const warnings = [];
  const dropped = runs.length + gaps.length - speech.length - bed.length;
  if (dropped) warnings.push(`${dropped} window(s) fall outside the mix; check --vo-offset`);
  if (estimate !== null && Math.abs(estimate - voOffset) > SEP_ALIGN_TOLERANCE + 1e-9) {
    warnings.push(`the voiceover seems to start at ${pyFixed(estimate, 1)}s in the mix; check --vo-offset`);
  }
  if (peak > -1.0) warnings.push("mix true peak above -1 dBTP");
  return {
    separation_lu: sep, speech_lkfs: speechLkfs, bed_lkfs: bedLkfs, runs: runs.length,
    gaps: gaps.length, peak_dbtp: peak, min_lu: minLu, passed: sep >= minLu,
    estimated_offset_s: estimate, warnings,
  };
}
