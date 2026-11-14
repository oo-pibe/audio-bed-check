// ITU-R BS.1770-4 loudness pieces in plain JavaScript: K-weighting, block loudness, momentary
// series, true peak. A port of the Python package's loudness.py; the numbers match it (see
// scripts/parity.py).
//
// Samples are an array of one or two channels, each a Float32Array or Float64Array of the same
// length. Channels are measured separately and their mean squares summed (BS.1770 weights L and R
// 1.0), never downmixed: a downmix cancels an anti-phase pair to silence and halves an uncorrelated
// bed against a centred voice.
//
// Everything is 48 kHz only. The decoder always produces 48 kHz, and the filter coefficients are the
// standard's published 48 kHz set; refusing other rates is cheaper than getting them silently wrong.

import { ValueError } from "./errors.js";
import { pyRoundInt } from "./pymath.js";

export const SR = 48000;
export const HOP = 0.1; // momentary hop, seconds
export const WINDOW = 0.4; // momentary window, seconds
export const FLOOR_LKFS = -100.0;
export const FLOOR_DBTP = -99.0;

// BS.1770-4 Annex 1, 48 kHz: [b0, b1, b2, a1, a2]. Stage 1 is the high shelf, stage 2 the RLB high pass.
const STAGE1 = [1.53512485958697, -2.69169618940638, 1.19839281085285, -1.69065929318241, 0.73248077421585];
const STAGE2 = [1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621];

function checkRate(sr) {
  if (sr !== SR) throw new ValueError("resample to 48 kHz");
}

/** The channels as validated typed arrays; ValueError for anything but 1 or 2 finite channels. */
export function channels(samples) {
  if (!Array.isArray(samples) || samples.length < 1 || samples.length > 2) {
    throw new ValueError("pass mono or stereo samples");
  }
  const n = samples[0]?.length;
  for (const ch of samples) {
    if (!(ch instanceof Float32Array || ch instanceof Float64Array) || ch.length !== n) {
      throw new ValueError("pass mono or stereo samples");
    }
    for (let i = 0; i < ch.length; i++) {
      if (!Number.isFinite(ch[i])) throw new ValueError("samples must be finite floats");
    }
  }
  return samples;
}

function biquad(x, [b0, b1, b2, a1, a2]) {
  const y = new Float64Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    y[i] = yi;
    x2 = x1; x1 = xi; y2 = y1; y1 = yi;
  }
  return y;
}

/**
 * The K-weighted signal (Float64Array channels, same shape), each channel filtered on its own.
 *
 * The zero-initial-state response of the two BS.1770 biquads, run as the recursive filters. The
 * Python version convolves with the cascade's impulse response truncated to 4096 taps; the slowest
 * pole has radius ~0.995, so the two differ by about 1e-9 of the signal.
 */
export function kWeight(samples, sr) {
  checkRate(sr);
  return channels(samples).map((ch) => biquad(biquad(ch, STAGE1), STAGE2));
}

/**
 * Loudness of an already K-weighted signal per block, in LKFS; floored at -100.
 *
 * The channels' mean squares are summed before the log. `window` and `hop` are seconds. Input
 * shorter than one window returns an empty array.
 */
export function blockLoudness(weighted, sr, window, hop) {
  checkRate(sr);
  const x = channels(weighted);
  const w = pyRoundInt(window * sr);
  const h = pyRoundInt(hop * sr);
  if (h < 1) throw new ValueError("hop must be at least one sample");
  const n = x[0].length;
  if (n < w) return new Float64Array(0);
  const cumulative = new Float64Array(n + 1);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const ch of x) s += ch[i] * ch[i];
    acc += s;
    cumulative[i + 1] = acc;
  }
  const count = Math.floor((n - w) / h) + 1;
  const out = new Float64Array(count);
  for (let k = 0; k < count; k++) {
    const s = k * h;
    const meanSquare = (cumulative[s + w] - cumulative[s]) / w;
    const lkfs = meanSquare > 0 ? -0.691 + 10 * Math.log10(meanSquare) : -Infinity;
    out[k] = Math.max(lkfs, FLOOR_LKFS);
  }
  return out;
}

/** BS.1770 momentary loudness: 400 ms windows every 100 ms. */
export function momentary(samples, sr) {
  return blockLoudness(kWeight(samples, sr), sr, WINDOW, HOP);
}

const UP = 4; // oversampling ratio for true peak, per BS.1770-4
const INTERP_HALF = 64; // kernel half-length in the oversampled domain: 16 input samples each side

/** Modified Bessel function of the first kind, order 0, by its power series (what np.i0 computes). */
function besselI0(x) {
  let sum = 1, term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 500; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

/** Windowed-sinc low-pass for a 4x zero-stuffed signal: cutoff at the original Nyquist, DC gain 4. */
function interpKernel() {
  const m = 2 * INTERP_HALF + 1;
  const alpha = (m - 1) / 2;
  const beta = 9.0;
  const h = new Float64Array(m);
  let sum = 0;
  for (let i = 0; i < m; i++) {
    const t = (i - INTERP_HALF) / UP;
    const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
    const r = (i - alpha) / alpha;
    const kaiser = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / besselI0(beta);
    h[i] = sinc * kaiser;
    sum += h[i];
  }
  for (let i = 0; i < m; i++) h[i] *= UP / sum;
  return h;
}

let kernelCache = null;

/**
 * Peak of the 4x oversampled waveform, in dBTP: the loudest channel.
 *
 * A polyphase FIR: filtering the original samples with each of the kernel's four phases gives every
 * oversampled value with no zero-stuffed copy. Full linear convolutions, so the file counts as
 * silent beyond its ends.
 */
export function truePeak(samples, sr) {
  checkRate(sr);
  const x = channels(samples);
  if (x[0].length === 0) return FLOOR_DBTP;
  kernelCache ??= interpKernel();
  // the four phases h[p::4], each padded with zeros to the longest (33 taps). A zero tap adds an
  // exact 0 to the sum, so every output equals the unpadded convolution's.
  const L = Math.ceil(kernelCache.length / UP);
  const h = Array.from({ length: UP }, (_, p) => {
    const taps = new Float64Array(L);
    for (let t = 0; p + UP * t < kernelCache.length; t++) taps[t] = kernelCache[p + UP * t];
    return taps;
  });
  const [h0, h1, h2, h3] = h;
  // no output can exceed the largest |sample| in its window times its phase's sum of |taps|; a
  // little over 1 so float rounding inside a sum can never beat the bound
  let gain = 0;
  for (const taps of h) gain = Math.max(gain, taps.reduce((s, v) => s + Math.abs(v), 0));
  gain *= 1 + 1e-9;
  let peak = 0;
  for (const ch of x) {
    const n = ch.length;
    // the channel with L-1 zeros on each side, so the full convolution needs no bounds checks;
    // the same element type, so the values are unchanged
    const xp = new ch.constructor(n + 2 * (L - 1));
    xp.set(ch, L - 1);
    // output j reads xp[j .. j+L-1]: at most two blocks of L, so its window max is one of two maxima
    const blocks = Math.ceil(xp.length / L);
    const blockMax = new Float64Array(blocks);
    let loudest = 0;
    for (let i = 0; i < xp.length; i++) {
      const a = Math.abs(xp[i]);
      if (a > blockMax[(i / L) | 0]) blockMax[(i / L) | 0] = a;
      if (a > Math.abs(xp[loudest])) loudest = i;
    }
    const at = (j) => {
      const base = j + L - 1;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let t = 0; t < L; t++) {
        const v = xp[base - t];
        s0 += v * h0[t]; s1 += v * h1[t]; s2 += v * h2[t]; s3 += v * h3[t];
      }
      return Math.max(Math.abs(s0), Math.abs(s1), Math.abs(s2), Math.abs(s3));
    };
    // seed with the outputs around the loudest sample, so the bound prunes from the start
    for (let j = Math.max(0, loudest - L + 1); j <= Math.min(n + L - 2, loudest); j++) peak = Math.max(peak, at(j));
    for (let j = 0; j < n + L - 1; j++) {
      const b = (j / L) | 0;
      const windowMax = Math.max(blockMax[b], b + 1 < blocks ? blockMax[b + 1] : 0);
      if (windowMax * gain <= peak) continue;   // cannot beat the peak already found
      const a = at(j);
      if (a > peak) peak = a;
    }
  }
  return peak > 0 ? 20 * Math.log10(peak) : FLOOR_DBTP;
}
