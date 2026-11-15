// Known answers from outside this package (BS.1770's reference tone, the inter-sample peak of a
// quarter-rate sine) and from signals built to have one fault of a known size. The comparison with
// the Python package (scripts/parity.py) is separate; these need nothing but Node.
import assert from "node:assert/strict";
import { test } from "node:test";
import { blockLoudness, kWeight, levelSteps, loopScore, separation, truePeak } from "../src/index.js";

const SR = 48000;

function seeded(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** Gaussian noise at `rms` (linear), seconds long. */
function noise(seconds, seed, rms = 0.05) {
  const rand = seeded(seed), n = Math.round(seconds * SR), x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.max(rand(), 1e-12), v = rand();
    x[i] = rms * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return x;
}

/** Noise whose level wanders slowly (about 3 dB), so a repeat of it is a real pattern. */
function textured(seconds, seed) {
  const x = noise(seconds, seed), rand = seeded(seed + 7);
  const knots = Array.from({ length: Math.ceil(seconds) + 2 }, () => 10 ** ((rand() - 0.5) * 3 / 20));
  for (let i = 0; i < x.length; i++) {
    const t = i / SR, k = Math.floor(t), f = t - k;
    x[i] *= knots[k] * (1 - f) + knots[k + 1] * f;
  }
  return x;
}

const db = (g) => 10 ** (g / 20);

test("BS.1770 reference: a 997 Hz sine at full scale on one channel reads -3.01 LKFS", () => {
  const n = 4 * SR, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((2 * Math.PI * 997 * i) / SR);
  const blocks = blockLoudness(kWeight([x], SR), SR, 0.4, 0.1);
  const mid = blocks[Math.floor(blocks.length / 2)];
  assert.ok(Math.abs(mid - -3.01) < 0.05, `read ${mid}`);
});

test("true peak finds the peak between samples: a quarter-rate sine sampled at 45 degrees", () => {
  // every sample is +-0.707 (-3.01 dBFS), the waveform itself reaches 1.0 (0 dBTP)
  const n = SR, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((Math.PI / 2) * i + Math.PI / 4);
  const peak = truePeak([x], SR);
  assert.ok(Math.abs(peak) < 0.2, `read ${peak} dBTP`);
});

test("a +8 dB join is measured as about 8 dB, near where it is", () => {
  const x = noise(30, 1);
  for (let i = 15 * SR; i < x.length; i++) x[i] *= db(8);
  const r = levelSteps([x], SR);
  assert.equal(r.passed, false);
  assert.ok(Math.abs(r.step_db - 8) < 0.3, `step ${r.step_db}`);
  assert.ok(Math.abs(r.step_at_s - 15) < 0.3, `at ${r.step_at_s}`);
});

test("steady noise has no step", () => {
  const r = levelSteps([noise(30, 2)], SR);
  assert.equal(r.passed, true);
  assert.ok(r.step_db < 1, `step ${r.step_db}`);
});

test("a 7 s clip looped four times fails at a 7 s period; fresh material passes", () => {
  const clip = textured(7, 3), x = new Float32Array(4 * clip.length);
  for (let k = 0; k < 4; k++) x.set(clip, k * clip.length);
  const looped = loopScore([x], SR);
  assert.equal(looped.passed, false);
  assert.ok(Math.abs(looped.period_s - 7) < 0.2, `period ${looped.period_s}`);
  assert.equal(loopScore([textured(28, 4)], SR).passed, true);
});

/** A voice in runs of 2 s with 1 s pauses, `over` dB above a textured bed, starting at `at` s. */
function mix(over, at = 0) {
  const seconds = 30, vo = noise(seconds, 5, 0.01 * db(over));
  for (let i = 0; i < vo.length; i++) if ((i / SR) % 3 >= 2) vo[i] = 0;
  const bed = textured(seconds + at, 6);
  const scale = 0.01 / Math.sqrt(bed.reduce((s, v) => s + v * v, 0) / bed.length);
  const m = new Float32Array(bed.length);
  for (let i = 0; i < m.length; i++) m[i] = bed[i] * scale;
  for (let i = 0; i < vo.length; i++) m[i + at * SR] += vo[i];
  return [[vo], [m]];
}

test("separation: a voice 15 dB over the bed passes at 10 LU, one level with it fails", () => {
  const [vo, m] = mix(15);
  const r = separation(vo, m, SR);
  assert.equal(r.passed, true);
  // speech windows hold voice plus bed: 10*log10(1 + 10**1.5) above the bed, about 15.1 LU
  assert.ok(Math.abs(r.separation_lu - 15.1) < 1.5, `read ${r.separation_lu}`);
  const [vo2, m2] = mix(0);
  assert.equal(separation(vo2, m2, SR).passed, false);
});

test("separation finds where the voice starts when no offset is given", () => {
  const [vo, m] = mix(15, 4);
  const r = separation(vo, m, SR);
  assert.equal(r.offset_found, true);
  assert.ok(Math.abs(r.vo_offset_s - 4) < 0.06, `found ${r.vo_offset_s}`);
  assert.equal(r.passed, true);
  assert.deepEqual(r.warnings, []);
  // a wrong offset given on purpose measures the wrong windows, and the estimate is named
  const wrong = separation(vo, m, SR, { voOffset: 0 });
  assert.ok(wrong.warnings.some((w) => w.includes("seems to start at 4.0s")));
});

test("a file that is one clip played twice fails; under twice --min-period it cannot loop", () => {
  for (const p of [6.0, 6.2]) {
    const clip = textured(p, 70), x = new Float32Array(2 * clip.length);
    x.set(clip, 0);
    x.set(clip, clip.length);
    const r = loopScore([x], SR);
    assert.equal(r.passed, false, `period ${p}`);
    assert.ok(Math.abs(r.period_s - p) < 0.15, `period ${r.period_s}`);
  }
  const short = loopScore([textured(10, 5)], SR);
  assert.equal(short.passed, true);
  assert.deepEqual(short.notes, ["shorter than 12s, so no repeat of 6s or more can play twice"]);
  const unmeasured = loopScore([textured(4.5, 9)], SR, { minPeriod: 2 });
  assert.deepEqual(unmeasured.notes, ["too short to test for a repeat of 2s or more"]);
  // under one 400 ms window there is no envelope at all; never "level is constant", never a pass
  assert.deepEqual(loopScore([textured(0.3, 9)], SR, { minPeriod: 0.1 }).notes,
    ["too short to test for a repeat of 0.1s or more"]);
});
