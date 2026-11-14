import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { main, parse, VERSION } from "../src/cli.js";

const BIN = fileURLToPath(new URL("../bin/audio-bed-check.js", import.meta.url));

function run(argv, env = {}) {
  const out = [], err = [];
  const code = main(argv, (s) => out.push(s), (s) => err.push(s));
  return { code, out: out.join("\n"), err: err.join("\n") };
}

/** A 16-bit mono WAV of `seconds` of seeded noise at about -20 dBFS RMS, with a step of `db` halfway. */
function wav(dir, name, seconds, { db = 0, seed = 1, channels = 1 } = {}) {
  const n = Math.round(seconds * 48000);
  const data = Buffer.alloc(44 + n * 2 * channels);
  data.write("RIFF", 0, "latin1"); data.writeUInt32LE(36 + n * 2 * channels, 4); data.write("WAVE", 8, "latin1");
  data.write("fmt ", 12, "latin1"); data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20);
  data.writeUInt16LE(channels, 22); data.writeUInt32LE(48000, 24); data.writeUInt32LE(48000 * 2 * channels, 28);
  data.writeUInt16LE(2 * channels, 32); data.writeUInt16LE(16, 34);
  data.write("data", 36, "latin1"); data.writeUInt32LE(n * 2 * channels, 40);
  let s = seed;
  const rand = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
  for (let i = 0; i < n; i++) {
    const g = i >= n / 2 ? 10 ** (db / 20) : 1;
    for (let c = 0; c < channels; c++) {
      data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, rand() * 0.17 * g)) * 32767), 44 + 2 * (i * channels + c));
    }
  }
  const file = path.join(dir, name);
  writeFileSync(file, data);
  return file;
}

const dir = mkdtempSync(path.join(tmpdir(), "abc-cli-"));
const hasFfmpeg = spawnSync("ffmpeg", ["-version"]).status === 0;

test("--version and --help exit 0", () => {
  assert.deepEqual(run(["--version"]), { code: 0, out: `audio-bed-check ${VERSION}`, err: "" });
  assert.equal(run(["--help"]).code, 0);
  assert.match(run(["bed", "--help"]).out, /--min-period S/);
});

test("usage errors exit 2 before any file is read", () => {
  for (const argv of [
    [], ["nope"], ["bed"], ["separation", "mix.wav"], ["bed", "a.wav", "--max-step", "0"],
    ["bed", "a.wav", "--max-step", "nan"], ["bed", "a.wav", "--max-step", "inf"], ["bed", "a.wav", "--max-step", ""],
    ["loop", "a.wav", "--loop-threshold", "1.5"], ["steps", "a.wav", "--edge", "-1"], ["bed", "a.wav", "--bogus"],
    ["separation", "m.wav", "--vo", "v.wav", "--profile", "loud"], ["bed", "a.wav", "--json=yes"],
    ["bed", "a.wav", "--max-step"],
  ]) {
    const r = run(argv);
    assert.equal(r.code, 2, argv.join(" "));
    assert.match(r.err, /audio-bed-check( \w+)?: error: /, argv.join(" "));
  }
});

test("flags parse the way argparse does", () => {
  assert.equal(parse(["bed", "a.wav", "--max-step=2.5"]).maxStep, 2.5);
  assert.equal(parse(["bed", "a.wav", "--max-s", "3"]).maxStep, 3); // unique prefix
  assert.equal(parse(["separation", "m.wav", "--vo", "v.wav", "--gate", "-30"]).gate, -30);
  assert.equal(parse(["separation", "m.wav", "--vo", "v.wav", "--vo-offset", "-1.5"]).voOffset, -1.5);
  assert.equal(parse(["bed", "a.wav", "--edge", "1_0"]).edge, 10);
  assert.equal(parse(["bed", "a.wav", "--edge", " 2 "]).edge, 2);
  assert.deepEqual(parse(["bed", "--", "-weird.wav"]).files, ["-weird.wav"]);
  assert.throws(() => parse(["bed", "a.wav", "--m", "3"]), /ambiguous/);
  assert.equal(parse(["separation", "m.wav", "--vo", "v.wav", "--min", "3"]).minSeparation, 3);
  assert.throws(() => parse(["separation", "m.wav", "n.wav", "--vo", "v.wav"]), /unrecognized/);
});

test("a missing file is an error line and exit 2, the rest of the batch still runs", { skip: !hasFfmpeg }, () => {
  const ok = wav(dir, "flat.wav", 20);
  const r = run(["steps", ok, path.join(dir, "missing.wav")]);
  assert.equal(r.code, 2);
  assert.match(r.out, /ok {4}step/);
  assert.match(r.out, /error .*missing\.wav: no such file/);
});

test("a step fails with exit 1 and names where it is", { skip: !hasFfmpeg }, () => {
  const stepped = wav(dir, "step.wav", 20, { db: 9 });
  const r = run(["steps", stepped]);
  assert.equal(r.code, 1);
  assert.match(r.out, /FAIL {2}step {8}9\.\d\d dB at 10\.0\ds/);
  const j = JSON.parse(run(["steps", stepped, "--json"]).out);
  assert.equal(j[0].passed, false);
  assert.equal(j[0].steps.step_at_s, 10.05);
});

test("stereo is read as two channels", { skip: !hasFfmpeg }, () => {
  const r = run(["steps", wav(dir, "stereo.wav", 20, { channels: 2 }), "--json"]);
  assert.equal(r.code, 0);
});

test("a file that is not audio is a decode error", { skip: !hasFfmpeg }, () => {
  const junk = path.join(dir, "junk.wav");
  writeFileSync(junk, "not audio at all");
  const r = run(["bed", junk]);
  assert.equal(r.code, 2);
  assert.match(r.out, /ffmpeg could not decode it/);
});

test("--ffmpeg pointing nowhere is exit 2 with a clear message", () => {
  const r = run(["bed", "a.wav", "--ffmpeg", path.join(dir, "no-ffmpeg-here")]);
  assert.equal(r.code, 2);
  assert.match(r.out + r.err, /ffmpeg not found at/);
});

test("the bin script runs and passes the exit code through", { skip: !hasFfmpeg }, () => {
  const p = spawnSync(process.execPath, [BIN, "steps", wav(dir, "bin.wav", 20, { db: 9 })]);
  assert.equal(p.status, 1);
  const v = spawnSync(process.execPath, [BIN, "--version"]);
  assert.equal(v.stdout.toString().trim(), `audio-bed-check ${VERSION}`);
});
