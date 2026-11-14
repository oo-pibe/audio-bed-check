import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { findFfmpeg, findRemotionFfmpeg, samplesFromWav } from "../src/decode.js";
import { DecodeError } from "../src/errors.js";

function riff(chunks) {
  const body = Buffer.concat(chunks.map(([id, data]) => {
    const head = Buffer.alloc(8);
    head.write(id, 0, "latin1");
    head.writeUInt32LE(data.length, 4);
    return Buffer.concat([head, data, data.length % 2 ? Buffer.alloc(1) : Buffer.alloc(0)]);
  }));
  const head = Buffer.alloc(12);
  head.write("RIFF", 0, "latin1"); head.writeUInt32LE(4 + body.length, 4); head.write("WAVE", 8, "latin1");
  return Buffer.concat([head, body]);
}

function fmt(channels) {
  const f = Buffer.alloc(16);
  f.writeUInt16LE(1, 0); f.writeUInt16LE(channels, 2); f.writeUInt32LE(48000, 4);
  f.writeUInt32LE(48000 * 2 * channels, 8); f.writeUInt16LE(2 * channels, 12); f.writeUInt16LE(16, 14);
  return f;
}

const pcm = (...values) => {
  const b = Buffer.alloc(values.length * 2);
  values.forEach((v, i) => b.writeInt16LE(v, 2 * i));
  return b;
};

test("reads interleaved stereo into two channels", () => {
  const [samples, count] = samplesFromWav(riff([["fmt ", fmt(2)], ["data", pcm(16384, -16384, 32767, -32768)]]));
  assert.equal(count, 2);
  assert.deepEqual(Array.from(samples[0]), [0.5, 32767 / 32768]);
  assert.deepEqual(Array.from(samples[1]), [-0.5, -1]);
});

test("walks the chunks: the word data inside an INFO chunk is not the data chunk", () => {
  const info = Buffer.from("INFOICMT\x08\x00\x00\x00data....", "latin1");
  const [samples] = samplesFromWav(riff([["fmt ", fmt(1)], ["LIST", info], ["data", pcm(16384)]]));
  assert.deepEqual(Array.from(samples[0]), [0.5]);
});

test("ignores the data chunk's size (ffmpeg writes a placeholder) and drops a partial frame", () => {
  const wav = riff([["fmt ", fmt(2)], ["data", pcm(1, 2, 3)]]);
  const [samples] = samplesFromWav(wav);
  assert.equal(samples[0].length, 1);
});

test("rejects what is not a WAV", () => {
  assert.throws(() => samplesFromWav(Buffer.from("hello world, not riff")), DecodeError);
  assert.throws(() => samplesFromWav(riff([["data", pcm(1)]])), DecodeError); // data before fmt
  assert.throws(() => samplesFromWav(Buffer.alloc(0)), DecodeError);
});

test("finds Remotion's ffmpeg from a project subfolder, by path only", () => {
  const root = mkdtempSync(path.join(tmpdir(), "abc-remotion-"));
  const { platform, arch } = process;
  const pkg = platform === "darwin" ? `compositor-darwin-${arch}`
    : platform === "linux" ? `compositor-linux-${arch}-gnu` : "compositor-win32-x64-msvc";
  const dir = path.join(root, "node_modules", "@remotion", pkg);
  mkdirSync(dir, { recursive: true });
  const exe = path.join(dir, platform === "win32" ? "ffmpeg.exe" : "ffmpeg");
  writeFileSync(exe, "#!/bin/sh\n");
  chmodSync(exe, 0o755);
  const sub = path.join(root, "src", "deep");
  mkdirSync(sub, { recursive: true });
  assert.equal(findRemotionFfmpeg(sub), exe);
  assert.equal(findRemotionFfmpeg(tmpdir()), null);
});

test("a relative PATH entry runs the ffmpeg it found, not one resolved again from its folder", () => {
  const root = realpathSync(mkdtempSync(path.join(tmpdir(), "abc-relpath-")));
  mkdirSync(path.join(root, "bin", "bin"), { recursive: true });
  const exe = process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
  for (const p of [path.join(root, "bin", exe), path.join(root, "bin", "bin", exe)]) {
    writeFileSync(p, "#!/bin/sh\n");
    chmodSync(p, 0o755);
  }
  const cwd = process.cwd(), saved = process.env.PATH;
  try {
    process.chdir(root);
    process.env.PATH = "bin";
    assert.equal(findFfmpeg(), path.join(root, "bin", exe));
  } finally {
    process.chdir(cwd);
    process.env.PATH = saved;
  }
});
