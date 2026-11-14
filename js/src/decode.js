// The only module that runs a subprocess. One ffmpeg call per file, samples piped back, no temp
// files, no shell: the arguments go to ffmpeg as a list.

import { spawnSync } from "node:child_process";
import { accessSync, constants, existsSync, statSync } from "node:fs";
import { constants as osConstants } from "node:os";
import path from "node:path";
import { DecodeError } from "./errors.js";

export const SR = 48000;

function isExecutable(file) {
  try {
    if (!statSync(file).isFile()) return false;
    accessSync(file, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** shutil.which: a name with a directory part is checked as given, a bare name is looked up on PATH. */
function which(candidate) {
  const exts = process.platform === "win32"
    ? ["", ...(process.env.PATHEXT || ".EXE;.CMD;.BAT;.COM").split(";").filter(Boolean)]
    : [""];
  if (candidate.includes("/") || (process.platform === "win32" && candidate.includes("\\"))) {
    for (const ext of exts) if (isExecutable(candidate + ext)) return path.resolve(candidate + ext);
    return null;
  }
  for (const dir of (process.env.PATH || "").split(path.delimiter)) {
    if (!dir) continue;
    for (const ext of exts) {
      // absolute: ffmpeg runs from its own folder, so a relative PATH entry would resolve twice
      const full = path.resolve(dir, candidate + ext);
      if (isExecutable(full)) return full;
    }
  }
  return null;
}

/** Remotion's compositor package names that can exist on this machine (one is installed). */
function remotionPackages() {
  const { platform, arch } = process;
  if (platform === "darwin") return [`compositor-darwin-${arch}`];
  if (platform === "win32" && arch === "x64") return ["compositor-win32-x64-msvc"];
  if (platform === "linux" && (arch === "x64" || arch === "arm64")) {
    return [`compositor-linux-${arch}-gnu`, `compositor-linux-${arch}-musl`];
  }
  return [];
}

/** On POSIX, true when the file and its folder belong to this user, so nobody else could have put it there. */
function ownedByMe(file) {
  if (typeof process.getuid !== "function") return true;
  try {
    const uid = process.getuid();
    return statSync(file).uid === uid && statSync(path.dirname(file)).uid === uid;
  } catch {
    return false;
  }
}

/**
 * The ffmpeg that Remotion installed, if there is one: looks in node_modules from `start` upward,
 * the way Node resolves a package. Nothing is imported or run to find it; it is only a path check.
 * On POSIX the binary and its folder must belong to the current user, so a node_modules someone
 * else wrote in a shared parent folder (such as /tmp) is never used.
 */
export function findRemotionFfmpeg(start = process.cwd()) {
  const exe = process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
  let dir = path.resolve(start);
  for (;;) {
    for (const pkg of remotionPackages()) {
      const candidate = path.join(dir, "node_modules", "@remotion", pkg, exe);
      if (isExecutable(candidate) && ownedByMe(candidate)) return candidate;
    }
    const parent = path.dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

/**
 * The argument, then $AUDIO_BED_CHECK_FFMPEG, then ffmpeg on PATH, then the ffmpeg Remotion
 * installed in this project. A name or a path; must be executable.
 */
export function findFfmpeg(ffmpeg = null) {
  const candidate = ffmpeg || process.env.AUDIO_BED_CHECK_FFMPEG || "ffmpeg";
  const resolved = which(candidate);
  if (resolved !== null) return resolved;
  if (candidate !== "ffmpeg") {
    throw new DecodeError(`ffmpeg not found at ${candidate}; check --ffmpeg or AUDIO_BED_CHECK_FFMPEG`);
  }
  const remotion = findRemotionFfmpeg();
  if (remotion !== null) return remotion;
  throw new DecodeError(
    "ffmpeg not found on PATH or in a Remotion project here; install it or set AUDIO_BED_CHECK_FFMPEG",
  );
}

/**
 * [channels, count] out of a 16-bit WAV stream: the body of the `data` chunk as Float32Array
 * channels in [-1, 1).
 *
 * Walks the RIFF chunks rather than searching for the bytes `data`, because an INFO chunk can carry
 * tag text containing that word. ffmpeg writes a WAV to a pipe with placeholder sizes, so the data
 * chunk's length is not trusted; the stream ends where the samples end, and a trailing partial
 * frame is dropped.
 */
export function samplesFromWav(data, expectStereo = false) {
  const notWav = () => new DecodeError("ffmpeg returned something that is not a WAV stream");
  if (data.length < 12 || data.toString("latin1", 0, 4) !== "RIFF" || data.toString("latin1", 8, 12) !== "WAVE") {
    throw notWav();
  }
  let at = 12, count = 0;
  while (at + 8 <= data.length) {
    const chunk = data.toString("latin1", at, at + 4);
    const size = data.readUInt32LE(at + 4);
    if (chunk === "fmt ") {
      if (at + 12 > data.length) break;
      count = data.readUInt16LE(at + 10); // fmt body bytes 2-3
    } else if (chunk === "data") {
      if (count < 1) break;
      if (count > 2 && expectStereo) break; // the -ac 2 downmix must give two channels
      const body = at + 8;
      const frames = Math.floor((data.length - body) / (2 * count));
      const out = Array.from({ length: count }, () => new Float32Array(frames));
      for (let f = 0; f < frames; f++) {
        for (let c = 0; c < count; c++) out[c][f] = data.readInt16LE(body + 2 * (f * count + c)) / 32768;
      }
      return [out, count];
    }
    at += 8 + size + (size & 1);
  }
  throw notWav();
}

// what the dynamic loader prints when ffmpeg cannot start: macOS dyld, then glibc's ld.so
const LOADER = /^dyld\[|^dyld:|error while loading shared libraries/m;

function run(exe, file, downmix) {
  const args = ["-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
    "-i", "file:" + path.resolve(file), "-map", "0:a:0", "-map_metadata", "-1", "-fflags", "+bitexact"];
  if (downmix) args.push("-ac", "2");
  args.push("-ar", String(SR), "-c:a", "pcm_s16le", "-f", "wav", "-");
  // run from ffmpeg's own folder: a bundled build (Remotion's) loads its libraries from there
  const dir = path.dirname(exe);
  const env = { ...process.env };
  if (process.platform === "darwin" && existsSync(path.join(dir, "libavformat.dylib"))) {
    env.DYLD_LIBRARY_PATH = dir;
  }
  const proc = spawnSync(exe, args, {
    cwd: dir, env, stdio: ["ignore", "pipe", "pipe"], maxBuffer: Infinity, windowsHide: true,
  });
  if (proc.error) throw new DecodeError(`${file}: ffmpeg failed to start (${proc.error.message})`);
  if (proc.status !== 0) {
    const stderr = proc.stderr.toString("utf8");
    // first: the loader aborts with a signal (dyld raises SIGABRT), and its line names the library
    const loader = LOADER.exec(stderr);
    if (loader) {
      const start = stderr.lastIndexOf("\n", loader.index) + 1;
      const end = stderr.indexOf("\n", loader.index);
      const line = stderr.slice(start, end >= 0 ? end : stderr.length).trim();
      throw new DecodeError(`${file}: ffmpeg failed to start (${line})`);
    }
    if (proc.signal) {
      // a number, as Python prints it
      const number = osConstants.signals[proc.signal] ?? proc.signal;
      throw new DecodeError(`${file}: ffmpeg failed to start (signal ${number})`);
    }
    const lines = stderr.trim().split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    let detail = lines.length ? lines[lines.length - 1] : "no detail from ffmpeg";
    if (lines.some((l) => l.includes("matches no streams"))) detail = "it has no audio stream";
    throw new DecodeError(`${file}: ffmpeg could not decode it (${detail})`);
  }
  return proc.stdout;
}

/**
 * Decode the first audio track of any file ffmpeg reads to { samples, sr }: Float32Array channels at
 * 48 kHz, one for mono and two for stereo. A track with more than two channels is decoded a second
 * time with ffmpeg's `-ac 2` downmix.
 *
 * The input is always a local file: the path is passed as `file:PATH` with ffmpeg's protocol
 * whitelist set to file and pipe, so a name such as `concat:a.wav|b.wav` is opened as that file.
 * Asks for 16-bit WAV: stripped ffmpeg builds bundled with video renderers (Remotion's among
 * them) ship the WAV muxer and the pcm_s16le encoder.
 */
export function decode(file, ffmpeg = null) {
  const exe = findFfmpeg(ffmpeg);
  if (!existsSync(file)) throw new DecodeError(`${file}: no such file`);
  // a FIFO or a device would block forever; Python's version hangs on one too
  if (!statSync(file).isFile()) throw new DecodeError(`${file}: not a regular file`);
  let [samples, count] = samplesFromWav(run(exe, file, false));
  if (count > 2) [samples, count] = samplesFromWav(run(exe, file, true), true);
  if (samples[0].length === 0) throw new DecodeError(`${file}: decoded to no audio`);
  return { samples, sr: SR };
}
