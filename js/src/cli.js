// Command line: subcommands loop, steps, bed, separation. Exit 0 pass, 1 fail, 2 usage/decode error.
// The same flags, output and exit codes as the Python package's cli.py.

import { readFileSync } from "node:fs";
import { SILENT_NOTE, levelSteps, loopScore, separation } from "./checks.js";
import { decode } from "./decode.js";
import { DecodeError, ValueError } from "./errors.js";
import { kWeight } from "./loudness.js";
import { pyFixed, pyRound } from "./pymath.js";

export const VERSION = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
const PROG = "audio-bed-check";
const PROFILES = { music: 10.0, ambience: 15.0, wcag: 20.0 };
const OVERRIDE = "--min-separation"; // what the profile column says when the flag replaced the profile

class UsageError extends Error {
  /** message null means "invalid value"; usage is the line printed above the error. */
  constructor(message, usage = null, prog = PROG) {
    super(message ?? "");
    this.detail = message;
    this.usage = usage;
    this.prog = prog;
  }
}

// Python's float(): sign, digits with single underscores between them, a point, an exponent;
// inf and nan parse but are not finite
const DECIMAL = /^[+-]?(\d(_?\d)*(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$/;
const NONFINITE = /^[+-]?(inf|infinity|nan)$/i;

function number(text) {
  const t = text.trim();
  if (NONFINITE.test(t)) throw new UsageError(`must be a finite number, got ${text}`);
  if (!DECIMAL.test(t)) throw new UsageError(null);
  return Number(t.replaceAll("_", ""));
}
const positive = (text) => {
  const v = number(text);
  if (v <= 0) throw new UsageError(`must be a positive number, got ${text}`);
  return v;
};
const nonNegative = (text) => {
  const v = number(text);
  if (v < 0) throw new UsageError(`must be zero or more, got ${text}`);
  return v;
};
const score = (text) => {
  const v = number(text);
  if (!(v > 0 && v <= 1)) throw new UsageError(`must be between 0 (exclusive) and 1, got ${text}`);
  return v;
};
const choice = (options) => (text) => {
  if (!options.includes(text)) {
    throw new UsageError(`invalid choice: '${text}' (choose from ${options.map((o) => `'${o}'`).join(", ")})`);
  }
  return text;
};

const COMMON = [
  { flag: "--json", key: "json", help: "print a JSON list, one object per file" },
  { flag: "--ffmpeg", key: "ffmpeg", metavar: "PATH", type: String,
    help: "ffmpeg to use (default: $AUDIO_BED_CHECK_FFMPEG, then PATH, then Remotion's)" },
];
const LOOP = [
  { flag: "--min-period", key: "minPeriod", metavar: "S", type: positive, default: 6.0,
    help: "shortest repeat length that counts as a loop, seconds (default 6; shorter repeats are noted)" },
  { flag: "--loop-threshold", key: "loopThreshold", metavar: "SCORE", type: score, default: 0.9,
    help: "correlation at which a repeat fails, 0 to 1 (default 0.9)" },
];
const STEPS = [
  { flag: "--max-step", key: "maxStep", metavar: "DB", type: positive, default: 6.0,
    help: "largest sustained level step allowed, dB (default 6)" },
  { flag: "--edge", key: "edge", metavar: "S", type: nonNegative, default: 0.75,
    help: "seconds ignored at each end so fades are not read as steps (default 0.75)" },
];
const SEPARATION = [
  { flag: "--vo", key: "vo", metavar: "VO", type: String, required: true,
    help: "the voiceover on its own, as it was placed in the mix" },
  { flag: "--profile", key: "profile", metavar: "{music,ambience,wcag}", type: choice(Object.keys(PROFILES)),
    default: "music", help: "minimum separation: music 10 LU, ambience 15 LU, wcag 20 LU (default music)" },
  { flag: "--min-separation", key: "minSeparation", metavar: "LU", type: positive, default: null,
    help: "override the profile's minimum" },
  { flag: "--gate", key: "gate", metavar: "DBFS", type: number, default: -44.0,
    help: "level above which the voiceover counts as speech (default -44)" },
  { flag: "--vo-offset", key: "voOffset", metavar: "S", type: number, default: 0.0,
    help: "where the voiceover starts in the mix, seconds (default 0)" },
];

const COMMANDS = {
  loop: { text: "does the bed repeat itself?", options: [...LOOP, ...COMMON], positional: "BED" },
  steps: { text: "does the level lurch at a join?", options: [...STEPS, ...COMMON], positional: "BED" },
  bed: { text: "loop and steps together", options: [...LOOP, ...STEPS, ...COMMON], positional: "BED" },
  separation: {
    text: "is the voice far enough above the bed?",
    description: "Measures the rendered mix inside the voiceover's speech windows against the gaps between them.",
    options: [...SEPARATION, ...COMMON], positional: "MIX",
  },
};

const MAIN_USAGE = `usage: ${PROG} [-h] [--version] {loop,steps,bed,separation} ...`;

function usage(name) {
  const cmd = COMMANDS[name];
  const flags = cmd.options.map((o) => {
    const part = o.type ? `${o.flag} ${o.metavar}` : o.flag;
    return o.required ? part : `[${part}]`;
  });
  const pos = cmd.positional === "BED" ? "BED [BED ...]" : "MIX";
  return `usage: ${PROG} ${name} [-h] ${flags.join(" ")} ${pos}`;
}

function help(name) {
  if (!name) {
    return [
      MAIN_USAGE, "",
      "Check an audio bed for loops and level steps, or a mix for voice-over-bed separation. " +
        "Exit 0 when every check passes, 1 when any fails, 2 when a check could not run.",
      "", "positional arguments:", "  {loop,steps,bed,separation}",
      ...Object.entries(COMMANDS).map(([n, c]) => `    ${n.padEnd(12)}${c.text}`),
      "", "options:", "  -h, --help     show this help message and exit",
      "  --version      show program's version number and exit",
    ].join("\n");
  }
  const cmd = COMMANDS[name];
  const posHelp = cmd.positional === "BED"
    ? "  BED            audio or video file; anything ffmpeg reads"
    : "  MIX            the rendered mix (audio or video)";
  return [
    usage(name), "", cmd.description || cmd.text, "", "positional arguments:", posHelp, "",
    "options:", "  -h, --help     show this help message and exit",
    ...cmd.options.map((o) => `  ${o.type ? `${o.flag} ${o.metavar}` : o.flag}\n                 ${o.help}`),
  ].join("\n");
}

/** argparse's rules: --flag value, --flag=value, unique prefixes, -- ends the options. */
export function parse(argv) {
  if (argv.length === 0) throw new UsageError("the following arguments are required: command", MAIN_USAGE);
  const [first, ...rest] = argv;
  if (first === "-h" || first === "--help") return { help: help(null) };
  if (first === "--version") return { version: `${PROG} ${VERSION}` };
  if (!(first in COMMANDS)) {
    const message = first.startsWith("-")
      ? `unrecognized arguments: ${first}`
      : `argument command: invalid choice: '${first}' (choose from 'loop', 'steps', 'bed', 'separation')`;
    throw new UsageError(message, MAIN_USAGE);
  }
  const cmd = COMMANDS[first];
  const fail = (message) => { throw new UsageError(message, usage(first), `${PROG} ${first}`); };
  const args = { command: first, json: false, ffmpeg: null };
  for (const o of cmd.options) if ("default" in o) args[o.key] = o.default;
  const positionals = [];
  let onlyPositional = false;
  for (let i = 0; i < rest.length; i++) {
    const token = rest[i];
    if (onlyPositional || !token.startsWith("-") || token === "-" || /^-\d|^-\.\d/.test(token)) {
      positionals.push(token);
      continue;
    }
    if (token === "--") { onlyPositional = true; continue; }
    if (token === "-h" || token === "--help") return { help: help(first) };
    const eq = token.indexOf("=");
    const name = eq >= 0 ? token.slice(0, eq) : token;
    const exact = cmd.options.find((o) => o.flag === name);
    const matches = exact ? [exact] : cmd.options.filter((o) => o.flag.startsWith(name) && name.startsWith("--"));
    if (matches.length === 0) fail(`unrecognized arguments: ${token}`);
    if (matches.length > 1) {
      fail(`ambiguous option: ${name} could match ${matches.map((o) => o.flag).join(", ")}`);
    }
    const o = matches[0];
    if (!o.type) {
      if (eq >= 0) fail(`argument ${o.flag}: ignored explicit argument '${token.slice(eq + 1)}'`);
      args[o.key] = true;
      continue;
    }
    let value;
    if (eq >= 0) value = token.slice(eq + 1);
    else {
      if (i + 1 >= rest.length || (rest[i + 1].startsWith("-") && !/^-\d|^-\.\d/.test(rest[i + 1]))) {
        fail(`argument ${o.flag}: expected one argument`);
      }
      value = rest[++i];
    }
    try {
      args[o.key] = o.type === String ? value : o.type(value);
    } catch (e) {
      if (!(e instanceof UsageError)) throw e;
      fail(`argument ${o.flag}: ${e.detail ?? `invalid value: '${value}'`}`);
    }
  }
  const missing = cmd.options.filter((o) => o.required && args[o.key] == null).map((o) => o.flag);
  if (cmd.positional === "MIX") {
    if (positionals.length === 0) missing.unshift("MIX");
    if (missing.length) fail(`the following arguments are required: ${missing.join(", ")}`);
    if (positionals.length > 1) fail(`unrecognized arguments: ${positionals.slice(1).join(" ")}`);
    args.mix = positionals[0];
  } else {
    if (positionals.length === 0) fail("the following arguments are required: BED");
    args.files = positionals;
  }
  return args;
}

/** 2 decimals, and anything within 0.005 of zero is 0.0, so nothing prints as -0.0 or -0.00. */
function round2(value) {
  return Math.abs(value) < 0.005 ? 0.0 : pyRound(value, 2);
}

/** A result with numbers rounded to 2 decimals; the library keeps full precision. */
function rounded(result) {
  const out = {};
  for (const [k, v] of Object.entries(result)) {
    out[k] = typeof v === "number" && !Number.isInteger(v) ? round2(v) : v;
  }
  return out;
}

function runChecks(args) {
  const results = [];
  if (args.command === "separation") {
    const { samples: vo, sr } = decode(args.vo, args.ffmpeg);
    const { samples: mix } = decode(args.mix, args.ffmpeg);
    const overridden = args.minSeparation != null;
    const minLu = overridden ? args.minSeparation : PROFILES[args.profile];
    const r = separation(vo, mix, sr, { minLu, gateDbfs: args.gate, voOffset: args.voOffset });
    results.push({ file: args.mix, passed: r.passed, profile: overridden ? OVERRIDE : args.profile, separation: rounded(r) });
    return results;
  }
  for (const file of args.files) {
    let decoded;
    try {
      decoded = decode(file, args.ffmpeg);
    } catch (e) {
      // one unreadable file must not throw away the rest of the batch
      if (!(e instanceof DecodeError)) throw e;
      results.push({ file, passed: false, error: e.message });
      continue;
    }
    const { samples, sr } = decoded;
    const entry = { file, passed: true };
    const weighted = kWeight(samples, sr); // once per file, shared by both checks
    if (args.command === "loop" || args.command === "bed") {
      const r = loopScore(samples, sr, { minPeriod: args.minPeriod, threshold: args.loopThreshold, weighted });
      entry.loop = rounded(r);
      entry.passed = entry.passed && r.passed;
    }
    if (args.command === "steps" || args.command === "bed") {
      const r = levelSteps(samples, sr, { maxStep: args.maxStep, edge: args.edge, weighted });
      entry.steps = rounded(r);
      entry.passed = entry.passed && r.passed;
    }
    results.push(entry);
  }
  return results;
}

/** A printed number: fixed decimals, and never "-0.0" or "-0.00" for something that rounds to zero. */
function num(value, decimals) {
  const text = pyFixed(value, decimals);
  return Number(text) === 0 ? (0).toFixed(decimals) : text;
}

const verdict = (passed) => (passed ? "ok  " : "FAIL");

export function render(entry) {
  const lines = [entry.file];
  if ("error" in entry) lines.push(`  error ${entry.error}`);
  if (entry.loop) {
    const r = entry.loop;
    let detail;
    if (r.notes.length) detail = r.notes[0];
    else if (r.period_s === null) detail = `no repeat at or above ${num(r.threshold, 2)}`;
    else detail = `repeats every ${num(r.period_s, 1)}s, at or above ${num(r.threshold, 2)}`;
    lines.push(`  ${verdict(r.passed)}  loop        ${num(r.score, 2)} (${detail})`);
  }
  if (entry.steps) {
    const r = entry.steps;
    const notes = r.notes.filter((n) => n !== SILENT_NOTE);
    if (r.step_at_s === null) {
      lines.push(`  ${verdict(r.passed)}  step        ${notes.length ? notes[0] : "no level step measured"}`);
    } else {
      lines.push(`  ${verdict(r.passed)}  step        ${num(r.step_db, 2)} dB at ${num(r.step_at_s, 2)}s (max ${num(r.max_step, 2)})`);
    }
    lines.push(`  info  range       ${num(r.range_db, 1)} dB   transient ${num(r.transient_db, 1)} dB   peak ${num(r.peak_dbtp, 1)} dBTP`);
    if (r.notes.includes(SILENT_NOTE)) lines.push(`  warn  ${SILENT_NOTE}`);
  }
  if (entry.separation) {
    const r = entry.separation;
    lines.push(`  ${verdict(r.passed)}  separation  ${num(r.separation_lu, 2)} LU (speech ${num(r.speech_lkfs, 2)}, ` +
      `bed ${num(r.bed_lkfs, 2)} LKFS; min ${num(r.min_lu, 2)}, ${entry.profile})`);
    lines.push(`  info  peak        ${num(r.peak_dbtp, 1)} dBTP   speech windows ${r.runs}   bed windows ${r.gaps}`);
    for (const w of r.warnings) lines.push(`  warn  ${w}`);
  }
  return lines.join("\n");
}

/** Run the CLI; returns the exit code. `out` and `err` take one string each. */
export function main(argv = process.argv.slice(2), out = (s) => process.stdout.write(s + "\n"),
  err = (s) => process.stderr.write(s + "\n")) {
  let args;
  try {
    args = parse(argv);
  } catch (e) {
    if (!(e instanceof UsageError)) throw e;
    err(`${e.usage}\n${e.prog}: error: ${e.detail}`);
    return 2;
  }
  if (args.help) { out(args.help); return 0; }
  if (args.version) { out(args.version); return 0; }
  let results;
  try {
    results = runChecks(args);
  } catch (e) {
    // NoSpeechError and the library's argument guards are ValueErrors
    if (!(e instanceof DecodeError || e instanceof ValueError)) throw e;
    err(`${PROG}: ${e.message}`);
    return 2;
  }
  out(args.json ? JSON.stringify(results, null, 2) : results.map(render).join("\n\n"));
  if (results.some((r) => "error" in r)) return 2;
  return results.every((r) => r.passed) ? 0 : 1;
}
