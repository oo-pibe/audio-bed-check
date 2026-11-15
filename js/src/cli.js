// Command line: subcommands loop, steps, bed, separation. Exit 0 pass, 1 fail, 2 usage/decode error.
// The same flags and results as the Python package's cli.py, plus --ffmpeg remotion.

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
  const value = Number(t.replaceAll("_", ""));
  // Python's float("1e400") is inf too, and _number refuses it
  if (!Number.isFinite(value)) throw new UsageError(`must be a finite number, got ${text}`);
  return value;
}
number.pyName = "_number";
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
positive.pyName = "_positive";
nonNegative.pyName = "_non_negative";
const score = (text) => {
  const v = number(text);
  if (!(v > 0 && v <= 1)) throw new UsageError(`must be between 0 (exclusive) and 1, got ${text}`);
  return v;
};
score.pyName = "_score";
const choice = (options) => (text) => {
  if (!options.includes(text)) {
    throw new UsageError(`invalid choice: '${text}' (choose from ${options.map((o) => `'${o}'`).join(", ")})`);
  }
  return text;
};

const COMMON = [
  { flag: "--json", key: "json", help: "print a JSON list, one object per file" },
  { flag: "--ffmpeg", key: "ffmpeg", metavar: "PATH", type: String,
    help: "ffmpeg to use, or remotion for the one Remotion installed (default: $AUDIO_BED_CHECK_FFMPEG, then PATH)" },
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
  { flag: "--vo-offset", key: "voOffset", metavar: "S", type: number, default: null,
    help: "where the voiceover starts in the mix, seconds (default: found from the audio)" },
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

// argparse reads a token starting with "-" as a value, not an option, when it looks like a negative
// number (and this parser has no option that does)
const NEGATIVE = /^-(\d+(_\d+)*(\.(\d+(_\d+)*)?)?|\.\d+(_\d+)*)([eE][+-]?\d+(_\d+)*)?$/;

/**
 * Classify a token the way argparse's _parse_optional does: null for a positional (or a value), or
 * { option, value } where option is the matched entry (null for an unknown option) and value the text
 * after "=". A unique prefix of a long option matches it; an ambiguous one is an error.
 */
function classify(token, options, fail) {
  if (!token.startsWith("-") || token === "-") return null;
  const exact = options.find((o) => o.flag === token);
  if (exact) return { option: exact, value: null };
  const eq = token.indexOf("=");
  if (eq >= 0) {
    const named = options.find((o) => o.flag === token.slice(0, eq));
    if (named) return { option: named, value: token.slice(eq + 1) };
  }
  if (token.startsWith("--")) {
    const name = eq >= 0 ? token.slice(0, eq) : token;
    const matches = options.filter((o) => o.flag.startsWith(name));
    if (matches.length > 1) {
      fail(`ambiguous option: ${name} could match ${matches.map((o) => o.flag).join(", ")}`);
    }
    if (matches.length === 1) return { option: matches[0], value: eq >= 0 ? token.slice(eq + 1) : null };
  }
  if (NEGATIVE.test(token)) return null;
  if (token.includes(" ")) return null;
  return { option: null, value: null };
}

const HELP = { flag: "--help", key: "help" };
/** -h, --help, or an abbreviation of --help: what argparse reads as a help request. */
const isHelp = (token) => token === "-h" || (token.length >= 3 && "--help".startsWith(token));
const SHORT_HELP = { flag: "-h", key: "help" };
const VERSION_FLAG = { flag: "--version", key: "version" };

/**
 * The command line, parsed by argparse's rules for this parser: every token is classified first
 * (an ambiguous abbreviation fails there), then read in order. Options take --flag value or
 * --flag=value; -h prints help wherever it is reached; the files are one contiguous run; after "--"
 * everything is a positional. A missing required argument is reported before anything left over,
 * and leftovers (unknown options, stray positionals) are reported by the main parser.
 */
export function parse(argv) {
  const mainFail = (message) => { throw new UsageError(message, MAIN_USAGE); };
  // help only on its own (`-h`, `bed -h`): with a file list from a glob, a file named --help would
  // otherwise turn the run into exit 0
  const end = argv.indexOf("--");
  const helpAt = (end < 0 ? argv : argv.slice(0, end)).findIndex(isHelp);
  if (helpAt >= 0 && !(argv.length === 1 || (argv.length === 2 && helpAt === 1 && argv[0] in COMMANDS))) {
    mainFail("-h/--help cannot be combined with other arguments");
  }
  const extras = [];
  let at = 0;
  for (; at < argv.length; at++) {
    const token = argv[at];
    if (token === "--") { at++; break; }
    const c = classify(token, [SHORT_HELP, HELP, VERSION_FLAG], mainFail);
    if (c === null) break;
    if (c.option?.key === "help") return { help: help(null) };
    if (c.option?.key === "version") return { version: `${PROG} ${VERSION}` };
    extras.push(token);
  }
  if (at >= argv.length) mainFail("the following arguments are required: command");
  const first = argv[at];
  if (!(first in COMMANDS)) {
    mainFail(`argument command: invalid choice: '${first}' (choose from 'loop', 'steps', 'bed', 'separation')`);
  }
  const rest = argv.slice(at + 1);
  const cmd = COMMANDS[first];
  const fail = (message) => { throw new UsageError(message, usage(first), `${PROG} ${first}`); };
  const options = [SHORT_HELP, HELP, ...cmd.options];
  const args = { command: first, json: false, ffmpeg: null };
  for (const o of cmd.options) if ("default" in o) args[o.key] = o.default;

  // pass 1: classify, as argparse does before it consumes anything
  let afterDashes = false;
  const kinds = rest.map((token) => {
    if (afterDashes) return null;
    if (token === "--") { afterDashes = true; return "--"; }
    return classify(token, options, fail);
  });

  // pass 2: consume in order
  const positionals = [];
  let positionalsTaken = false;
  for (let i = 0; i < rest.length; i++) {
    const kind = kinds[i];
    if (kind === "--") continue;
    if (kind === null) {
      // a run of positionals: the first run fills BED (all of it) or MIX (one); the rest is left over
      const run = [];
      while (i < rest.length && (kinds[i] === null || kinds[i] === "--")) {
        if (kinds[i] === null) run.push(rest[i]);
        i++;
      }
      i--;
      if (positionalsTaken) extras.push(...run);
      else if (cmd.positional === "BED") { positionals.push(...run); positionalsTaken = run.length > 0; }
      else { positionals.push(run[0]); extras.push(...run.slice(1)); positionalsTaken = true; }
      continue;
    }
    const { option: o, value: attached } = kind;
    if (o === null) { extras.push(rest[i]); continue; }
    if (o.key === "help") return { help: help(first) };
    if (!o.type) {
      if (attached !== null) fail(`argument ${o.flag}: ignored explicit argument '${attached}'`);
      args[o.key] = true;
      continue;
    }
    let value = attached;
    if (value === null) {
      if (i + 1 >= rest.length || kinds[i + 1] !== null) fail(`argument ${o.flag}: expected one argument`);
      value = rest[++i];
    }
    try {
      args[o.key] = o.type === String ? value : o.type(value);
    } catch (e) {
      if (!(e instanceof UsageError)) throw e;
      fail(`argument ${o.flag}: ${e.detail ?? `invalid ${o.type.pyName} value: '${value}'`}`);
    }
  }
  const missing = cmd.options.filter((o) => o.required && args[o.key] == null).map((o) => o.flag);
  if (positionals.length === 0) missing.unshift(cmd.positional);
  if (missing.length) fail(`the following arguments are required: ${missing.join(", ")}`);
  if (extras.length) mainFail(`unrecognized arguments: ${extras.join(" ")}`);
  if (cmd.positional === "MIX") args.mix = positionals[0];
  else args.files = positionals;
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
    out[k] = typeof v === "number" && !INT_KEYS.has(k) ? round2(v) : v;
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
      const short = r.notes.find((n) => n.startsWith("too short"));
      if (short !== undefined) {
        // nothing was measured: could not check, never a pass
        results.push({ file, passed: false, error: `${file}: ${short}` });
        continue;
      }
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

// the only integer fields in the output; every other number is a float in the Python package
const INT_KEYS = new Set(["runs", "gaps"]);

/** Python's repr of a float: shortest digits, ".0" on whole numbers, exponent form from 1e16. */
function pyFloat(v) {
  if (Number.isInteger(v)) {
    if (Math.abs(v) < 1e16) return `${v}.0`;
    return v.toExponential().replace(/e([+-])(\d)$/, "e$10$2");
  }
  const text = String(v);
  return text.includes("e") ? text.replace(/e([+-])(\d)$/, "e$10$2") : text;
}

/** Python's json.dumps(value, indent=2), including ensure_ascii and float repr. */
function pyJson(value, key = null, pad = "") {
  const inner = pad + "  ";
  if (value === null) return "null";
  if (typeof value === "boolean") return String(value);
  if (typeof value === "number") return INT_KEYS.has(key) ? String(value) : pyFloat(value);
  if (typeof value === "string") {
    return JSON.stringify(value).replace(/[\u007f-\uffff]/g, (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    return "[\n" + value.map((v) => inner + pyJson(v, key, inner)).join(",\n") + "\n" + pad + "]";
  }
  const entries = Object.entries(value);
  if (entries.length === 0) return "{}";
  return "{\n" + entries.map(([k, v]) => `${inner}${pyJson(k)}: ${pyJson(v, k, inner)}`).join(",\n") + "\n" + pad + "}";
}

/**
 * Text output of a string that came from outside (a file name, an error that quotes one): control
 * characters as \xNN, so a name cannot move the cursor, recolor the terminal or start a new line,
 * and "::" broken up, so a CI log never reads it as a workflow command. JSON escapes on its own.
 */
export function safe(text) {
  return String(text)
    .replace(/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/g, (c) => {
      const code = c.charCodeAt(0);
      return code <= 0xff ? `\\x${code.toString(16).padStart(2, "0")}` : `\\u${code.toString(16)}`;
    })
    .replaceAll("::", ":\\x3a");
}

const verdict = (passed) => (passed ? "ok  " : "FAIL");

export function render(entry) {
  const lines = [safe(entry.file)];
  if ("error" in entry) lines.push(`  error ${safe(entry.error)}`);
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
    if (r.offset_found) lines.push(`  info  voice       starts at ${num(r.vo_offset_s, 2)}s in the mix (found)`);
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
    err(`${e.usage}\n${e.prog}: error: ${safe(e.detail)}`);
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
    err(`${PROG}: ${safe(e.message)}`);
    return 2;
  }
  out(args.json ? pyJson(results) : results.map(render).join("\n\n"));
  if (results.some((r) => "error" in r)) return 2;
  return results.every((r) => r.passed) ? 0 : 1;
}
