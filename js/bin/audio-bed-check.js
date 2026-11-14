#!/usr/bin/env node
// Exit 0 pass, 1 a check failed, 2 could not run. Anything that goes wrong here is "could not run":
// a crash must never exit 1 and read as a verdict.
const fail = (message) => {
  try { process.stderr.write(`audio-bed-check: ${message}\n`); } catch {}
  process.exit(2);
};
// output closed early (piped into head): stop quietly, exit 2
process.stdout.on("error", (e) => (e.code === "EPIPE" ? process.exit(2) : fail(`internal error: ${e.message}`)));
process.on("uncaughtException", (e) => fail(`internal error: ${e?.stack ?? e}`));

const major = Number(process.versions.node.split(".")[0]);
if (major < 20) fail(`needs Node 20 or later; this is ${process.version}`);
import("../src/cli.js").then(({ main }) => {
  process.exitCode = main();
}, (e) => fail(`internal error: ${e?.stack ?? e}`));
