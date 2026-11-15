#!/usr/bin/env node
// Exit 0 pass, 1 a check failed, 2 could not run. Anything that goes wrong here is "could not run":
// a crash must never exit 1 and read as a verdict.
const fail = (message) => {
  // control characters escaped, as cli.js does for everything it prints
  const text = String(message).replace(/[\u0000-\u0009\u000b-\u001f\u007f-\u009f]/g,
    (c) => `\\x${c.charCodeAt(0).toString(16).padStart(2, "0")}`).replaceAll("::", ":\\x3a");
  try { process.stderr.write(`audio-bed-check: ${text}\n`); } catch {}
  process.exit(2);
};
// output closed early (piped into head): stop quietly, exit 2
process.stdout.on("error", (e) => (e.code === "EPIPE" ? process.exit(2) : fail(`internal error: ${e.message}`)));
process.on("uncaughtException", (e) => fail(`internal error: ${e?.stack ?? e}`));

// 20.15.1 and 22.4.1 carry Node's complete fix for running Windows batch files with arguments
// (CVE-2024-27980, CVE-2024-36138); this tool runs only .exe files, and asks for a fixed Node anyway
const [major, minor, patch] = process.versions.node.split(".").map(Number);
const atLeast = (m, p) => minor > m || (minor === m && patch >= p);
const supported = major >= 23 || (major === 22 && atLeast(4, 1)) || (major === 20 && atLeast(15, 1));
if (!supported) fail(`needs Node 20.15.1 or later (22.4.1 on Node 22); this is ${process.version}`);
else {
  import("../src/cli.js").then(({ main }) => {
    process.exitCode = main();
  }, (e) => fail(`internal error: ${e?.stack ?? e}`));
}
