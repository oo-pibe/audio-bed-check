#!/usr/bin/env node
import { main } from "../src/cli.js";

try {
  process.exitCode = main();
} catch (e) {
  // a bug, not a verdict: exit 2 ("could not run"), never 1 ("a check failed")
  process.stderr.write(`audio-bed-check: internal error: ${e?.stack ?? e}\n`);
  process.exitCode = 2;
}
