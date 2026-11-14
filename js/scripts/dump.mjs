// Full-precision results for scripts/parity.py: reads a JSON manifest of cases from argv[2] and
// prints one result per case. Not part of the package.
import { readFileSync } from "node:fs";
import { decode, kWeight, levelSteps, loopScore, separation } from "../src/index.js";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = [];
for (const c of cases) {
  try {
    if (c.check === "separation") {
      const { samples: vo, sr } = decode(c.vo);
      const { samples: mix } = decode(c.mix);
      out.push(separation(vo, mix, sr, c.opts));
    } else {
      const { samples, sr } = decode(c.file);
      const weighted = kWeight(samples, sr);
      out.push(c.check === "loop"
        ? loopScore(samples, sr, { ...c.opts, weighted })
        : levelSteps(samples, sr, { ...c.opts, weighted }));
    }
  } catch (e) {
    out.push({ error: `${e.name}: ${e.message}` });
  }
}
process.stdout.write(JSON.stringify(out));
