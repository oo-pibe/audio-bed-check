// Python's rounding and number formatting, so every printed number matches the Python package.
// Python rounds exact ties to even (round(0.125, 2) is 0.12, f"{0.125:.2f}" is "0.12");
// JavaScript's toFixed and Math.round round them up.

/** True when |x| * 10**d sits exactly halfway between two integers (x is a binary double, so exact). */
function isTie(x, d) {
  // toFixed is exact up to 100 digits; a tie at d decimals is a 5 at d+1 followed by zeros only
  const s = Math.abs(x).toFixed(Math.min(100, d + 60));
  const dot = s.indexOf(".");
  const tail = s.slice(dot + 1 + d);
  return /^50*$/.test(tail);
}

/** Python's round(x, d) for d >= 0: nearest, ties to even. */
export function pyRound(x, d = 0) {
  if (!Number.isFinite(x)) return x;
  const text = pyFixed(x, d);
  const value = Number(text);
  return Object.is(value, -0) ? -0 : value;
}

/** Python's round(x) to an integer: nearest, ties to even. */
export function pyRoundInt(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Python's f"{x:.{d}f}": correctly rounded, exact ties to even. */
export function pyFixed(x, d) {
  if (!isTie(x, d)) return x.toFixed(d);
  // toFixed rounded the tie away from zero; step back when that made the last digit odd
  const up = Math.abs(x).toFixed(d);
  const digits = up.replace(".", "");
  const last = Number(digits[digits.length - 1]);
  let text = up;
  if (last % 2 === 1) {
    const scaled = BigInt(digits) - 1n;
    let s = scaled.toString().padStart(d + 1, "0");
    text = d > 0 ? `${s.slice(0, s.length - d)}.${s.slice(s.length - d)}` : s;
  }
  return x < 0 || Object.is(x, -0) ? `-${text}` : text;
}

/** Python's f"{x:g}": 6 significant digits, trailing zeros dropped, exponent form outside 1e-4..1e6. */
export function pyG(x) {
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  const exp = Number(x.toExponential(5).split("e")[1]);
  if (exp < -4 || exp >= 6) {
    const [mant, e] = x.toExponential(5).split("e");
    const m = mant.includes(".") ? mant.replace(/0+$/, "").replace(/\.$/, "") : mant;
    const n = Number(e);
    return `${m}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  const fixed = x.toFixed(Math.max(0, 5 - exp));
  return fixed.includes(".") ? fixed.replace(/0+$/, "").replace(/\.$/, "") : fixed;
}

/** numpy's percentile with the default linear method, including its lerp form. */
export function percentile(values, q) {
  const v = Float64Array.from(values).sort();
  const pos = (q / 100) * (v.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, v.length - 1);
  const t = pos - lo;
  const a = v[lo], b = v[hi];
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

export function mean(values) {
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}
