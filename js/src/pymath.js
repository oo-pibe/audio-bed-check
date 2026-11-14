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
  if (Number.isFinite(x) && Math.abs(x) >= 1e21) {
    // toFixed switches to exponent form here; every such double is an integer
    return BigInt(x).toString() + (d > 0 ? "." + "0".repeat(d) : "");
  }
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

/**
 * Python's f"{x:g}": 6 significant digits rounded half to even, trailing zeros dropped, exponent
 * form when the exponent is under -4 or 6 and over.
 */
export function pyG(x) {
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  const P = 6;
  const sign = x < 0 ? "-" : "";
  const a = Math.abs(x);
  let digits, exp;
  if (a < 1e21 && a >= 1e-30) {
    // toFixed(100) is exact here as far as a tie can matter: a tie needs the expansion to end right
    // after the 7th significant digit, and for |x| >= 1e-30 that is well inside 100 decimals
    const [whole, frac] = a.toFixed(100).split(".");
    const all = whole + frac;
    const lead = all.search(/[1-9]/);
    exp = whole.length - 1 - lead;
    let n = BigInt(all.slice(lead, lead + P));
    const tail = all.slice(lead + P);
    const roundUp = tail[0] > "5" || (tail[0] === "5" && (/[1-9]/.test(tail.slice(1)) || n % 2n === 1n));
    if (roundUp) n += 1n;
    digits = n.toString();
    if (digits.length > P) { digits = digits.slice(0, P); exp += 1; }
  } else {
    const [m, e] = a.toExponential(P - 1).split("e");
    digits = m.replace(".", "");
    exp = Number(e);
  }
  const strip = (t) => (t.includes(".") ? t.replace(/0+$/, "").replace(/\.$/, "") : t);
  if (exp < -4 || exp >= P) {
    const mant = strip(digits[0] + "." + digits.slice(1));
    return `${sign}${mant}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  let text;
  if (exp >= 0) text = digits.slice(0, exp + 1) + "." + digits.slice(exp + 1);
  else text = "0." + "0".repeat(-exp - 1) + digits;
  return sign + strip(text);
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

/** numpy's pairwise sum of float64 (8 accumulators up to 128 values, halves above), so means match. */
export function pairwiseSum(a, start = 0, n = a.length) {
  if (n < 8) {
    let s = -0.0;
    for (let i = 0; i < n; i++) s += a[start + i];
    return s;
  }
  if (n <= 128) {
    const r = new Float64Array(8);
    for (let k = 0; k < 8; k++) r[k] = a[start + k];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let k = 0; k < 8; k++) r[k] += a[start + i + k];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[start + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2);
}

export function mean(values) {
  return pairwiseSum(values) / values.length;
}
