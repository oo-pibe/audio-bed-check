"""Check that the JavaScript port gives the Python package's results. Not part of either test suite.

    .venv/bin/python js/scripts/parity.py            # the standard corpus
    .venv/bin/python js/scripts/parity.py --sweep    # plus the full 400-bed loop sweep (minutes)

Builds every fixture with the generators in tests/synth.py and scripts/loop_sweep.py, writes them
as 16-bit WAVs, then runs each check two ways: the Python library on decode(), and the JS library
through js/scripts/dump.mjs. Every field must match: booleans, notes, periods and positions exactly,
levels within 1e-6 dB. Then the CLI text output of both is compared byte for byte, flags included.
Run from the repository root.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from loop_sweep import UNLOOPED, looped  # noqa: E402

from audio_bed_check import decode, k_weight, level_steps, loop_score, separation  # noqa: E402
from tests.synth import (  # noqa: E402
    SR,
    fade,
    fade_edges,
    gain,
    gated,
    irregular_read,
    noise,
    sine,
    step_gain,
    swell_gain,
    textured,
    tile,
    write_wav,
)

TOL = 1e-6
JS = ROOT / "js"


def beds(sweep: bool):
    """(name, signal) for the loop and steps checks."""
    count = {True: None, False: 2}[sweep]
    for seconds, n in UNLOOPED:
        for j in range(n if count is None else min(n, count)):
            yield f"unlooped {seconds}s #{j}", textured(seconds, j)
    for name, x, _ in looped(5000):
        yield f"looped: {name}", x
    bed = textured(120, 6000)
    every_8 = np.prod([swell_gain(len(bed), c, 3, 6) for c in np.arange(5, 120, 8)], axis=0)
    yield "periodic swells", (bed * every_8).astype(np.float32)
    for db in (-12, -7, -3, 3, 6.0, 6.01, 9):
        x = textured(40, 7)
        yield f"step {db:+g} dB at 20 s", (x * step_gain(len(x), 20.0, db) * 0.5).astype(np.float32)
    x = textured(40, 8)
    yield "swell +8 dB", (x * swell_gain(len(x), 20, 1.5, 8) * 0.5).astype(np.float32)
    yield "music-length repeat (bar)", tile(textured(2, 9), 20)
    yield "silence", np.zeros(20 * SR, dtype=np.float32)
    yield "near silence", noise(20, 10, rms_dbfs=-85)
    yield "constant tone", sine(20, 440, 0.25)
    yield "short 3 s", textured(3, 11)
    yield "short 7 s", textured(7, 12)
    yield "one 6.2 s clip twice (12.4 s)", tile(textured(6.2, 18), 2)
    yield "just under twice --min-period (11.9 s)", textured(11.9, 19)
    yield "fade edges", fade_edges(textured(30, 13), 2.0)
    left, right = textured(30, 14), textured(30, 15)
    yield "stereo independent", np.stack([left, right], axis=1)
    yield "stereo anti-phase", np.stack([left, -left], axis=1)
    loopy = tile(textured(10, 16), 5)
    yield "stereo loop", np.stack([loopy, gain(loopy, -3)], axis=1)
    five = np.stack([textured(30, 20 + c) * 0.4 for c in range(6)], axis=1)
    yield "5.1 downmixed by ffmpeg", five.astype(np.float32)
    yield "hot peak", np.clip(textured(30, 17) * 3, -1, 1).astype(np.float32)


def mixes():
    """(name, voiceover, mix, opts) for the separation check."""
    for k, db in enumerate((-20, -10, -3, 0, 6)):
        runs = irregular_read(40, k)
        vo = fade(gated(noise(40, 100 + k, rms_dbfs=-18), runs))
        bed = gain(textured(40, 200 + k), db - 18)
        yield f"voice over bed, bed at {db:+d} dB", vo, (vo + bed).astype(np.float32), {}
    runs = irregular_read(30, 9)
    vo = fade(gated(noise(30, 109, rms_dbfs=-18), runs))
    bed = gain(textured(36, 209), -30)
    shifted = bed.copy()
    shifted[int(3.0 * SR):int(3.0 * SR) + len(vo)] += vo
    yield "offset 3 s, given", vo, shifted, {"voOffset": 3.0}
    yield "offset 3 s, not given", vo, shifted, {}
    yield "offset 3 s, wcag", vo, shifted, {"voOffset": 3.0, "minLu": 20.0}
    stereo_vo = np.stack([vo, vo * 0.5], axis=1)
    stereo_mix = np.stack([shifted[:len(vo)], bed[:len(vo)]], axis=1)
    yield "stereo", stereo_vo, stereo_mix.astype(np.float32), {}
    yield "gate -30", vo, (vo + gain(textured(30, 210), -40)).astype(np.float32), {"gateDbfs": -30.0}
    yield "no speech", np.zeros(10 * SR, dtype=np.float32), textured(10, 211), {}
    yield "no pauses", noise(10, 112, rms_dbfs=-18), textured(10, 212), {}
    yield "silent bed", vo, vo, {}
    yield "outside the mix", vo, (vo + bed[:len(vo)]).astype(np.float32), {"voOffset": 100.0}
    yield "offset 3 s, wrong one given", vo, shifted, {"voOffset": 0.0}
    buried = bed.copy()
    buried[int(3.0 * SR):int(3.0 * SR) + len(vo)] += gain(vo, -22)
    yield "buried 3 s in, not found", vo, buried.astype(np.float32), {"gateDbfs": -70.0}
    from tests.synth import speechlike_bed_mix
    for seed, start in ((1, 2.0), (2, 5.3), (3, 0.0)):   # untrusted start: could not tell, or agreed
        yield f"speech-like bed, {start} s in", *speechlike_bed_mix(seed, start), {}


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0, abs_tol=TOL)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def tied_positions(file, edge=0.75):
    """step_at_s of every boundary whose sustained step is within 1e-6 dB of Python's largest,
    computed from Python's own envelope with level_steps's arithmetic."""
    from audio_bed_check.checks import STEP_BLOCK, STEP_FLOOR_LKFS, STEP_GAP, STEP_HOP, STEP_SIDE
    from audio_bed_check.loudness import block_loudness
    x, sr = decode(file)
    env = block_loudness(k_weight(x, sr), sr, STEP_BLOCK, STEP_HOP)
    skip = math.ceil(edge / STEP_HOP - 1e-9)
    env = env[skip:len(env) - skip] if skip else env
    c = np.concatenate([[0.0], np.cumsum(np.maximum(env, STEP_FLOOR_LKFS))])
    i = np.arange(STEP_SIDE, len(env) - STEP_GAP - STEP_SIDE + 1)
    after = (c[i + STEP_GAP + STEP_SIDE] - c[i + STEP_GAP]) / STEP_SIDE
    steps = np.abs(after - (c[i] - c[i - STEP_SIDE]) / STEP_SIDE)
    tied = i[steps >= steps.max() - 1e-6]
    return {round(float(((skip + k - 1) * STEP_HOP + STEP_BLOCK + (skip + k + STEP_GAP) * STEP_HOP) / 2), 2)
            for k in tied}


def py_result(case):
    try:
        if case["check"] == "separation":
            vo, sr = decode(case["vo"])
            mix, _ = decode(case["mix"])
            o = case["opts"]
            r = separation(vo, mix, sr, min_lu=o.get("minLu", 10.0), gate_dbfs=o.get("gateDbfs", -44.0),
                           vo_offset=o.get("voOffset"))
        else:
            x, sr = decode(case["file"])
            w = k_weight(x, sr)
            r = (loop_score(x, sr, weighted=w) if case["check"] == "loop" else level_steps(x, sr, weighted=w))
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.__dict__.items()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def cli(argv, js: bool):
    node = ["node", str(JS / "bin" / "audio-bed-check.js")]
    exe = node if js else [sys.executable, "-m", "audio_bed_check"]
    p = subprocess.run(exe + argv, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    # the error line only: argparse wraps its usage line to the terminal, the port does not
    lines = p.stderr.strip().splitlines()
    return p.returncode, p.stdout, lines[-1] if lines else ""


STEP_LINE = re.compile(r'(.*dB at |\s*"step_at_s": )([\d.]+)(s \(max .*|,?)')


def cli_ties_only(py_out: str, js_out: str, edge: float) -> bool:
    """True when the outputs differ only in step positions, and on every such line JavaScript names
    the earliest of the positions where Python's own steps tie its largest."""
    la, lb = py_out.splitlines(), js_out.splitlines()
    if len(la) != len(lb):
        return False
    current, block_start = None, True
    for x, y in zip(la, lb, strict=True):
        m = re.match(r'\s*"file": (".*"),', x)
        if m:
            current = json.loads(m[1])
        elif block_start and x and not x.startswith(" "):
            current = x
        block_start = x == ""
        if x == y:
            continue
        mx, my = STEP_LINE.fullmatch(x), STEP_LINE.fullmatch(y)
        if not (mx and my and mx[1] == my[1] and mx[3] == my[3] and current):
            return False
        if float(my[2]) != min(tied_positions(current, edge)):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="store_true", help="every bed of the published loop sweep")
    sweep = ap.parse_args().sweep
    tmp = Path(tempfile.mkdtemp(prefix="abc-parity-"))
    cases, names = [], []
    for i, (name, x) in enumerate(beds(sweep)):
        path = tmp / f"bed{i:03d}.wav"
        write_wav(path, x, clip=True)
        for check in ("loop", "steps"):
            cases.append({"check": check, "file": str(path), "opts": {}})
            names.append(f"{check:5} {name}")
    for i, (name, vo, mix, opts) in enumerate(mixes()):
        v, m = tmp / f"vo{i:02d}.wav", tmp / f"mix{i:02d}.wav"
        write_wav(v, vo, clip=True)
        write_wav(m, mix, clip=True)
        cases.append({"check": "separation", "vo": str(v), "mix": str(m), "opts": opts})
        names.append(f"sep   {name}")
    manifest = tmp / "cases.json"
    manifest.write_text(json.dumps(cases))
    js = json.loads(subprocess.run(["node", str(JS / "scripts" / "dump.mjs"), str(manifest)],
                                   capture_output=True, text=True, check=True).stdout)
    bad = 0
    for name, case, j in zip(names, cases, js, strict=True):
        p = py_result(case)
        if "error" in p or "error" in j:
            ok = p.get("error") == j.get("error")   # class name and message
        else:
            ok = set(p) == set(j) and all(same(p[k], j[k]) for k in p if k != "step_at_s")
            if ok and "step_at_s" in p and not same(p["step_at_s"], j["step_at_s"]):
                # tied steps on a looped bed: JS reports the earliest; accept it only where
                # Python's own step ties its largest
                ok = j["step_at_s"] == min(tied_positions(case["file"]))
        if not ok:
            bad += 1
            diffs = {k: (p.get(k), j.get(k)) for k in set(p) | set(j) if not same(p.get(k), j.get(k))}
            print(f"MISMATCH {name}: {diffs}")
    print(f"library: {len(cases) - bad} of {len(cases)} cases match")

    flags = [
        ["bed"], ["loop", "--min-period", "1.5"], ["loop", "--loop-threshold", "0.5"],
        ["steps", "--max-step", "2.5", "--edge", "0"], ["steps", "--edge", "3"], ["bed", "--json"],
    ]
    files = sorted(str(p) for p in tmp.glob("bed*.wav"))
    runs, cli_bad = 0, 0

    def compare(argv, label, edge=0.75, output=True):
        nonlocal runs, cli_bad
        runs += 1
        a, b = cli(argv, js=False), cli(argv, js=True)
        if not output:   # help text: argparse formats it its own way; the exit code must agree
            a, b = (a[0],), (b[0],)
        if a == b or (a[0] == b[0] and a[2] == b[2] and cli_ties_only(a[1], b[1], edge)):
            return
        cli_bad += 1
        print(f"CLI MISMATCH {label}:\n  py {a}\n  js {b}")

    for f in flags:
        edge = float(f[f.index("--edge") + 1]) if "--edge" in f else 0.75
        compare([f[0], *files, *f[1:]], " ".join(f), edge)
    for i, (_, _, _, opts) in enumerate(mixes()):
        argv = ["separation", str(tmp / f"mix{i:02d}.wav"), "--vo", str(tmp / f"vo{i:02d}.wav")]
        if "voOffset" in opts:
            argv += ["--vo-offset", str(opts["voOffset"])]
        if "gateDbfs" in opts:
            argv += ["--gate", str(opts["gateDbfs"])]
        if "minLu" in opts:
            argv += ["--profile", "wcag"]
        compare(argv, f"separation {i}")
        compare([*argv, "--json"], f"separation {i} --json")
        compare([*argv, "--min-separation", "3.5"], f"separation {i} --min-separation")

    # the command line itself: odd orders, abbreviations, bad values
    b, m, v = files[0], str(tmp / "mix00.wav"), str(tmp / "vo00.wav")
    short = str(tmp / "short.wav")
    write_wav(short, textured(5, 77), clip=True)
    tiny, sub_window = str(tmp / "tiny.wav"), str(tmp / "subwindow.wav")
    write_wav(tiny, textured(4.5, 78), clip=True)
    write_wav(sub_window, textured(0.3, 79), clip=True)
    odd = [
        ["loop", b, "--json", b], ["loop", "-x"], ["loop", "--version"], ["bed", b, "--m", "3"],
        ["bed", b, "--max-s=2", "-x"], ["bed", b, "--max-step", "1e400"],
        ["loop", b, "--min-period", "1e400"],
        ["steps", b, "--edge", "1e999"], ["separation", m, "--vo", v, "--gate=-1e400"],
        ["separation", m, "--vo", v, "--vo-offset", "1e400"], ["bed", b, "--edge", "1_0"],
        ["bed", b, "--edge", " 2 "], ["bed", b, "--max-step", "abc"], ["bed", b, "--edge", "xyz"],
        ["loop", b, "--loop-threshold", "q"], ["separation", m, "--vo", v, "--gate", "q"],
        ["separation", m, "--vo", v, "--profile", "loud"], ["separation", m, "n.wav", "--vo", v],
        ["--bogus", "loop", b], ["loop", b, "--max-step", "3"], ["bed", "--", b], ["bed", b, "--json=yes"],
        ["bed", b, "--max-step"], [], ["nope"], ["separation", m], ["separation", "--vo", v],
        ["separation", "--vo", v, m, "--min-separation", "12", "--json"],
        ["separation", m, "--vo", v, "--min-sep", "25"], ["separation", m, "--vo", v, "--vo-offset", "-0.5"],
        ["loop", b, "--min-period", "9186.625"], ["steps", b, "--max-step", "1e22"],
        ["bed", b, "--loop-threshold", "1"], ["bed", b, "--max-step", "-1"], ["bed", b, "--max-step", "-.5"],
        ["steps", str(tmp / "missing.wav"), b], ["bed", "-my file.wav"], ["--vers"],
        ["steps", b, "--ffmpeg", str(tmp / "no-such-ffmpeg")],
        ["steps", short], ["bed", short, b], ["loop", short],
        # a file that could hold a repeat but is too short to measure one; and one under a 400 ms window
        ["loop", tiny, "--min-period", "2"], ["bed", tiny, b, "--min-period", "2"],
        ["loop", sub_window, "--min-period", "0.1"], ["loop", sub_window, b, "--min-period", "0.1"],
    ]
    for argv in odd:
        compare(argv, " ".join(argv) or "(no arguments)")
    for argv in (["--he"], ["-h"], ["loop", "--he"]):
        compare(argv, " ".join(argv), output=False)
    # help with anything else is refused, with the same message on both sides
    for argv in (["loop", "--bogus", "-h"], ["bed", b, "--help"], ["-h", "bed"], ["bed", "--hel", b]):
        compare(argv, " ".join(argv))
    # control characters and "::" in a name are escaped the same way
    weird = str(tmp / "a\x1b[31m::x\n.wav")
    compare(["steps", weird, b], "escaped name")
    print(f"cli: {runs - cli_bad} of {runs} runs match")
    sys.exit(1 if bad or cli_bad else 0)


if __name__ == "__main__":
    main()
