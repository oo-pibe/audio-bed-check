"""Reproduce the loop check's validation figures on generated beds. Not part of the test suite.

    python scripts/loop_sweep.py               # the published run
    python scripts/loop_sweep.py --seed 1000   # the same sweep on other seeds

Prints the worst score of the unlooped beds per length, the lowest score among the looped beds and
whether each was caught with the right period, and the highest score among beds with periodic swells.
Run from the repository root; it uses the generators in tests/synth.py. The 600 s beds make a full
run take a few minutes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_bed_check.checks import loop_score  # noqa: E402
from tests.synth import SR, fade_edges, gain, swell_gain, textured, tile  # noqa: E402

UNLOOPED = [(30, 120), (48, 120), (60, 120), (180, 20), (600, 20)]   # (seconds, beds)


def fade_out(x, seconds, power=1):
    y = x.astype(np.float64).copy()
    k = int(seconds * SR)
    y[-k:] *= np.linspace(1, 0, k) ** power
    return y.astype(np.float32)


def looped(seed):
    """(name, signal, period in seconds) for every kind of loop the docs say is caught."""
    loop = tile(textured(10, seed), 6)
    for s in (0.25, 0.5, 1.0, 2.0, 5.0):
        yield f"{s:g} s fade-out", fade_out(loop, s), 10.0
    yield "1 s fade in and out", fade_edges(loop, 1.0), 10.0
    yield "whole-file fade", fade_out(loop, 60.0), 10.0
    yield "30 s quadratic fade-out", fade_out(loop, 30.0, power=2), 10.0
    clip = textured(10, seed)
    yield "4 tiles then 20 s unrelated", np.concatenate([tile(clip, 4), textured(20, seed + 1)]), 10.0
    yield "20 s unrelated then 4 tiles", np.concatenate([textured(20, seed + 2), tile(clip, 4)]), 10.0
    yield "60 s unrelated then 2 tiles", np.concatenate([textured(60, seed + 3), tile(clip, 2)]), 10.0
    for c in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0):
        yield f"{c:g} s crossfade", tile(textured(12, seed), 4, crossfade=c), 12.0 - c
    rng = np.random.default_rng(seed)
    for spread in (3, 6):
        x = np.concatenate([gain(textured(10, seed), rng.uniform(-spread, spread)) for _ in range(6)])
        yield f"gains within +-{spread} dB", x, 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0, help="first seed (default 0)")
    seed = ap.parse_args().seed

    print("unlooped textured beds (none should fail)")
    print(f"  {'length':>7} {'beds':>5} {'failed':>7} {'worst':>6}")
    overall = 0.0
    for seconds, count in UNLOOPED:
        scores = [loop_score(textured(seconds, seed + j), SR) for j in range(count)]
        failed = sum(not r.passed for r in scores)
        worst = max(r.score for r in scores)
        overall = max(overall, worst)
        print(f"  {seconds:>6}s {count:>5} {failed:>7} {worst:>6.2f}")
    print(f"  worst of all: {overall:.2f}")

    print("looped beds (all should fail with the right period)")
    lowest, wrong = 1.0, []
    for name, x, period in looped(seed + 5000):
        r = loop_score(x, SR)
        lowest = min(lowest, r.score)
        if r.passed or r.period_s is None or abs(r.period_s - period) > 0.15:
            wrong.append(f"{name}: score {r.score:.2f}, period {r.period_s}")
    print(f"  lowest score: {lowest:.2f}; missed or wrong period: {len(wrong)}")
    for line in wrong:
        print(f"    {line}")

    print("periodic swells on a 120 s textured bed (none should fail)")
    highest, failed = 0.0, 0
    for j in range(10):
        bed = textured(120, seed + 6000 + j)
        every_20 = np.prod([swell_gain(len(bed), c, 4, 6) for c in np.arange(10, 120, 20)], axis=0)
        every_8 = np.prod([swell_gain(len(bed), c, 3, 6) for c in np.arange(5, 120, 8)], axis=0)
        for g in (every_20, every_8):
            r = loop_score((bed * g).astype(np.float32), SR)
            highest, failed = max(highest, r.score), failed + (not r.passed)
    print(f"  20 beds, failed: {failed}, highest score: {highest:.2f}")


if __name__ == "__main__":
    main()
