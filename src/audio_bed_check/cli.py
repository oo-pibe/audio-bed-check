"""Command line: subcommands loop, steps, bed, separation. Exit 0 pass, 1 fail, 2 usage/decode error."""

import argparse
import dataclasses
import json
import sys
from importlib.metadata import version as _version

from .checks import NoSpeechError, level_steps, loop_score, separation
from .decode import DecodeError, decode

PROFILES = {"music": 10.0, "ambience": 15.0, "wcag": 20.0}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-bed-check",
        description="Check an audio bed for loops and level steps, or a mix for voice-over-bed separation. "
                    "Exit 0 when every check passes, 1 when any fails, 2 when a check could not run.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_version('audio-bed-check')}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--json", action="store_true", help="print a JSON list, one object per file")
        sp.add_argument("--ffmpeg", metavar="PATH",
                        help="ffmpeg to use (default: $AUDIO_BED_CHECK_FFMPEG, then PATH)")

    def loop_flags(sp):
        sp.add_argument("--min-period", type=float, default=6.0, metavar="S",
                        help="shortest repeat length that counts as a loop, seconds "
                             "(default 6; shorter repeats are noted)")
        sp.add_argument("--loop-threshold", type=float, default=0.75, metavar="SCORE",
                        help="autocorrelation score at which a repeat fails (default 0.75)")

    def steps_flags(sp):
        sp.add_argument("--max-step", type=float, default=6.0, metavar="DB",
                        help="largest sustained level step allowed, dB (default 6)")
        sp.add_argument("--edge", type=float, default=0.75, metavar="S",
                        help="seconds ignored at each end so fades are not read as steps (default 0.75)")

    for name, flag_sets, text in (
        ("loop", (loop_flags,), "does the bed repeat itself?"),
        ("steps", (steps_flags,), "does the level lurch at a join?"),
        ("bed", (loop_flags, steps_flags), "loop and steps together"),
    ):
        sp = sub.add_parser(name, help=text, description=text)
        sp.add_argument("files", nargs="+", metavar="BED", help="audio or video file; anything ffmpeg reads")
        for add in flag_sets:
            add(sp)
        common(sp)

    sp = sub.add_parser("separation", help="is the voice far enough above the bed?",
                        description="Measures the rendered mix inside the voiceover's speech windows "
                                    "against the gaps between them.")
    sp.add_argument("mix", metavar="MIX", help="the rendered mix (audio or video)")
    sp.add_argument("--vo", required=True, metavar="VO",
                    help="the voiceover on its own, as it was placed in the mix")
    sp.add_argument("--profile", choices=PROFILES, default="music",
                    help="minimum separation: music 10 LU, ambience 15 LU, wcag 20 LU (default music)")
    sp.add_argument("--min-separation", type=float, metavar="LU", help="override the profile's minimum")
    sp.add_argument("--gate", type=float, default=-44.0, metavar="DBFS",
                    help="level above which the voiceover counts as speech (default -44)")
    sp.add_argument("--vo-offset", type=float, default=0.0, metavar="S",
                    help="where the voiceover starts in the mix, seconds (default 0)")
    common(sp)
    return p


def _result(r) -> dict:
    """A result as a dict with floats rounded to 2 decimals; the dataclasses keep full precision."""
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in dataclasses.asdict(r).items()}


def _run(args) -> list[dict]:
    results = []
    if args.command == "separation":
        vo, sr = decode(args.vo, args.ffmpeg)
        mix, _ = decode(args.mix, args.ffmpeg)
        min_lu = args.min_separation if args.min_separation is not None else PROFILES[args.profile]
        r = separation(vo, mix, sr, min_lu=min_lu, gate_dbfs=args.gate, vo_offset=args.vo_offset)
        results.append({"file": args.mix, "passed": r.passed, "profile": args.profile,
                        "separation": _result(r)})
        return results
    for path in args.files:
        samples, sr = decode(path, args.ffmpeg)
        entry = {"file": path, "passed": True}
        if args.command in ("loop", "bed"):
            r = loop_score(samples, sr, min_period=args.min_period, threshold=args.loop_threshold)
            entry["loop"] = _result(r)
            entry["passed"] = entry["passed"] and r.passed
        if args.command in ("steps", "bed"):
            r = level_steps(samples, sr, max_step=args.max_step, edge=args.edge)
            entry["steps"] = _result(r)
            entry["passed"] = entry["passed"] and r.passed
        results.append(entry)
    return results


def _verdict(passed: bool) -> str:
    return "ok  " if passed else "FAIL"


def _render(entry: dict) -> str:
    lines = [entry["file"]]
    if "loop" in entry:
        r = entry["loop"]
        if r["notes"]:
            detail = r["notes"][0]
        elif r["period_s"] is None:
            detail = f"no repeat at or above {r['threshold']:g}"
        else:
            detail = f"repeats every {r['period_s']:.1f}s, at or above {r['threshold']:g}"
        lines.append(f"  {_verdict(r['passed'])}  loop        {r['score']:.2f} ({detail})")
    if "steps" in entry:
        r = entry["steps"]
        if r["step_at_s"] is None:
            lines.append(f"  ok    step        {r['notes'][0]}")
        else:
            lines.append(f"  {_verdict(r['passed'])}  step        {r['step_db']:.1f} dB at "
                         f"{r['step_at_s']:.1f}s (max {r['max_step']:g})")
        lines.append(f"  info  range       {r['range_db']:.1f} dB   transient {r['transient_db']:.1f} dB"
                     f"   peak {r['peak_dbtp']:.1f} dBTP")
    if "separation" in entry:
        r = entry["separation"]
        lines.append(f"  {_verdict(r['passed'])}  separation  {r['separation_lu']:.1f} LU "
                     f"(speech {r['speech_lkfs']:.1f}, bed {r['bed_lkfs']:.1f} LKFS; "
                     f"min {r['min_lu']:.1f}, {entry['profile']})")
        lines.append(f"  info  peak        {r['peak_dbtp']:.1f} dBTP   speech windows {r['runs']}"
                     f"   bed windows {r['gaps']}")
        for w in r["warnings"]:
            lines.append(f"  warn  {w}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = _run(args)
    except (DecodeError, NoSpeechError) as e:
        print(f"audio-bed-check: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n\n".join(_render(r) for r in results))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
