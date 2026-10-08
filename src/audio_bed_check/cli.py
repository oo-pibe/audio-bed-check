"""Command line: subcommands loop, steps, bed, separation. Exit 0 pass, 1 fail, 2 usage/decode error."""

import argparse
import dataclasses
import json
import math
import sys
from importlib.metadata import version as _version

from .checks import SILENT_NOTE, level_steps, loop_score, separation
from .decode import DecodeError, decode
from .loudness import k_weight

PROFILES = {"music": 10.0, "ambience": 15.0, "wcag": 20.0}
OVERRIDE = "--min-separation"   # what the profile column says when the flag replaced the profile


def _number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text}")
    return value


def _positive(text: str) -> float:
    value = _number(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {text}")
    return value


def _non_negative(text: str) -> float:
    value = _number(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more, got {text}")
    return value


def _score(text: str) -> float:
    value = _number(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be between 0 (exclusive) and 1, got {text}")
    return value


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
        sp.add_argument("--min-period", type=_positive, default=6.0, metavar="S",
                        help="shortest repeat length that counts as a loop, seconds "
                             "(default 6; shorter repeats are noted)")
        sp.add_argument("--loop-threshold", type=_score, default=0.9, metavar="SCORE",
                        help="correlation at which a repeat fails, 0 to 1 (default 0.9)")

    def steps_flags(sp):
        sp.add_argument("--max-step", type=_positive, default=6.0, metavar="DB",
                        help="largest sustained level step allowed, dB (default 6)")
        sp.add_argument("--edge", type=_non_negative, default=0.75, metavar="S",
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
    sp.add_argument("--min-separation", type=_positive, metavar="LU", help="override the profile's minimum")
    sp.add_argument("--gate", type=_number, default=-44.0, metavar="DBFS",
                    help="level above which the voiceover counts as speech (default -44)")
    sp.add_argument("--vo-offset", type=_number, default=0.0, metavar="S",
                    help="where the voiceover starts in the mix, seconds (default 0)")
    common(sp)
    return p


def _round(value: float) -> float:
    """2 decimals, and anything within 0.005 of zero is 0.0, so nothing prints as -0.0 or -0.00."""
    return 0.0 if abs(value) < 0.005 else round(value, 2)


def _result(r) -> dict:
    """A result as a dict with floats rounded to 2 decimals; the dataclasses keep full precision."""
    return {k: _round(v) if isinstance(v, float) else v for k, v in dataclasses.asdict(r).items()}


def _run(args) -> list[dict]:
    results = []
    if args.command == "separation":
        vo, sr = decode(args.vo, args.ffmpeg)
        mix, _ = decode(args.mix, args.ffmpeg)
        overridden = args.min_separation is not None
        min_lu = args.min_separation if overridden else PROFILES[args.profile]
        r = separation(vo, mix, sr, min_lu=min_lu, gate_dbfs=args.gate, vo_offset=args.vo_offset)
        results.append({"file": args.mix, "passed": r.passed,
                        "profile": OVERRIDE if overridden else args.profile, "separation": _result(r)})
        return results
    for path in args.files:
        try:
            samples, sr = decode(path, args.ffmpeg)
        except DecodeError as e:   # one unreadable file must not throw away the rest of the batch
            results.append({"file": path, "passed": False, "error": str(e)})
            continue
        entry = {"file": path, "passed": True}
        weighted = k_weight(samples, sr)   # once per file, shared by both checks
        if args.command in ("loop", "bed"):
            r = loop_score(samples, sr, min_period=args.min_period, threshold=args.loop_threshold,
                           weighted=weighted)
            entry["loop"] = _result(r)
            entry["passed"] = entry["passed"] and r.passed
        if args.command in ("steps", "bed"):
            r = level_steps(samples, sr, max_step=args.max_step, edge=args.edge, weighted=weighted)
            entry["steps"] = _result(r)
            entry["passed"] = entry["passed"] and r.passed
        results.append(entry)
    return results


def _num(value: float, decimals: int) -> str:
    """A printed number: fixed decimals, and never "-0.0" or "-0.00" for something that rounds to zero."""
    text = f"{value:.{decimals}f}"
    return f"{0.0:.{decimals}f}" if float(text) == 0 else text


def _verdict(passed: bool) -> str:
    return "ok  " if passed else "FAIL"


def _render(entry: dict) -> str:
    lines = [entry["file"]]
    if "error" in entry:
        lines.append(f"  error {entry['error']}")
    if "loop" in entry:
        r = entry["loop"]
        if r["notes"]:
            detail = r["notes"][0]
        elif r["period_s"] is None:
            detail = f"no repeat at or above {_num(r['threshold'], 2)}"
        else:
            detail = f"repeats every {_num(r['period_s'], 1)}s, at or above {_num(r['threshold'], 2)}"
        lines.append(f"  {_verdict(r['passed'])}  loop        {_num(r['score'], 2)} ({detail})")
    if "steps" in entry:
        r = entry["steps"]
        notes = [n for n in r["notes"] if n != SILENT_NOTE]
        if r["step_at_s"] is None:
            detail = notes[0] if notes else "no level step measured"
            lines.append(f"  {_verdict(r['passed'])}  step        {detail}")
        else:
            lines.append(f"  {_verdict(r['passed'])}  step        {_num(r['step_db'], 2)} dB at "
                         f"{_num(r['step_at_s'], 1)}s (max {_num(r['max_step'], 2)})")
        lines.append(f"  info  range       {_num(r['range_db'], 1)} dB   "
                     f"transient {_num(r['transient_db'], 1)} dB   peak {_num(r['peak_dbtp'], 1)} dBTP")
        if SILENT_NOTE in r["notes"]:
            lines.append(f"  warn  {SILENT_NOTE}")
    if "separation" in entry:
        r = entry["separation"]
        lines.append(f"  {_verdict(r['passed'])}  separation  {_num(r['separation_lu'], 2)} LU "
                     f"(speech {_num(r['speech_lkfs'], 2)}, bed {_num(r['bed_lkfs'], 2)} LKFS; "
                     f"min {_num(r['min_lu'], 2)}, {entry['profile']})")
        lines.append(f"  info  peak        {_num(r['peak_dbtp'], 1)} dBTP   speech windows {r['runs']}   "
                     f"bed windows {r['gaps']}")
        for w in r["warnings"]:
            lines.append(f"  warn  {w}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = _run(args)
    except (DecodeError, ValueError) as e:   # NoSpeechError and the library's argument guards are ValueErrors
        print(f"audio-bed-check: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n\n".join(_render(r) for r in results))
    if any("error" in r for r in results):
        return 2
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
