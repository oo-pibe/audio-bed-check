---
name: audio-bed-check
description: "Use when a rendered audio bed, ambience track or voiceover mix needs checking before it ships: whether a bed repeats itself (a loop), whether its level jumps at a join, or whether a voiceover sits far enough above the bed to be heard. Covers the audio-bed-check command and Python library (BS.1770 loudness, ffmpeg on PATH)."
license: MIT
---

# audio-bed-check

Three checks on finished audio, each a pass or fail with the number behind it. Install with
`pip install audio-bed-check`; ffmpeg must be on PATH or named in `AUDIO_BED_CHECK_FFMPEG`.

## Which check

- The bed on its own, before it goes under anything: `audio-bed-check bed BED.wav`. Runs `loop`
  (does it repeat?) and `steps` (does the level lurch?). Several files at once are fine.
- A rendered mix with a voiceover: `audio-bed-check separation MIX.mp4 --vo READ.wav`. Needs the
  voiceover file on its own, exactly as it was placed in the mix (`--vo-offset` if it does not start
  at zero).

Exit 0 means every check passed, 1 means at least one failed, 2 means a check could not run (no
ffmpeg, unreadable file, no speech found). Add `--json` to get the numbers as a list of objects.

## Reading the result

- `loop`: score is the autocorrelation of the loudness envelope at the repeat it found; 1.0 is an
  exact repeat. Fails at 0.75 or above when the repeat is at or longer than `--min-period` (6 s). A
  shorter repeat is reported as a note, because music repeats at bar length and that is not a loop.
  A clip repeated once to fill the file is caught by a second test on the overlapping halves.
- `steps`: fails on a sustained step above `--max-step` (6 dB), the mean level 2 s after a boundary
  against 2 s before. Range and the largest half-second transient are printed for information only;
  a crowd surge is content, a join is a fault.
- `separation`: the mix inside the voiceover's speech windows against the bed-only gaps between
  them, in LU. Fails under the profile's minimum: `music` 10, `ambience` 15, `wcag` 20.

Thresholds and their sources: [references/checks.md](references/checks.md). Every flag and every
message: [references/cli.md](references/cli.md).

## From Python

```python
from audio_bed_check import decode, loop_score, level_steps, separation

samples, sr = decode("bed.wav")
print(loop_score(samples, sr).passed, level_steps(samples, sr).step_db)
vo, _ = decode("read.wav"); mix, _ = decode("final.mp4")
print(separation(vo, mix, sr, min_lu=15).separation_lu)
```

The check functions take any float array at 48 kHz; `decode` always returns 48 kHz mono.

## Do not

- Do not run `separation` on the bed alone or on the voiceover alone; it needs the voiceover and the
  mix that contains it.
- Do not treat a high `transient` as a failure. Only `step` fails, by design.
- Do not lower `--min-period` below the length of a musical bar unless you want bars flagged.
