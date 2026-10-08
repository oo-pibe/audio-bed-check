---
name: audio-bed-check
description: "Use when a rendered audio bed, ambience track or voiceover mix needs checking before it ships: whether a bed repeats itself (a loop), whether its level jumps at a join, or whether a voiceover sits far enough above the bed to be heard. Covers the audio-bed-check command and Python library (BS.1770 loudness, ffmpeg on PATH)."
license: MIT
---

# audio-bed-check

Three checks on finished audio, each a pass or fail with the number behind it. Install with
`pip install audio-bed-check`; ffmpeg must be on PATH or named in `AUDIO_BED_CHECK_FFMPEG`.
`python -m audio_bed_check` is the same command. Files keep their channels (stereo is measured per
channel and summed, as BS.1770 does); only the first audio track is read.

## Which check

- The bed on its own, before it goes under anything: `audio-bed-check bed BED.wav`. Runs `loop`
  (does it repeat?) and `steps` (does the level lurch?). Several files at once are fine.
- A rendered mix with a voiceover: `audio-bed-check separation MIX.mp4 --vo READ.wav`. Needs the
  voiceover file on its own, exactly as it was placed in the mix (`--vo-offset` if it does not start
  at zero).

Exit 0 means every check passed, 1 means at least one failed, 2 means a check could not run (no
ffmpeg, unreadable file, no speech found). `warn` lines (a silent file, a hot peak, an offset that
disagrees with the voiceover) never change the exit code; a `warn` line is something to look at. Add `--json` to get the numbers as a list
of objects; `--edge`, `--gate` and `--vo-offset` are not recorded in it.

Check the bed before it is encoded: ffmpeg's AAC at 128k rebuilt exact loops of steady noise to
scores between 0.88 and 0.97, some under the threshold. `loop` needs 13 s
and `steps` 6.5 s at the defaults; a shorter file reports ok with a note, not a verdict.

## Reading the result

- `loop`: score is the correlation of the level's frame-to-frame changes at the repeat it found,
  over the best-matching 10 s window; 1.0 is an exact repeat. Fails at 0.9 or above when the repeat
  is at or longer than `--min-period` (6 s). A shorter repeat is reported as a note, because music
  repeats at bar length and that is not a loop. A loop in part of the file, under a fade or at
  changing gain is still found. A repeat is seen only when the repeating stretch is at least one
  period plus 10 s long, so a 6 to 8 s clip played twice usually is not; a repeat longer than the
  file minus about 10.5 s is not seen either.
- `steps`: fails on a sustained step above `--max-step` (6 dB): the mean level over the 2 s after a
  0.5 s gap at a boundary against the 2 s before, on half-second blocks every 0.1 s. A change held
  about 2 s or longer fails even if it comes back (a +7 dB plateau reads 6.7 held 2 s). Range and
  the largest half-second transient are printed for information only, because a crowd surge comes
  back and a join does not.
- `separation`: the mix inside the voiceover's speech windows against the bed-only gaps between
  them, in LU. Fails under the profile's minimum: `music` 10, `ambience` 15, `wcag` 20. The read
  needs pauses of 0.9 s or more. When the voiceover seems to start somewhere other than
  `--vo-offset` (more than 0.2 s off), a `warn` line names where; rerun with that offset. A voice
  too far under the bed to match gives no estimate and no warning. A bed
  ducked under the speech barely changes the figure, so treat the floors as approximate for ducked
  mixes.

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

The check functions take a float array at 48 kHz, `(n,)` mono or `(n, 2)` stereo; `decode` returns
48 kHz, mono or stereo as the file is (more than two channels downmixed to two).

## Do not

- Do not run `separation` on the bed alone or on the voiceover alone; it needs the voiceover and the
  mix that contains it.
- Do not treat a high `transient` as a failure. Only `step` fails.
- Do not lower `--min-period` below the length of a musical bar unless you want bars flagged.

## Limitations

A steady gain cycle with no texture under it (tremolo, the same swell twice on room tone, beating
between detuned tones) also fails; the check measures repetition, not provenance.
