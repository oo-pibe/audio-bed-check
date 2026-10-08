# audio-bed-check

**Three faults that get past a listener who has heard the file too many times, caught by number.**

A bed is the ambience or music that sits under a voiceover or a cut. Three things go wrong with a
rendered one that nothing in a normal pipeline measures: the file repeats itself, the level jumps
where two pieces were joined, or the voice over it is not far enough above it to be heard without
effort. This command measures each one and exits non-zero when one fails.

It needs Python 3.10 or later and ffmpeg, either on PATH or named with `--ffmpeg` or
`AUDIO_BED_CHECK_FFMPEG`. It reads anything ffmpeg reads, video included.

```
pip install audio-bed-check

audio-bed-check bed ambience.wav              # loop and level steps
audio-bed-check separation final.mp4 --vo read.wav
```

```
ambience.wav
  FAIL  loop        1.00 (repeats every 12.0s, at or above 0.90)
  ok    step        1.42 dB at 18.5s (max 6.00)
  info  range       6.2 dB   transient 3.1 dB   peak -3.4 dBTP
```

Exit 0 when every check passes, 1 when any fails, 2 when it could not run (a bad flag, an unreadable
file, no speech found in the voiceover). `warn` lines, such as a file peaking under -60 dBTP or a
voiceover that seems to start somewhere other than `--vo-offset`, never change the exit code.
`--json` prints the numbers as a list of objects; the field list is in the command reference.
`python -m audio_bed_check` is the same command.

## The checks

`loop`: the frame-to-frame changes of the loudness envelope, correlated with themselves at every
lag over the best-matching 10 s window. A copied clip copies its texture, so a loop scores near 1.0
even when only part of the file repeats, or it fades out, or each repeat sits at a different gain.
Among the peaks that reach the threshold (`--loop-threshold`, 0.9), it reports the shortest one
scoring within 0.05 of the strongest. That is the repeat length rather than a multiple of it. A
repeat of 6 seconds or longer (`--min-period`) fails. A shorter one is reported and passes, because
music repeats at bar length.

`steps`: half-second loudness blocks every 0.1 s. The sustained step is the mean level over two
seconds after a half-second gap against the two seconds before; it fails above 6 dB (`--max-step`).
A bad join moves the level and it stays moved. A crowd surge spikes and comes back, so the largest
half-second transient and the range are printed but never fail the file. A change held about two
seconds or longer does fail, even if it comes back: a +7 dB plateau reads 3.9 dB held one second
and 6.8 held two.

`separation`: speech windows are found in the voiceover file, then compared with the bed-only gaps
between them. Each window's level is its 90th percentile of 50 ms K-weighted blocks. Both kinds of
window are measured in the mix, so the figure is voice plus bed against bed, which is closer to what
a listener hears than the gain on each stem. Profiles set the minimum: `music` 10 LU (the default),
`ambience` 15 LU, `wcag` 20 LU. The read needs pauses of 0.9 seconds or more. The voiceover's start
in the mix is estimated from the two envelopes, and a warning names it when it is more than 0.2 s
from `--vo-offset`; a mix whose bed-only windows are silent (the voiceover passed as the mix) is
refused.

| Check | Fails when | Default | Source |
|---|---|---|---|
| loop | repeat score at or above | 0.9, repeats of 6 s or longer | validation below |
| steps | sustained step above | 6 dB | production use |
| separation, music | voice less than | 10 LU above bed | Torcoli et al., JAES 2019 |
| separation, ambience | voice less than | 15 LU above bed | Torcoli et al., JAES 2019 |
| separation, wcag | voice less than | 20 LU above bed | WCAG 2 technique G56 (stated there in dB(A) SPL) |

Every default is a flag. The full list is in
[skills/audio-bed-check/references/cli.md](https://github.com/oo-pibe/audio-bed-check/blob/main/skills/audio-bed-check/references/cli.md).

## How well it works

The loop method was rebuilt before release; the earlier calibration figures no longer describe it.
It is validated on generated signals, and the figures here are from one run of
[scripts/loop_sweep.py](https://github.com/oo-pibe/audio-bed-check/blob/main/scripts/loop_sweep.py),
which anyone can repeat. In that run, 600 unlooped beds of 30 to 600 seconds produced no failures
and the worst scored 0.75. Other seeds reach about 0.78, and longer files score higher by chance.
Loops under fades, in part of a file, with crossfaded joins or at changing gain all failed with the
right period, the lowest at 0.96. On one other seed a 10 s loop at gains up to 6 dB apart was named
at 20 s; it still failed. Beds with a 6 dB swell every 8 or 20 seconds passed, the highest at 0.85.

A repeat is seen only when the repeating stretch is at least one period plus the 10 s window long,
so a 6 to 8 second clip played twice usually is not. The longest repeat it can see is the file
length minus the window: about 10.5 s less than the file, or a third less under 30 s. A gain cycle
with nothing under it, such as a steady tremolo or the same swell twice on room tone, reads as a
loop; the details are in
[references/checks.md](https://github.com/oo-pibe/audio-bed-check/blob/main/skills/audio-bed-check/references/checks.md).

The loudness code is ITU-R BS.1770-4 K-weighting written in numpy. The test suite compares its
momentary loudness with ffmpeg's `ebur128` filter on the same file (median difference under 0.1 LU,
95th percentile under 0.2 LU), and its true peak on a test tone (within 0.2 dB). These tests skip
when no ffmpeg is found, or when the one found was built without `ebur128`.

## From Python

```python
from audio_bed_check import decode, loop_score, level_steps, separation

samples, sr = decode("ambience.wav")
loop_score(samples, sr).passed        # False if it repeats
level_steps(samples, sr).step_db      # the worst sustained step, dB

vo, _ = decode("read.wav")
mix, _ = decode("final.mp4")
separation(vo, mix, sr, min_lu=15).separation_lu
```

The check functions are pure functions of a float array at 48 kHz; `decode` always returns that.

## For coding agents

The package ships an [Agent Skill](https://github.com/oo-pibe/audio-bed-check/tree/main/skills/audio-bed-check)
describing when to run which check and how to read the result. Claude Code users can install it as a
plugin:

```
/plugin marketplace add oo-pibe/audio-bed-check
/plugin install audio-bed-check@audio-bed-check
```

## Prior art

- ITU-R BS.1770-4, the loudness algorithm.
- Rafii and Pardo, REPET, IEEE TASLP 21(1), 2013: the repeating-period idea behind the loop check.
- Torcoli, Freke-Morin, Paulus, Simon and Shirley, "Preferred Levels for Background Ducking to
  Produce Esthetically Pleasing Audio for TV with Clear Speech", JAES 67(12), 2019,
  doi:10.17743/jaes.2019.0052: the source of the 10 LU (music) and 15 LU (ambience) floors. The
  same study found non-expert listeners wanted about 4 LU more than experts.
- WCAG 2 technique G56, the 20 dB speech-over-background rule, stated there in dB(A) SPL.
- ffmpeg's `ebur128` filter measures loudness; it does not judge loops, joins or separation.

## Origin

Extracted from the video pipeline at [Road to Kickoff](https://roadtokickoff.com). A bed there
stepped 7 dB at two joins, got past every automated check, and was caught by ear. This tool measures
the things those checks did not.

## License

MIT.
