# audio-bed-check

**Three faults that get past a listener who has heard the file too many times, caught by number.**

A bed is the ambience or music that sits under a voiceover or a cut. Three things go wrong with a
rendered one that nothing in a normal pipeline measures: the file repeats itself, the level jumps
where two pieces were joined, or the voice over it is not far enough above it to be heard without
effort. This command checks each one and exits non-zero when it finds it.

It needs Python 3.10 or later and ffmpeg on PATH. It reads anything ffmpeg reads, video included.

```
pip install audio-bed-check

audio-bed-check bed ambience.wav              # loop and level steps
audio-bed-check separation final.mp4 --vo read.wav
```

```
ambience.wav
  FAIL  loop        0.97 (repeats every 12.0s, at or above 0.75)
  ok    step        1.4 dB at 18.5s (max 6)
  info  range       6.2 dB   transient 3.1 dB   peak -3.4 dBTP
```

Exit 0 when every check passes, 1 when any fails, 2 when a check could not run. `--json` prints
the numbers as a list of objects.

## The checks

**loop.** The loudness envelope's autocorrelation, by FFT. It reports the shortest repeat it finds
and fails when that repeat is 6 seconds or longer (`--min-period`) and the score is 0.75 or higher
(`--loop-threshold`). A shorter repeat is a note, not a failure: music repeats at bar length.

**steps.** Half-second loudness blocks. The sustained step is the mean level two seconds after a
boundary against two seconds before; it fails above 6 dB (`--max-step`). A join stays shifted. A
crowd surge spikes and comes back, so the largest transient and the range are printed but never
fail the file.

**separation.** Speech windows are found in the voiceover file, then measured in the mix and
compared with the bed-only gaps between them. The difference, in LU, is what a listener hears.
Profiles set the minimum: `music` 10 LU, `ambience` 15 LU, `wcag` 20 LU.

| Check | Fails when | Default | Source |
|---|---|---|---|
| loop | repeat score at or above | 0.75, repeats of 6 s or longer | calibration below |
| steps | sustained step above | 6 dB | production use |
| separation, music | voice less than | 10 LU above bed | Torcoli et al., JAES 2019 |
| separation, ambience | voice less than | 15 LU above bed | Torcoli et al., JAES 2019 |
| separation, wcag | voice less than | 20 LU above bed | WCAG 2 technique G56 (stated there in dB(A) SPL) |

Every default is a flag. The full list is in
[skills/audio-bed-check/references/cli.md](https://github.com/oo-pibe/audio-bed-check/blob/main/skills/audio-bed-check/references/cli.md).

## How well it works

The loop check was calibrated on 36 stadium ambience beds from a production reel pipeline. Looped
files scored 0.79 to 1.00, with the reported lag equal to the repeat length. Unlooped files scored
0.11 to 0.43. Two caveats from the same set: a concourse recording with its own regular rhythm
scored 0.605 and was missed at the 0.75 threshold, and a composed music bed scored 0.82 at its bar
length, which is why the minimum period exists. Those files are not in this repository; the test
suite uses generated signals with known repeats, joins and gains.

The loudness code is ITU-R BS.1770-4 K-weighting written in numpy. The test suite checks it against
ffmpeg's `ebur128` filter on the same file, within 0.5 LU.

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
- Rafii and Pardo, REPET, IEEE TASLP 2013: the repeating-period idea behind the loop check.
- Torcoli, Freke-Morin, Paulus, Simon and Shirley, "Preferred Levels for Background Ducking to
  Produce Esthetically Pleasing Audio for TV with Clear Speech", JAES 67(12), 2019,
  doi:10.17743/jaes.2019.0052: the source of the 10 LU (music) and 15 LU (ambience) floors. The
  same study found non-expert listeners wanted about 4 LU more than experts.
- WCAG 2 technique G56, the 20 dB speech-over-background rule, stated there in dB(A) SPL.
- ffmpeg's `ebur128` filter measures loudness; it does not judge loops, joins or separation.

## Origin

Extracted from the video pipeline at [Road to Kickoff](https://roadtokickoff.com), where a bed that
stepped 7 dB at two joins shipped through every automated gate and was caught by ear. The checks
here are the measurements that were missing.

## Licence

MIT.
