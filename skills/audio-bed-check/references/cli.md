# Command reference

`audio-bed-check SUBCOMMAND ...`. `--help` on any subcommand lists its flags; `--version` prints the
version. Files can be anything ffmpeg reads, video included; each is decoded once to 48 kHz mono.

## Flags on every subcommand

- `--json`: print a JSON list, one object per file, with every number (rounded to 2 decimals), the
  thresholds used, `passed`, `notes` and `warnings`. Keys are the result field names. Separation
  entries also carry `profile`: the profile name, or `--min-separation` when that flag replaced it.
- Numeric flags are checked on the way in: periods, steps and separations must be positive, the
  loop threshold between 0 and 1, the edge zero or more, every value finite. A bad value is a usage
  error (exit 2) before any file is read.
- `--ffmpeg PATH`: the ffmpeg binary to use. Default: `$AUDIO_BED_CHECK_FFMPEG`, then `ffmpeg` on
  PATH.

## loop BED [BED ...]

- `--min-period S` (6): the shortest repeat that counts as a loop. A strong repeat under this is
  reported as a note (music repeats at bar length) and does not fail.
- `--loop-threshold SCORE` (0.75): the autocorrelation score at which a repeat fails.

## steps BED [BED ...]

- `--max-step DB` (6): the largest sustained level step allowed (mean of the 2 s after a boundary
  against the 2 s before).
- `--edge S` (0.75): seconds ignored at each end, rounded up to whole half-second blocks, so a fade
  in or out is not read as a step.

## bed BED [BED ...]

`loop` and `steps` together, with all four flags above. Keeps going after a failure so a batch
reports every file.

## separation MIX --vo VO

- `--vo VO` (required): the voiceover on its own, exactly as it was placed in the mix.
- `--profile music|ambience|wcag` (music): the minimum separation, 10, 15 or 20 LU.
- `--min-separation LU`: overrides the profile's minimum.
- `--gate DBFS` (-44): the level above which the voiceover counts as speech.
- `--vo-offset S` (0): where the voiceover starts in the mix.

## Output

One block per file. Each line starts with `ok`, `FAIL`, `info` or `warn`. `info` lines never affect
the exit code.

## Messages

- "too short to test for a repeat longer than Ns": the file is under twice the minimum period plus
  a second, so no repeat that long could be seen. Passes, with the score 0.
- "repeats every Ns, under --min-period Ms, not flagged": a strong repeat shorter than the minimum
  period was found. Passes. Lower `--min-period` if you want it flagged.
- "too short to measure level steps (needs at least 6.5s)": fewer than nine half-second blocks remain
  after the edges are dropped. Passes, with zeros.
- "level is constant; nothing to correlate": the loop check on digital silence or a constant tone
  (under 0.01 dB of envelope variation). Passes with score 0. A steady bed still varies by tenths of
  a dB and is checked normally.
- "min_period must be positive" / "threshold must be between 0 (exclusive) and 1" /
  "edge must be >= 0": library argument guards. The command line rejects these values before they get this far;
  a Python caller sees the ValueError.
- "no speech found in the voiceover above -44 dBFS": nothing in the voiceover file crossed the gate
  for 250 ms. Exit 2. Check the file, or lower `--gate`.
- "the voiceover has no gap of 0.4s or more between speech runs, so there is no bed-only window to
  compare against": the read is continuous. Exit 2. The check needs pauses in the read.
  (The 0.4 is the module constant and prints from it.)
- "the voiceover's speech windows fall outside the mix; check --vo-offset": the windows, shifted by
  the offset, land past the end of the mix. Exit 2.
- "mix true peak above -1 dBTP": a warning on the separation output; the mix has no headroom.
- "N window(s) fall outside the mix; check --vo-offset": a warning; some speech or gap windows,
  shifted by the offset, landed before the start or past the end of the mix and were left out.
- "ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG": exit 2. When `--ffmpeg` or
  the variable named a path that does not exist, the message says "ffmpeg not found at PATH; ..."
  instead.
- "FILE: ffmpeg could not decode it (it has no audio stream)": the file has video only. Exit 2.
- "FILE: no such file": exit 2.
- "FILE: ffmpeg could not decode it (...)": the last line of ffmpeg's own error follows. Exit 2.
- "FILE: decoded to no audio": ffmpeg produced no samples (a video with no audio track, for
  instance). Exit 2.
- "ffmpeg returned something that is not a WAV stream": the decoder asks ffmpeg for a WAV on stdout
  and did not get one; a broken or very unusual ffmpeg build. Exit 2.
- "resample to 48 kHz": raised by the library functions when handed an array at another rate.
  `decode()` always returns 48 kHz, so this only reaches callers who bring their own arrays.

## Exit codes

- 0: every check passed.
- 1: at least one check failed.
- 2: usage error, decode error, or a check that could not run (`NoSpeechError`). One sentence on
  stderr, prefixed `audio-bed-check:`.
