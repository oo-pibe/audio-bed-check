# Command reference

`audio-bed-check SUBCOMMAND ...`, or `python -m audio_bed_check SUBCOMMAND ...` (the same command).
`--help` on any subcommand lists its flags.
`audio-bed-check --version` prints the version. It is a top-level flag, not one of the flags
below, and goes before any subcommand.

Files can be anything ffmpeg reads, video included. Each is decoded once to 48 kHz. Files keep
their channels: mono stays mono, stereo is measured per channel and summed the way BS.1770 does,
more than two channels are downmixed to two by ffmpeg; only the first audio track is read.

## Flags on every subcommand

- `--json`: print a JSON list, one object per file, with every number rounded to 2 decimals. Each
  object has `file`, `passed` and the sections that ran; a file that could not be decoded has
  `error` instead of sections. The fields:
  - `loop`: `score`, `period_s` (null when no repeat reached the threshold), `threshold`,
    `min_period_s`, `passed`, `notes`.
  - `steps`: `step_db`, `step_at_s` (null when too short or nothing changed), `range_db`,
    `transient_db`, `peak_dbtp`, `max_step`, `passed`, `notes`.
  - `separation`: `separation_lu`, `speech_lkfs`, `bed_lkfs`, `runs`, `gaps`, `peak_dbtp`,
    `min_lu`, `passed`, `estimated_offset_s` (null when either file is under 0.4 s), `warnings`.
  - top level, separation only: `profile`, the profile name, or `--min-separation` when that flag
    replaced it.

  `--edge`, `--gate` and `--vo-offset` are not recorded in the output; keep the command line if you
  need them.
- `--ffmpeg PATH`: the ffmpeg binary to use. Default: `$AUDIO_BED_CHECK_FFMPEG`, then `ffmpeg` on
  PATH.

Numeric flags are checked on the way in: periods, steps and separations must be positive, the loop
threshold between 0 and 1, the edge zero or more, every value finite. A bad value is a usage error
(exit 2) before any file is read.

## loop BED [BED ...]

- `--min-period S` (6): the shortest repeat that counts as a loop. A strong repeat under this is
  reported as a note (music repeats at bar length) and does not fail.
- `--loop-threshold SCORE` (0.9): the correlation (of the level's frame-to-frame changes, over the
  best-matching 10 s window) at which a repeat fails. 1.0 is an exact copy.

## steps BED [BED ...]

- `--max-step DB` (6): the largest sustained level step allowed (mean of the 2 s after a 0.5 s
  gap at a boundary against the 2 s before it).
- `--edge S` (0.75): seconds ignored at each end, rounded up to whole 0.1 s frames, so a fade
  in or out is not read as a step.

## bed BED [BED ...]

`loop` and `steps` together, with all four flags above. A failing check does not stop the batch;
every file is reported. A file that cannot be decoded is reported with an `error` line and the run
exits 2 after the other files are checked; in `--json` its entry is `{"file", "passed": false,
"error"}`. The same holds for `loop` and `steps` with several files.

## separation MIX --vo VO

- `--vo VO` (required): the voiceover on its own, exactly as it was placed in the mix.
- `--profile music|ambience|wcag` (music): the minimum separation, 10, 15 or 20 LU.
- `--min-separation LU`: overrides the profile's minimum.
- `--gate DBFS` (-44): the level above which the voiceover counts as speech.
- `--vo-offset S` (0): where the voiceover starts in the mix.

## Output

Each block starts with the file name; the `ok`, `FAIL`, `info` and `warn` lines follow, or a single
`error` line for a file that could not be decoded. `info` and `warn` lines never affect the exit
code: a `warn` line is something to look at (a silent file, a hot peak, an offset that disagrees
with the voiceover), not a failure. Numbers print with fixed decimals, and a value that rounds to
zero prints as 0.0, never -0.0.

## Messages

- "too short to test for a repeat longer than Ns": the file is under twice the minimum period plus
  a second, so no repeat that long could be seen. Passes, with the score 0.
- "repeats every Ns, under --min-period Ms, not flagged": a strong repeat shorter than the minimum
  period was found. Passes. Lower `--min-period` if you want it flagged.
- "too short to measure level steps (needs at least 6.5s)": fewer than 45 half-second blocks (one
  every 0.1 s) remain after the edges are dropped; 6.5 s at the default edge. Passes, with zeros.
- "no level change anywhere": the steps check on digital silence, or on a file whose every block is
  under -70 LKFS (counted as -70 for the step); every boundary measures 0.0 dB, so no place is
  named. Passes.
- "the file is silent (peak under -60 dBTP)": a `warn` line under the steps output (in `--json`, the
  last entry of the steps `notes`). The file peaks at or under -60 dBTP: probably the wrong file, a
  muted export or a bed rendered at the wrong gain. It does not change the verdict or the exit code.
- "level is constant; nothing to correlate": the loop check on digital silence or a constant tone
  (under 0.01 dB of envelope variation). Passes with score 0. A steady bed still varies by tenths of
  a dB and is checked normally.
- "min_period must be positive" / "threshold must be between 0 (exclusive) and 1" /
  "edge must be >= 0": library argument guards. The command line rejects these values before they
  get this far; a Python caller sees the ValueError.
- "weighted must be k_weight(samples, sr): same shape as samples": library only; `loop_score` or
  `level_steps` was given a precomputed K-weighted signal that does not match the samples.
- "hop must be at least one sample": library only; the loudness blocking function was given a hop
  under one sample.
- "must be a finite number, got": a numeric flag was NaN or infinite. Argparse usage error, exit 2.
- "must be a positive number, got": a flag that needs a value above zero got zero or less. Usage
  error, exit 2.
- "must be zero or more, got": `--edge` got a negative value. Usage error, exit 2.
- "must be between 0 (exclusive) and 1, got": `--loop-threshold` was outside 0 to 1. Usage error,
  exit 2.
- "no speech found in the voiceover above -44 dBFS": nothing in the voiceover file crossed the gate
  for 250 ms. Exit 2. Check the file, or lower `--gate`.
- "the voiceover has no pause of 0.9s or more between speech runs (0.4s after a 0.25s guard at each
  end); either --vo is the mix rather than the voiceover alone, or its noise floor is above --gate":
  no bed-only window could be found. Exit 2. The check needs pauses in the read of 0.9 s or more; a
  room tone or hiss above `--gate` fills every pause, so raise `--gate` above it.
- "the bed-only windows are silent in the mix; is MIX the rendered mix?": the windows between speech
  runs read -70 LKFS or less in the mix, so the mix holds no bed. Usually the voiceover was given as
  MIX. Exit 2.
- "the voiceover's speech windows fall outside the mix; check --vo-offset": every speech window, or
  every gap window, shifted by the offset, lands outside the mix (before its start or past its
  end). Exit 2.
- "mix true peak above -1 dBTP": a warning on the separation output; the mix has no headroom.
- "the voiceover seems to start at Ns in the mix; check --vo-offset": a warning. The voiceover's
  loudness envelope inside its speech runs was cross-correlated with the mix's, and the best match
  is more than 0.2 s from `--vo-offset`. The figure is also in `--json` as `estimated_offset_s`
  (0.1 s resolution). A read with regular pauses can match nearly as well at several lags; the given
  offset stands when its match is within 2% of the best.
- "N window(s) fall outside the mix; check --vo-offset": a warning; some speech or gap windows,
  shifted by the offset, landed before the start or past the end of the mix and were left out.
- "ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG": exit 2.
- "ffmpeg not found at X; check --ffmpeg or AUDIO_BED_CHECK_FFMPEG": `--ffmpeg` or the variable named
  something that is not an executable file (a missing path, a file without execute permission, or a
  name not on PATH). Exit 2.
- "FILE: ffmpeg failed to start (signal N)" or "FILE: ffmpeg failed to start (...)": ffmpeg itself
  crashed, or could not load its shared libraries; then the loader's line naming the missing
  library follows, even when the loader aborted ffmpeg with a signal. The file was never read. Fix
  the ffmpeg install, or point `--ffmpeg` at a working one. Exit 2.
- "FILE: ffmpeg could not decode it (it has no audio stream)": the file has video only. Exit 2.
- "FILE: no such file": exit 2.
- "FILE: ffmpeg could not decode it (...)": the last line of ffmpeg's own error follows. Exit 2.
  The file is always opened as a local file, so a name that looks like a URL or an ffmpeg protocol
  (`concat:a.wav|b.wav`) is read as that file and usually fails here.
- "FILE: decoded to no audio": ffmpeg found an audio stream but produced no samples (an empty or
  zero-length track). Exit 2.
- "ffmpeg returned something that is not a WAV stream": the decoder asks ffmpeg for a WAV on stdout
  and did not get one; a broken or very unusual ffmpeg build. Exit 2.
- "resample to 48 kHz": raised by the library functions when handed an array at another rate.
  `decode()` always returns 48 kHz, so this only reaches callers who bring their own arrays.
- "pass mono or stereo samples": raised by the library functions for an array that is not `(n,)` or
  `(n, 2)`. `decode()` downmixes anything with more than two channels to two, so this too only
  reaches callers who bring their own arrays.
- "samples must be finite floats": raised by the library functions for an integer array (scale it
  to floats in [-1, 1] first) or one holding NaN or infinity.

## Exit codes

- 0: every check passed.
- 1: at least one check failed.
- 2: usage error (argparse prints its usage and error line), or a check that could not run (one
  sentence on stderr, prefixed `audio-bed-check:`). For `loop`, `steps` and `bed`, a file that
  cannot be decoded is reported with an `error` line and the run exits 2 after the other files are
  checked; for `separation`, a decode error is one sentence on stderr.
