# The three checks

Loudness uses ITU-R BS.1770-4 K-weighting, computed in numpy at 48 kHz. Two measurements are not
K-weighted: speech detection in the voiceover is plain RMS in dBFS, and true peak is unweighted.
Levels are LKFS; differences between levels are LU. The steps check reports its figures in dB.

Files keep their channels: mono stays mono, stereo is measured per channel and summed the way
BS.1770 does, more than two channels are downmixed to two by ffmpeg; only the first audio track is
read. Every file is decoded to 48 kHz.

Check the bed before it is encoded. ffmpeg's built-in AAC at 128k rebuilt exact loops of steady
noise to scores between 0.88 and 0.97, some under the threshold; textured loops stayed at 1.00 after
AAC and MP3.

A 10-minute mono file takes about 2 s for `bed` and about 1.3 GB of memory; an hour takes about
35 s and 3.7 GB. A `warn` line never changes the exit code.

## loop

Envelope: momentary loudness, 400 ms windows every 100 ms, in LKFS. The check works on its
frame-to-frame changes. A copied clip copies its fine texture, so the changes repeat exactly where
the audio repeats, while a fade or a gain change moves the level and barely touches them.

For every lag from 0.5 s up to the file length minus one window, the score is the best Pearson
correlation between a 10 s window of changes and the window that many frames later, over window
starts every 0.5 s. Files under 30 s use a window of a third of their length, never under 3 s. The
best window decides, so a loop is found when only part of the file repeats, under a fade, or at a
different gain on each repeat. Lags resolve in 0.1 s steps.

Two limits on length follow from the window. A repeat is seen only when the repeating stretch is at
least one period plus the window (10 s) long, so a 6 to 8 s clip played twice usually is not: a 6 s
clip twice inside 40 s of other material passes, three times fails. The longest repeat that can be
seen is the file length minus the window, about 10.5 s less than the file (a third less under
30 s): a 54 s clip repeated into a 60 s file is not caught.

The repeat reported is the fundamental: among the lags where the score is a local peak at or above
`--loop-threshold` (0.9), the shortest one within 0.05 of the strongest. A loop correlates at every
multiple of its period at about the same height, so this names the period rather than a multiple
or a half-period peak. If the fundamental is shorter than `--min-period` (6 s) the file passes with
a note, because music repeats at bar length; at or longer, it fails. With no peak at the threshold,
the score shown is the strongest local peak at lags of `--min-period` or more, and the file passes.

Files shorter than twice the minimum period plus a second (13 s at the defaults) are not judged;
they report ok with a note.

Validation, on generated signals. The figures are from one run of `scripts/loop_sweep.py` in the
repository, which prints them. 600 unlooped textured beds of 30 to 600 s: no failures, worst score
0.75; other seeds reach about 0.78, and longer files score higher by chance. Loops under fade-outs
of 0.25 to 30 s, loops in part of the file, crossfaded joins of 0.2 to 5 s, and repeats at random
gains of up to 6 dB each: all caught with the right period in that run, lowest score 0.96. On one
other seed a 10 s loop at gains up to 6 dB apart was named at 20 s (a multiple) and still failed.
Twenty 120 s beds with a 6 dB swell every 8 or 20 seconds all pass, highest 0.85 in that run.

Limitations, measured: a gain cycle with nothing under it looks like a repeat to any envelope
method. A 4 dB tremolo with a 7 s period on steady noise scores 0.95 and fails, and two identical
6 dB swells 30 s apart on steady room tone score 0.97 and fail. A natural bed with a regular cycle
of 6 s or more may fail. The check measures repetition, not where the audio came from.

Background: scoring every lag to find a repeating period is the beat-spectrum idea in Rafii and
Pardo, "REpeating Pattern Extraction Technique (REPET): A Simple Method for Music/Voice Separation",
IEEE Transactions on Audio, Speech, and Language Processing 21(1), 2013.

## steps

Envelope: K-weighted level in half-second blocks every 0.1 s, the first and last `--edge` seconds
(0.75, rounded up to whole 0.1 s frames) dropped so a fade is not read as a jump.

- `step`: at every 0.1 s boundary, the mean of the 2 s of blocks after a 0.5 s transition gap
  against the 2 s of blocks before it; the largest absolute difference. Fails above `--max-step`
  (6 dB). The gap keeps every block on both sides clear of the join, so the reading does not depend
  on where the join falls between blocks, and the time printed names the join to within 0.05 s;
  a join inside the first 2.5 s run-up is reported at the first measurable boundary (3.25 s from
  either end at the default edge).
  A hard join shifts the level and it stays shifted. Blocks under -70 LKFS count as -70 for the
  step, so noise far below anything audible cannot fail a file; a bed coming in out of silence is
  still a step.
- `range`: loudest block minus quietest. Information only.
- `transient`: largest change between two half-second blocks 0.5 s apart. Information only. A
  loud crowd surges and comes back, and blocking on that would flatten exactly the recordings worth
  keeping.
- `peak`: true peak, 4x oversampled, dBTP. A bed near 0 dBTP has no headroom under a mix.
  A file peaking at or under -60 dBTP gets a `warn` line, "the file is silent": probably the wrong
  file or a muted export. It does not change the verdict.

What fails, measured on generated noise:

- A change held about 2 s or longer fails even if it comes back. A +7 dB plateau reads 2.2 dB held
  0.5 s, 3.9 held 1 s, 5.7 held 1.5 s, 6.8 held 2 s and 7.0 held 3 s.
- A +7 dB raised-cosine swell passes: 5.6 dB at 4 s wide, 3.6 at 2 s wide.
- A 10 dB ramp reads 5.0 dB over 5 s and 0.9 over 30 s; both pass. A slow fade is not a join.
- Two +4 dB steps 1 s apart (8 dB louder and staying louder) read about 7 dB and fail.
- A join of 6.1 dB reads 6.05–6.16 and fails.

Needs at least 6.5 seconds of audio at the default edge (45 blocks after the edges are dropped; the
message states the figure for the edge in use). A shorter file reports ok with a note. A join less
than about 2.5 s inside the dropped edge (the 2 s window plus the gap) reads smaller than it is.
Inside the dropped edge it is not seen at all, not even in `range` or `transient`.

## separation

Inputs: the voiceover on its own and the rendered mix.

1. Speech runs in the voiceover: RMS per 20 ms above `--gate` (-44 dBFS) for at least 250 ms.
2. Gaps: the space between runs, trimmed 250 ms at each end, kept if at least 400 ms remains. Gaps
   hold only the bed.
3. Each run and gap is measured in the mix as the 90th percentile of 50 ms K-weighted blocks inside
   it; the percentile so a breath inside a run does not drag the figure down.
4. Separation = mean over runs minus mean over gaps, in LU.

The read needs pauses of 0.9 s or more: a gap is kept only if 0.4 s remains after the 0.25 s
guards, less one 20 ms gate hop because a pause that starts between hops measures a hop short, so a read with no pause that long cannot be checked (exit 2). A room tone or hiss in the
voiceover above `--gate` fills every pause the same way.

Two user errors are caught. If the bed-only windows read -70 LKFS or less in the mix, the mix holds
no bed (usually the voiceover was given as the mix) and the check stops with exit 2. And the
voiceover's loudness envelope inside its speech runs is cross-correlated with the mix's, at 0.1 s
resolution over plus or minus the mix's length; the best match is reported as
`estimated_offset_s`, and when it is more than 0.2 s from `--vo-offset` a warning names it. A read
with regular pauses matches nearly as well at several lags, so the given offset stands when its
match is within 2% of the best. The estimate is trusted only when, at the lag found, the
voiceover's level inside its speech runs and the mix's level at the same moments correlate at 0.5
or more (Pearson); otherwise `estimated_offset_s` is null and there is no warning. A voice 10 dB
under a textured bed read 0.32 or less and matched at lags up to 14 s from the right one; a voice
level with the bed read 0.40 to 0.73, and one 10 dB over it 0.99.

The check compares speech windows with the bed in the gaps between them, so a bed that is ducked
under speech barely changes the figure. Torcoli et al. measured speech against the ducked
background, so treat the floors as approximate for ducked mixes.

This is voice-plus-bed against bed, which is what a listener hears and what WCAG describes, not the
ratio of the two stems. It is also a K-weighted figure, so a voice and a bed with different spectra
are compared by loudness, not by sample gain. A voice with less high-frequency content than the bed
shows less separation than the gain difference applied to it.

Profiles and where their numbers come from:

- `music` 10 LU and `ambience` 15 LU: Torcoli, Freke-Morin, Paulus, Simon and Shirley, "Preferred
  Levels for Background Ducking to Produce Esthetically Pleasing Audio for TV with Clear Speech",
  Journal of the Audio Engineering Society 67(12), 2019, doi:10.17743/jaes.2019.0052. From the
  abstract: "we recommend at least 10 LU difference for CoM [commentary over music] and at least
  15 LU for CoA [commentary over ambience]". The same paper found non-expert listeners preferred
  about 4 LU more than experts, so these are floors.
- `wcag` 20 LU: WCAG 2 technique G56, "Mixing audio files so that non-speech sounds are at least 20
  decibels lower than the speech audio content". G56 states its 20 dB in dB(A) SPL; this tool's LU
  difference is the nearest file-based measure of the same idea, not the same unit.

`--min-separation` overrides. The mix's true peak is printed, with a warning above -1 dBTP.
