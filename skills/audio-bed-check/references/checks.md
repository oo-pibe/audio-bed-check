# The three checks

All measurements use ITU-R BS.1770-4 K-weighting, computed in numpy at 48 kHz (files are decoded to
48 kHz by ffmpeg, mono or stereo as the file is; more than two channels are downmixed to two). Stereo
is measured per channel and the channels' powers summed, as BS.1770 does, never downmixed to mono. Levels are LKFS; differences between levels are LU.
The steps check reports its figures in dB.

## loop

Envelope: momentary loudness, 400 ms windows every 100 ms, in LKFS. The check works on its
frame-to-frame changes. A copied clip copies its fine texture, so the changes repeat exactly where
the audio repeats, while a fade or a gain change moves the level and barely touches them.

For every lag from 0.5 s up to the file length minus one window, the score is the best Pearson
correlation between a 10 s window of changes and the window that many frames later, over window
starts every 0.5 s. Files under 30 s use a window of a third of their length, never under 3 s. The
best window decides, so a loop is found when only part of the file repeats, under a fade, or at a
different gain on each repeat. Lags resolve in 0.1 s steps. The longest repeat that can be seen is
the file length minus the window: a 54 s clip repeated into a 60 s file is not caught.

The repeat reported is the fundamental: among the lags where the score is a local peak at or above
`--loop-threshold` (0.9), the shortest one within 0.05 of the strongest. A loop correlates at every
multiple of its period at about the same height, so this names the period rather than a multiple
or a half-period peak. If the fundamental is shorter than `--min-period` (6 s) the file passes with
a note, because music repeats at bar length; at or longer, it fails. With no peak at the threshold,
the score shown is the strongest local peak at lags of `--min-period` or more, and the file passes.

Files shorter than twice the minimum period plus a second are not judged.

Validation, on generated signals (the method was rebuilt before release; the earlier calibration
figures no longer describe it). 600 unlooped textured beds of 30 to 600 s: no failures, worst score
0.75, so 0.15 under the threshold. Loops under fade-outs of 0.25 to 30 s, loops in part of the file,
crossfaded joins of 0.2 to 5 s, and repeats at random gains of up to 6 dB each: all caught with the
right period, lowest score 0.97. A bed with a 6 dB swell every 8 or 20 seconds passes (highest
0.83).

Limitations, measured: a gain cycle with nothing under it looks like a repeat to any envelope
method. A 4 dB tremolo with a 7 s period on steady noise scores 0.95 and fails, and two identical
6 dB swells 30 s apart on steady room tone score 0.97 and fail. A natural bed with a regular cycle
of 6 s or more may fail. The check measures repetition, not where the audio came from.

Background: scoring every lag to find a repeating period is the beat-spectrum idea in Rafii and
Pardo, "REpeating Pattern Extraction Technique (REPET): A Simple Method for Music/Voice Separation",
IEEE Transactions on Audio, Speech, and Language Processing 21(1), 2013.

## steps

Envelope: K-weighted level in non-overlapping half-second blocks, the first and last `--edge`
seconds (0.75, rounded up to whole blocks) dropped so a fade is not read as a jump.

- `step`: the largest absolute difference between the mean of the four blocks after a boundary and
  the four before. Fails above `--max-step` (6 dB). A hard join shifts the level and it stays
  shifted.
- `range`: loudest block minus quietest. Information only.
- `transient`: largest change between adjacent blocks. Information only. A loud crowd surges and
  comes back, and blocking on that would flatten exactly the recordings worth keeping.
- `peak`: true peak, 4x oversampled, dBTP. A bed near 0 dBTP has no headroom under a mix.

Needs at least 6.5 seconds of audio at the default edge (nine half-second blocks after the edges are
dropped; the message states the figure for the edge in use). A join less than 3 s from either end
(the edge rounded up to 1 s, plus the 2 s run-up) reads smaller than it is, about half its height at
2 s from the end. Inside the dropped edge it is not seen at all, not even in `range` or `transient`.

## separation

Inputs: the voiceover on its own and the rendered mix.

1. Speech runs in the voiceover: RMS per 20 ms above `--gate` (-44 dBFS) for at least 250 ms.
2. Gaps: the space between runs, trimmed 250 ms at each end, kept if at least 400 ms remains. Gaps
   hold only the bed.
3. Each run and gap is measured in the mix as the 90th percentile of 50 ms K-weighted blocks inside
   it; the percentile so a breath inside a run does not drag the figure down.
4. Separation = mean over runs minus mean over gaps, in LU.

This is voice-plus-bed against bed, which is what a listener hears and what WCAG describes, not the
ratio of the two stems. It is also a K-weighted figure: a voice and a bed with different spectra
measure by their loudness, not their sample gain: a voice with less high-frequency content than the
bed shows less separation than the gain difference applied to it.

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
