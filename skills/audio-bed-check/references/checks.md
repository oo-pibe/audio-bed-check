# The three checks

All measurements use ITU-R BS.1770-4 K-weighting, computed in numpy at 48 kHz (files are decoded to
48 kHz mono by ffmpeg). Levels are LKFS; differences between levels are LU.

## loop

Envelope: momentary loudness, 400 ms windows every 100 ms, linear trend removed. Autocorrelation by
FFT, unbiased and normalised so an exact repeat scores 1.0.

The repeat reported is the shortest lag of 0.5 s or more where the autocorrelation peaks at or above
`--loop-threshold` (0.75). If that lag is under `--min-period` (6 s) the file passes with a note:
music repeats at bar length. If it is at or above, the file fails. With no peak above the threshold,
the score shown is the best value in the flaggable range and the file passes.

Files shorter than twice the minimum period plus a second are not judged.

Calibration, on 36 stadium ambience beds from a production reel pipeline (not included in this
repository): looped files scored 0.79 to 1.00 with the lag equal to the repeat length; unlooped
files 0.11 to 0.43; one concourse recording with its own regular rhythm scored 0.605 and was missed
at 0.75; a composed music bed scored 0.82 at its bar length, which the minimum period now keeps out.

Method: the beat-spectrum idea in Rafii and Pardo, "REpeating Pattern Extraction Technique (REPET):
A Simple Method for Music/Voice Separation", IEEE Transactions on Audio, Speech, and Language
Processing 21(1), 2013.

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

Needs at least 6.5 seconds of audio (nine half-second blocks after the edges are dropped). It
cannot see a step within about 2.75 s of either end: the dropped edge plus the two-second run-up.
A join that close to an edge still shows in `range` and `transient`.

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
measure by their loudness, not their sample gain (a band-limited voice placed +10 dB over white
noise in the sample domain measured about 8.3 LU, because K-weighting favours the brighter bed).

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
