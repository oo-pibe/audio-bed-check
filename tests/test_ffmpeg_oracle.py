"""Our BS.1770 code against ffmpeg's ebur128 filter on the same file. Skipped without ffmpeg."""

import re
import subprocess

import numpy as np
import pytest

from audio_bed_check.decode import DecodeError, find_ffmpeg
from audio_bed_check.loudness import HOP, WINDOW, momentary, true_peak
from tests.synth import SR, fade, noise, read_wav, sine, write_wav


def _ffmpeg_with_ebur128() -> str | None:
    """The ffmpeg to use as the oracle, or None: stripped builds ship without the ebur128 filter."""
    try:
        exe = find_ffmpeg()
    except DecodeError:
        return None
    filters = subprocess.run(
        [exe, "-hide_banner", "-filters"], capture_output=True, text=True
    ).stdout
    return exe if "ebur128" in filters else None


FFMPEG = _ffmpeg_with_ebur128()

pytestmark = pytest.mark.skipif(
    FFMPEG is None, reason="ffmpeg with the ebur128 filter not available"
)


def ebur128(path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(frame times, momentary readings, integrated LUFS, true peak dBFS) parsed from ffmpeg's log."""
    proc = subprocess.run(
        [FFMPEG, "-nostdin", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    log = proc.stderr
    assert proc.returncode == 0, log[-2000:]
    frames = re.findall(r"t:\s*([\d.]+)\s+TARGET:.*?M:\s*(-?[\d.]+)", log)
    times = np.array([float(t) for t, _ in frames])
    m = np.array([float(v) for _, v in frames])
    integrated = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])
    peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", log).group(1))
    return times, m, integrated, peak


@pytest.fixture(scope="module")
def tone_file(tmp_path_factory):
    # white noise under a 6 dB peak-to-peak sinusoidal modulation at 1 Hz: a one-frame (0.1 s) offset moves
    # the momentary reading by about 1 dB (median), far outside the bound, so the comparison checks timing too
    x = noise(20, 7, rms_dbfs=-20.0)
    t = np.arange(len(x)) / SR
    x = x * 10 ** (np.sin(2 * np.pi * 1.0 * t) * 3.0 / 20)
    path = tmp_path_factory.mktemp("oracle") / "tone.wav"
    write_wav(path, x)
    return path


def test_momentary_matches_ffmpeg(tone_file):
    times, theirs, _, _ = ebur128(tone_file)
    assert len(theirs) > 100
    samples, sr = read_wav(tone_file)
    ours = momentary(samples, sr)
    # ffmpeg stamps a momentary reading at the END of its 400 ms window
    idx = np.round((times - WINDOW) / HOP).astype(int)
    keep = (idx >= 0) & (idx < len(ours)) & (theirs > -70)
    diff = np.abs(ours[idx[keep]] - theirs[keep])
    assert np.median(diff) < 0.1  # aligned, the difference is ffmpeg's one-decimal print rounding
    assert np.percentile(diff, 95) < 0.2


def test_integrated_matches_ffmpeg_on_a_stationary_signal(tone_file):
    _, _, integrated, _ = ebur128(tone_file)
    samples, sr = read_wav(tone_file)
    ours = momentary(samples, sr)
    mean_square = np.mean(10 ** ((ours + 0.691) / 10))
    assert abs((-0.691 + 10 * np.log10(mean_square)) - integrated) < 0.5


@pytest.fixture(scope="module")
def peak_file(tmp_path_factory):
    # fs/4 sine at 45 degrees: every sample sits 3.01 dB under the true peak of 0.5 (-6.02 dBTP), so a
    # true_peak that returned the sample peak would miss by 3 dB; wideband noise cannot tell the two apart
    path = tmp_path_factory.mktemp("oracle") / "peak.wav"
    write_wav(path, fade(sine(5, 12000, 0.5, phase=np.pi / 4)))
    return path


def test_true_peak_matches_ffmpeg(peak_file):
    _, _, _, peak = ebur128(peak_file)
    samples, sr = read_wav(peak_file)
    assert abs(true_peak(samples, sr) - peak) < 0.2
    # the oracle itself sees the inter-sample peak
    assert peak > 20 * np.log10(np.abs(samples).max()) + 2.5
