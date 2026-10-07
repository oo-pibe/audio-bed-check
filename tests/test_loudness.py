import numpy as np
import pytest

from audio_bed_check.loudness import HOP, block_loudness, fft_convolve, k_weight, momentary, true_peak
from tests.synth import SR, fade, noise, sine


def test_997hz_full_scale_sine_reads_minus_3_lkfs():
    # BS.1770-4: a 0 dBFS 997 Hz sine is -3.01 LKFS; the -0.691 offset exists to make that true.
    m = momentary(sine(5, 997, 1.0), SR)
    assert abs(m[5:].mean() + 3.01) < 0.1


def test_minus_20_dbfs_sine_reads_minus_23_lkfs():
    m = momentary(sine(5, 997, 0.1), SR)
    assert abs(m[5:].mean() + 23.01) < 0.1


def test_dc_is_removed_by_the_high_pass():
    m = momentary(np.full(SR * 3, 0.5, dtype=np.float32), SR)
    assert m[10:].max() < -60


def test_momentary_frame_count_and_hop():
    m = momentary(noise(10), SR)
    assert HOP == 0.1
    # 97 frames; integer arithmetic so there is no float drift
    assert len(m) == (10 * SR - int(0.4 * SR)) // int(0.1 * SR) + 1


def test_block_loudness_floor_is_minus_100():
    b = block_loudness(np.zeros(SR), SR, 0.5, 0.5)
    assert np.all(b == -100.0)


def test_other_sample_rates_are_refused():
    with pytest.raises(ValueError, match="resample to 48 kHz"):
        momentary(noise(1), 44100)
    with pytest.raises(ValueError, match="resample to 48 kHz"):
        block_loudness(np.zeros(100), 44100, 0.5, 0.5)
    with pytest.raises(ValueError, match="resample to 48 kHz"):
        true_peak(np.zeros(100), 44100)


def test_true_peak_of_a_half_scale_sine():
    assert abs(true_peak(fade(sine(2, 1000, 0.5)), SR) + 6.02) < 0.01


def test_true_peak_sees_the_inter_sample_peak():
    # fs/4 with a 45 degree phase: every sample is 0.3536 (-9.03 dBFS) but the waveform peaks at 0.5 (-6.02).
    x = fade(sine(2, SR / 4, 0.5, phase=np.pi / 4))
    assert abs(float(np.abs(x).max()) - 0.3536) < 0.01
    assert abs(true_peak(x, SR) + 6.02) < 0.01


def test_true_peak_of_silence():
    assert true_peak(np.zeros(SR), SR) == -99.0


def test_true_peak_does_not_depend_on_block_size():
    # a 0.9 sine is -0.915 dBTP; a chunked method must give the same number whatever the chunk
    x = sine(5, 1000, 0.9)
    assert abs(true_peak(x, SR) + 0.915) < 0.05
    assert abs(true_peak(x, SR) - true_peak(x, SR, block=4096)) < 1e-9


def test_k_weight_keeps_length():
    assert len(k_weight(noise(3), SR)) == 3 * SR


def test_fft_convolve_matches_direct_convolution_across_blocks():
    rng = np.random.default_rng(0)
    x, h = rng.standard_normal(10_000), rng.standard_normal(300)
    ref = np.convolve(x, h)
    assert np.allclose(fft_convolve(x, h, block=1024, full=True), ref, atol=1e-9)
    assert np.allclose(fft_convolve(x, h, block=1024), ref[: len(x)], atol=1e-9)
