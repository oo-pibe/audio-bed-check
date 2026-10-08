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


def test_true_peak_matches_a_direct_zero_stuffed_reference():
    # the polyphase split is an identity: four short convolutions of the original samples give exactly
    # the samples of one long convolution of the 4x zero-stuffed signal with the whole kernel
    from audio_bed_check.loudness import _UP, _interp_kernel
    x = sine(5, 1000, 0.9)
    assert abs(true_peak(x, SR) + 0.915) < 0.05     # a 0.9 sine is -0.915 dBTP
    for signal in (x, noise(3, 11), fade(sine(1, SR / 4, 0.5, phase=np.pi / 4))):
        up = np.zeros(len(signal) * _UP)
        up[::_UP] = signal
        reference = 20 * np.log10(np.abs(fft_convolve(up, _interp_kernel(), full=True)).max())
        assert abs(true_peak(signal, SR) - reference) < 1e-9


def test_true_peak_takes_no_block_size():
    import inspect
    assert list(inspect.signature(true_peak).parameters) == ["samples", "sr"]


def test_k_weight_keeps_length():
    assert len(k_weight(noise(3), SR)) == 3 * SR


def test_fft_convolve_matches_direct_convolution_across_blocks():
    rng = np.random.default_rng(0)
    x, h = rng.standard_normal(10_000), rng.standard_normal(300)
    ref = np.convolve(x, h)
    assert np.allclose(fft_convolve(x, h, block=1024, full=True), ref, atol=1e-9)
    assert np.allclose(fft_convolve(x, h, block=1024), ref[: len(x)], atol=1e-9)


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1)


def test_mono_duplicated_to_stereo_reads_3_01_lkfs_louder():
    # BS.1770 sums the channels' mean squares with weight 1.0 for L and R: two equal channels are +3.01
    x = noise(5, 1)
    diff = momentary(stereo(x, x), SR) - momentary(x, SR)
    assert np.all(np.abs(diff - 10 * np.log10(2)) < 0.02)


def test_an_anti_phase_pair_is_not_silence():
    # a mono downmix cancels L = -R to nothing; a per-channel sum reads it like any other stereo pair
    x = noise(5, 2)
    diff = momentary(stereo(x, -x), SR) - momentary(x, SR)
    assert np.all(np.abs(diff - 10 * np.log10(2)) < 0.02)


def test_block_loudness_accepts_two_channels():
    x = k_weight(noise(3, 3), SR)
    both = block_loudness(stereo(x, x), SR, 0.5, 0.5)
    assert np.allclose(both - block_loudness(x, SR, 0.5, 0.5), 10 * np.log10(2), atol=1e-9)


def test_k_weight_filters_each_channel():
    x, y = noise(2, 4), noise(2, 5)
    w = k_weight(stereo(x, y), SR)
    assert w.shape == (2 * SR, 2)
    assert np.allclose(w[:, 0], k_weight(x, SR)) and np.allclose(w[:, 1], k_weight(y, SR))


def test_true_peak_is_the_loudest_channel():
    # a -0.26 dBFS tone on one channel only; a mono downmix read it 6 dB low and missed the warning
    left = fade(sine(2, 1000, 10 ** (-0.26 / 20)))
    assert abs(true_peak(stereo(left, np.zeros_like(left)), SR) + 0.26) < 0.05
    assert abs(true_peak(stereo(np.zeros_like(left), left), SR) + 0.26) < 0.05


@pytest.mark.parametrize("fn", [
    lambda x: k_weight(x, SR), lambda x: momentary(x, SR), lambda x: true_peak(x, SR),
    lambda x: block_loudness(x, SR, 0.5, 0.5),
])
def test_bad_sample_arrays_are_refused(fn):
    with pytest.raises(ValueError, match="^samples must be finite floats$"):
        fn(np.zeros(SR, dtype=np.int16))
    with pytest.raises(ValueError, match="^samples must be finite floats$"):
        bad = np.zeros(SR, dtype=np.float32)
        bad[100] = np.nan
        fn(bad)
    with pytest.raises(ValueError, match="^samples must be finite floats$"):
        fn(np.full((SR, 2), np.inf))
    with pytest.raises(ValueError, match="^pass mono or stereo samples$"):
        fn(np.zeros((SR, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="^pass mono or stereo samples$"):
        fn(np.zeros((2, SR, 1), dtype=np.float32))
