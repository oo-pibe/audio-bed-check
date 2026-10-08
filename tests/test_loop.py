import numpy as np
import pytest

from audio_bed_check.checks import LoopResult, level_steps, loop_score
from audio_bed_check.loudness import k_weight
from tests.synth import SR, ar1_envelope, fade_edges, gain, noise, step_gain, swell_gain, textured, tile


def _fade_out(x, seconds, power=1):
    y = x.astype(np.float64).copy()
    k = int(seconds * SR)
    y[-k:] *= np.linspace(1, 0, k) ** power
    return y.astype(np.float32)


def test_a_tiled_bed_fails_with_its_repeat_length():
    r = loop_score(tile(textured(12, 1), 4), SR)
    assert isinstance(r, LoopResult)
    assert not r.passed
    assert r.score > 0.99
    assert abs(r.period_s - 12.0) < 0.15


@pytest.mark.parametrize("crossfade", [0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_a_crossfaded_join_is_still_a_loop(crossfade):
    r = loop_score(tile(textured(12, 2), 4, crossfade=crossfade), SR)
    assert not r.passed
    assert r.score > 0.95
    assert abs(r.period_s - (12.0 - crossfade)) < 0.15   # the crossfade shortens the period


LOOP = tile(textured(10, 3), 6)


@pytest.mark.parametrize("name, faded", [
    ("0.25 s fade-out", _fade_out(LOOP, 0.25)),
    ("0.5 s fade-out", _fade_out(LOOP, 0.5)),
    ("1 s fade-out", _fade_out(LOOP, 1.0)),
    ("2 s fade-out", _fade_out(LOOP, 2.0)),
    ("5 s fade-out", _fade_out(LOOP, 5.0)),
    ("1 s fade in and out", fade_edges(LOOP, 1.0)),
    ("whole-file fade", _fade_out(LOOP, 60.0)),
    ("30 s quadratic fade-out", _fade_out(LOOP, 30.0, power=2)),
])
def test_a_faded_loop_is_still_caught(name, faded):
    # a silent tail used to dominate the whole-file correlation; the windows inside the loop still match
    r = loop_score(faded, SR)
    assert not r.passed, name
    assert abs(r.period_s - 10.0) < 0.15, name


@pytest.mark.parametrize("name, x", [
    ("four tiles, then 20 s unrelated", np.concatenate([tile(textured(10, 3), 4), textured(20, 99)])),
    ("20 s unrelated, then four tiles", np.concatenate([textured(20, 98), tile(textured(10, 3), 4)])),
    ("60 s unrelated, then two tiles", np.concatenate([textured(60, 97), tile(textured(10, 3), 2)])),
])
def test_a_loop_in_part_of_the_file_is_caught(name, x):
    r = loop_score(x, SR)
    assert not r.passed, name
    assert abs(r.period_s - 10.0) < 0.15, name


@pytest.mark.parametrize("spread, seed", [(3, 0), (3, 1), (6, 0), (6, 1)])
def test_repeats_at_different_gains_name_the_right_period(spread, seed):
    rng = np.random.default_rng(100 + seed)
    x = np.concatenate([gain(textured(10, 3), rng.uniform(-spread, spread)) for _ in range(6)])
    r = loop_score(x, SR)
    assert not r.passed
    assert abs(r.period_s - 10.0) < 0.15   # not 20 or 50


def test_a_clip_repeated_once_to_fill_the_file_is_caught():
    r = loop_score(tile(textured(35, 11), 2)[: 60 * SR], SR)
    assert not r.passed
    assert abs(r.period_s - 35.0) < 0.15
    assert r.notes == ()


def test_a_repeat_longer_than_the_file_minus_a_window_is_not_seen():
    # documented limit: the longest lag tested is the file length minus a 10 s window
    r = loop_score(tile(textured(54, 13), 2)[: 60 * SR], SR)
    assert r.passed and r.period_s is None


@pytest.mark.parametrize("seconds, seed", [(30, 1), (30, 2), (48, 3), (60, 4), (60, 12), (180, 5)])
def test_an_unrepeated_bed_passes(seconds, seed):
    r = loop_score(textured(seconds, seed), SR)
    assert r.passed
    # one run of scripts/loop_sweep.py, 600 beds of 30-600 s: worst 0.75; other seeds reach about 0.78
    assert r.score < 0.8
    assert r.period_s is None
    assert r.notes == ()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_periodic_swells_on_a_textured_bed_are_not_a_loop(seed):
    bed = textured(120, 50 + seed)
    every_20 = np.prod([swell_gain(len(bed), c, 4, 6) for c in np.arange(10, 120, 20)], axis=0)
    every_8 = np.prod([swell_gain(len(bed), c, 3, 6) for c in np.arange(5, 120, 8)], axis=0)
    for g in (every_20, every_8):
        r = loop_score((bed * g).astype(np.float32), SR)
        assert r.passed and r.period_s is None


def test_a_tremolo_on_steady_noise_reads_as_a_loop():
    # documented limitation: a gain cycle with no texture under it looks like a repeat to any
    # envelope method
    t = np.arange(60 * SR) / SR
    r = loop_score((noise(60, 0) * 10 ** (2 * np.sin(2 * np.pi * t / 7) / 20)).astype(np.float32), SR)
    assert not r.passed
    assert abs(r.period_s - 7.0) < 0.3


def test_two_identical_swells_on_room_tone_read_as_a_loop():
    # documented limitation: steady room tone has almost no texture, so the same swell twice
    # correlates like a copy
    n = 50 * SR
    r = loop_score(noise(50, 70) * swell_gain(n, 8, 4, 6) * swell_gain(n, 38, 4, 6), SR)
    assert not r.passed
    assert abs(r.period_s - 30.0) < 0.15


def test_a_half_period_peak_does_not_name_the_repeat():
    # seed 43's envelope also correlates at 1.0 s; the fundamental must still be the 2 s tile
    r = loop_score(tile(textured(2, 43), 20), SR, min_period=1.5)
    assert not r.passed
    assert abs(r.period_s - 2.0) < 0.15


def test_a_bar_length_repeat_is_a_note_not_a_failure():
    r = loop_score(tile(textured(2, 4), 20), SR)
    assert r.passed
    assert abs(r.period_s - 2.0) < 0.15
    assert r.score > 0.9
    assert r.notes == ("repeats every 2.0s, under --min-period 6s, not flagged",)


def test_lowering_min_period_flags_the_same_file():
    r = loop_score(tile(textured(2, 4), 20), SR, min_period=1.5)
    assert not r.passed
    assert abs(r.period_s - 2.0) < 0.15


def test_a_short_file_is_reported_not_judged():
    r = loop_score(textured(10, 5), SR)
    assert r.passed
    assert r.period_s is None
    assert r.score == 0.0
    assert r.notes == ("too short to test for a repeat longer than 6s",)


def test_a_13s_file_is_just_long_enough_to_judge():
    r = loop_score(textured(13, 5), SR)
    assert r.passed and r.period_s is None and r.notes == ()
    assert 0.0 < r.score < 0.8


def test_threshold_is_recorded():
    r = loop_score(textured(20, 6), SR, threshold=0.5)
    assert r.threshold == 0.5 and r.min_period_s == 6.0


def test_the_default_threshold_is_0_9():
    assert loop_score(textured(20, 6), SR).threshold == 0.9


def test_a_flat_bed_passes_with_a_note():
    r = loop_score(np.zeros(20 * SR, dtype=np.float32), SR)
    assert r.passed and r.period_s is None
    assert r.notes == ("level is constant; nothing to correlate",)


def test_bad_arguments_raise_even_on_a_short_file():
    short = np.zeros(SR, dtype=np.float32)
    with pytest.raises(ValueError, match="min_period must be positive"):
        loop_score(short, SR, min_period=0)
    with pytest.raises(ValueError, match="threshold must be"):
        loop_score(short, SR, threshold=1.5)


def test_a_steady_looped_bed_is_still_caught():
    # room tone, crowd hum: the envelope moves by tenths of a dB, and a loop of it must still fail
    r = loop_score(tile(noise(12, 7, rms_dbfs=-30.0), 4), SR)
    assert not r.passed
    assert abs(r.period_s - 12.0) < 0.15


def test_a_step_halfway_is_not_read_as_a_repeat():
    r = loop_score(noise(60, 3, rms_dbfs=-30.0) * step_gain(60 * SR, 30.0, 7.0), SR)
    assert r.passed and r.period_s is None


def test_stereo_reads_the_same_loop_as_mono():
    x = tile(textured(12, 1), 4)
    mono, both = loop_score(x, SR), loop_score(np.stack([x, x], axis=1), SR)
    assert not both.passed
    assert abs(both.score - mono.score) < 0.01
    assert both.period_s == mono.period_s


def test_a_precomputed_k_weighted_signal_gives_the_same_results():
    x = tile(textured(12, 1), 4)
    w = k_weight(x, SR)
    assert loop_score(x, SR, weighted=w) == loop_score(x, SR)
    assert level_steps(x, SR, weighted=w) == level_steps(x, SR)


def test_a_weighted_signal_of_the_wrong_shape_raises():
    x = textured(20, 6)
    with pytest.raises(ValueError, match="weighted must be"):
        loop_score(x, SR, weighted=k_weight(x[:SR], SR))
    with pytest.raises(ValueError, match="weighted must be"):
        level_steps(x, SR, weighted=k_weight(x[:SR], SR))


def test_lag_scores_match_a_naive_pearson_loop():
    # oracle: every lag, every window start, np.corrcoef directly
    from audio_bed_check.checks import LOOP_STRIDE, _lag_scores

    d = np.random.default_rng(0).standard_normal(400)
    window = 100
    got = _lag_scores(d, window, 1, len(d) - window)
    for i, k in enumerate(range(1, len(d) - window + 1)):
        want = max(np.corrcoef(d[s:s + window], d[s + k:s + k + window])[0, 1]
                   for s in range(0, len(d) - k - window + 1, LOOP_STRIDE))
        assert abs(got[i] - want) < 1e-10, k


def test_with_no_repeat_the_score_is_not_the_value_at_min_period():
    # a slowly wandering envelope (2 s knots): the correlation of its changes falls away from the
    # shortest lags, so the value at min_period is high and the reported score, the strongest
    # local maximum at or past it, is well below
    from audio_bed_check.checks import _lag_scores
    from audio_bed_check.loudness import momentary

    x = (noise(30, 9, rms_dbfs=-30.0) * ar1_envelope(30 * SR, 9, depth_db=20, knot_s=2.0)).astype(np.float32)
    env = momentary(x, SR)
    at_min_period = _lag_scores(np.diff(env), len(env) // 3, 6, 6)[0]   # the window loop_score uses
    r = loop_score(x, SR, min_period=0.6)
    assert r.passed and r.period_s is None
    assert r.score < at_min_period - 0.1


def test_a_burst_then_silence_scores_zero_not_minus_one():
    # every window in the silent tail has no variance and scores -1; that plateau is not a peak
    x = np.concatenate([textured(2, 1), np.zeros(18 * SR, dtype=np.float32)])
    r = loop_score(x, SR)
    assert r.passed and r.period_s is None
    assert r.score == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_short_clip_twice_is_not_seen_three_times_is(seed):
    # documented gap: a repeat is seen only when the repeating stretch is at least one period plus
    # the 10 s window long. A 6 s clip twice is 12 s of repeat and passes; three times is 18 s.
    clip = textured(6, 20 + seed)
    twice, thrice = (np.concatenate([textured(20, 40 + seed), tile(clip, n), textured(20, 60 + seed)])
                     for n in (2, 3))
    assert loop_score(twice, SR).passed
    r = loop_score(thrice, SR)
    assert not r.passed
    assert abs(r.period_s - 6.0) < 0.15
