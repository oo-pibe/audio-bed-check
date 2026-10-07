from audio_bed_check.checks import StepsResult, level_steps
from tests.synth import SR, fade_edges, noise, step_gain, swell_gain


def test_a_hard_join_fails_with_its_size_and_time():
    x = noise(60, 1) * step_gain(60 * SR, 30.0, 7.0)
    r = level_steps(x, SR)
    assert isinstance(r, StepsResult)
    assert not r.passed
    assert abs(r.step_db - 7.0) < 1.0
    assert abs(r.step_at_s - 30.0) < 1.0
    assert abs(r.range_db - 7.0) < 1.0


def test_a_swell_of_the_same_height_passes():
    # A 1 s raised-cosine swell of +9 dB: the half-second transient is large, the 2 s means barely move.
    x = noise(60, 2) * swell_gain(60 * SR, 30.0, 1.0, 9.0)
    r = level_steps(x, SR)
    assert r.passed
    assert r.step_db < 4.0
    assert r.transient_db > 3.0


def test_fades_at_the_edges_are_ignored():
    r = level_steps(fade_edges(noise(60, 3), 1.0), SR)
    assert r.passed
    assert r.step_db < 1.0


def test_peak_is_reported_in_dbtp():
    r = level_steps(noise(20, 4, rms_dbfs=-20.0), SR)
    assert -12.0 < r.peak_dbtp < -4.0   # white noise crest factor is roughly 10-13 dB


def test_a_short_file_is_reported_not_judged():
    r = level_steps(noise(4, 5), SR)
    assert r.passed and r.step_at_s is None
    assert r.notes == ("too short to measure level steps (needs about 6s)",)


def test_max_step_is_respected():
    x = noise(60, 6) * step_gain(60 * SR, 30.0, 5.0)
    assert level_steps(x, SR).passed
    assert not level_steps(x, SR, max_step=4.0).passed
