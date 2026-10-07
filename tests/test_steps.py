import numpy as np
import pytest

from audio_bed_check.checks import level_steps
from tests.synth import SR, fade_edges, noise, step_gain, swell_gain


def test_a_hard_join_fails_with_its_size_and_time():
    x = noise(60, 1) * step_gain(60 * SR, 30.0, 7.0)
    r = level_steps(x, SR)
    assert not r.passed
    assert abs(r.step_db - 7.0) < 1.0
    assert abs(r.step_at_s - 30.0) < 1.0
    assert r.step_at_s % 0.5 == 0   # boundaries land on block edges
    assert abs(r.range_db - 7.0) < 1.0


def test_a_short_swell_bigger_than_the_join_passes():
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


def test_a_join_inside_the_blind_zone_is_not_seen_but_shows_in_range():
    # the dropped edge plus the 2 s run-up: a step at 2 s cannot be judged, by design, but it is visible
    x = noise(20, 7) * step_gain(20 * SR, 2.0, 7.0)
    r = level_steps(x, SR)
    assert r.passed
    assert r.range_db > 6.0


def test_peak_is_reported_in_dbtp_as_a_plain_float():
    r = level_steps(noise(20, 4, rms_dbfs=-20.0), SR)
    assert -12.0 < r.peak_dbtp < -4.0   # white noise crest factor is roughly 10-13 dB
    assert type(r.peak_dbtp) is float


def test_a_short_file_is_reported_not_judged():
    r = level_steps(noise(4, 5), SR)
    assert r.passed and r.step_at_s is None
    assert r.notes == ("too short to measure level steps (needs at least 6.5s)",)


def test_the_too_short_message_follows_the_edge():
    r = level_steps(noise(6.5, 6), SR, edge=2.0)
    assert r.step_at_s is None
    assert r.notes == ("too short to measure level steps (needs at least 8.5s)",)


def test_negative_edge_is_refused():
    with pytest.raises(ValueError, match="edge must be >= 0"):
        level_steps(noise(10, 8), SR, edge=-0.5)


def test_max_step_is_respected():
    x = noise(60, 6) * step_gain(60 * SR, 30.0, 5.0)
    assert level_steps(x, SR).passed
    assert not level_steps(x, SR, max_step=4.0).passed


def test_a_flat_bed_names_no_place():
    r = level_steps(np.zeros(20 * SR, dtype=np.float32), SR)
    assert r.passed and r.step_db == 0.0
    assert r.step_at_s is None
    assert r.notes == ("no level change anywhere",)
