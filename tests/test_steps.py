import numpy as np
import pytest

from audio_bed_check.checks import level_steps
from tests.synth import SR, fade_edges, noise, step_gain, swell_gain


def test_a_hard_join_fails_with_its_size_and_time():
    x = noise(60, 1) * step_gain(60 * SR, 30.0, 7.0)
    r = level_steps(x, SR)
    assert not r.passed
    assert abs(r.step_db - 7.0) < 1.0
    assert abs(r.step_at_s - 30.0) <= 0.1   # names the join, not the start of a block
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
    assert level_steps(noise(6.5, 5), SR).step_at_s is not None   # 6.5 s is exactly enough


def test_the_too_short_message_follows_the_edge():
    r = level_steps(noise(6.5, 6), SR, edge=2.0)
    assert r.step_at_s is None
    assert r.notes == ("too short to measure level steps (needs at least 8.9s)",)


@pytest.mark.parametrize("at", [30.0, 30.1, 30.25, 30.3])
def test_a_step_just_over_the_line_fails_wherever_it_falls_between_blocks(at):
    # half-second blocks on a 0.5 s grid read a 6.1 dB join 0.25 s past a block edge as 5.5 dB
    x = noise(60, 11) * step_gain(60 * SR, at, 6.1)
    r = level_steps(x, SR)
    assert not r.passed
    assert abs(r.step_at_s - at) <= 0.1


def test_two_steps_a_second_apart_read_as_one_lurch():
    # +4 dB then +4 dB again a second later: 8 dB louder and it stays louder. The 2 s after the gap still
    # holds half a second of the middle level, so it reads about 6.9, not 8; on the old grid it read 6.0
    x = noise(60, 12) * step_gain(60 * SR, 30.0, 4.0) * step_gain(60 * SR, 31.0, 4.0)
    r = level_steps(x, SR)
    assert not r.passed
    assert 6.5 < r.step_db < 8.0
    assert type(r.step_at_s) is float


def test_a_plateau_held_two_seconds_fails_even_though_it_comes_back():
    def plateau(seconds):
        return noise(60, 13) * step_gain(60 * SR, 30.0, 7.0) * step_gain(60 * SR, 30.0 + seconds, -7.0)
    assert level_steps(plateau(1.0), SR).passed
    assert not level_steps(plateau(2.0), SR).passed


def test_the_transient_compares_blocks_half_a_second_apart():
    x = noise(60, 14) * step_gain(60 * SR, 30.0, 7.0)
    assert abs(level_steps(x, SR).transient_db - 7.0) < 0.5


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
    assert r.notes == ("no level change anywhere", "the file is silent (peak at or under -60 dBTP)")


def test_a_silent_file_says_so():
    quiet = noise(20, 19, rms_dbfs=-75.0)   # peaks near -63 dBTP; every block is under the -70 floor
    silent = "the file is silent (peak at or under -60 dBTP)"
    assert level_steps(quiet, SR).notes == ("every block is under -70 LKFS; nothing to measure", silent)
    short = level_steps(noise(4, 19, rms_dbfs=-75.0), SR)
    assert short.notes[-1] == silent
    assert level_steps(noise(20, 19, rms_dbfs=-65.0), SR).notes == ()


def test_stereo_reads_the_same_step_as_mono():
    # identical channels add 3.01 dB to every block; a step is a difference, so it does not move
    x = noise(60, 1) * step_gain(60 * SR, 30.0, 7.0)
    mono, both = level_steps(x, SR), level_steps(np.stack([x, x], axis=1), SR)
    assert abs(both.step_db - mono.step_db) < 0.1
    assert both.step_at_s == mono.step_at_s
    assert abs(both.range_db - mono.range_db) < 0.1
    assert abs(both.peak_dbtp - mono.peak_dbtp) < 0.1


def test_a_quiet_bed_still_fails_on_a_step():
    x = noise(60, 17, rms_dbfs=-60.0) * step_gain(60 * SR, 30.0, 7.0)
    assert not level_steps(x, SR).passed


def test_blocks_under_minus_70_lkfs_count_as_silence_for_the_step():
    # without the floor a +7 dB change at -90 dBFS failed: noise far under any monitor cannot lurch
    x = noise(60, 17, rms_dbfs=-90.0) * step_gain(60 * SR, 30.0, 7.0)
    r = level_steps(x, SR)
    assert r.passed and r.step_db == 0.0
    assert r.range_db > 6.0   # range and transient still read the real blocks


def test_a_bed_that_starts_after_silence_is_a_step():
    # the floor caps silence at -70 for the step; a bed coming in out of it is still a lurch
    x = noise(60, 18).copy()
    x[: 30 * SR] = 0.0
    r = level_steps(x, SR)
    assert not r.passed and abs(r.step_at_s - 30.0) <= 0.1
