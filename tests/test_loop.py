from audio_bed_check.checks import LoopResult, loop_score
from tests.synth import SR, ar1_envelope, noise, tile


def textured(seconds, seed):
    x = noise(seconds, seed)
    return x * ar1_envelope(len(x), seed)


def test_a_tiled_bed_fails_with_its_repeat_length():
    r = loop_score(tile(textured(12, 1), 4), SR)
    assert isinstance(r, LoopResult)
    assert not r.passed
    assert r.score > 0.9
    assert abs(r.period_s - 12.0) < 0.3


def test_a_crossfaded_join_is_still_a_loop():
    r = loop_score(tile(textured(12, 2), 4, crossfade=0.2), SR)
    assert not r.passed
    assert r.score > 0.85
    assert abs(r.period_s - 11.8) < 0.3


def test_an_unrepeated_bed_passes():
    r = loop_score(textured(48, 3), SR)
    assert r.passed
    assert r.score < 0.5
    assert r.notes == ()


def test_a_bar_length_repeat_is_a_note_not_a_failure():
    r = loop_score(tile(textured(2, 4), 20), SR)
    assert r.passed
    assert abs(r.period_s - 2.0) < 0.2
    assert r.score > 0.9
    assert r.notes == ("repeats every 2.0s, under --min-period 6s, not flagged",)


def test_lowering_min_period_flags_the_same_file():
    r = loop_score(tile(textured(2, 4), 20), SR, min_period=1.5)
    assert not r.passed
    assert abs(r.period_s - 2.0) < 0.2


def test_a_short_file_is_reported_not_judged():
    r = loop_score(textured(10, 5), SR)
    assert r.passed
    assert r.period_s is None
    assert r.score == 0.0
    assert r.notes == ("too short to test for a repeat longer than 6s",)


def test_threshold_is_recorded():
    r = loop_score(textured(20, 6), SR, threshold=0.5)
    assert r.threshold == 0.5 and r.min_period_s == 6.0
