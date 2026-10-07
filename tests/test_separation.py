import numpy as np
import pytest

from audio_bed_check.checks import NoSpeechError, SeparationResult, gaps_between, separation, speech_runs
from tests.synth import SR, gain, gated, noise

RUNS = [(1.0, 4.0), (6.0, 9.0), (11.0, 14.0), (16.0, 19.0), (21.0, 24.0)]


def mix_at(vo_over_bed_db: float, seed: int = 0):
    # -40 dBFS bed: even +20 dB of voice stays under -1 dBTP, so no fixture trips the peak warning by accident
    bed = noise(30, seed, rms_dbfs=-40.0)
    vo = gated(gain(noise(30, seed + 100, rms_dbfs=-40.0), vo_over_bed_db), RUNS)
    return vo, (bed + vo).astype(np.float32)


def expected(vo_over_bed_db: float) -> float:
    # speech windows hold bed + voice; the listener hears 10*log10(1 + 10**(g/10)) above the bed-only gaps
    return 10 * np.log10(1 + 10 ** (vo_over_bed_db / 10))


@pytest.mark.parametrize("g", [6.0, 10.0, 20.0])
def test_separation_matches_the_applied_gain(g):
    vo, mix = mix_at(g)
    r = separation(vo, mix, SR)
    assert isinstance(r, SeparationResult)
    assert abs(r.separation_lu - expected(g)) < 1.0
    assert r.runs == 5 and r.gaps == 4
    assert r.warnings == ()


def test_profiles_decide_pass_fail():
    vo, mix = mix_at(6.0)
    assert not separation(vo, mix, SR, min_lu=10.0).passed
    vo, mix = mix_at(20.0)   # reads about 20.0 LU, so test the decision 2 LU either side, not on the line
    assert separation(vo, mix, SR, min_lu=18.0).passed
    assert not separation(vo, mix, SR, min_lu=22.0).passed


def test_speech_runs_and_gaps():
    vo, _ = mix_at(10.0)
    runs = speech_runs(vo, SR)
    assert len(runs) == 5
    assert all(abs(a - ea) < 0.05 and abs(b - eb) < 0.05 for (a, b), (ea, eb) in zip(runs, RUNS, strict=True))
    flat = [v for gap in gaps_between(runs) for v in gap]
    assert flat == pytest.approx([4.25, 5.75, 9.25, 10.75, 14.25, 15.75, 19.25, 20.75], abs=0.05)


def test_vo_offset_shifts_the_windows():
    vo, mix = mix_at(10.0)
    shifted = np.concatenate([np.zeros(SR, dtype=np.float32), mix])
    assert abs(separation(vo, shifted, SR, vo_offset=1.0).separation_lu - expected(10.0)) < 1.0
    assert separation(vo, shifted, SR).separation_lu < expected(10.0) - 2.0


def test_a_negative_offset_does_not_wrap_to_the_end_of_the_mix():
    vo, mix = mix_at(10.0)
    with pytest.raises(NoSpeechError, match="fall outside the mix"):
        separation(vo, mix, SR, vo_offset=-30.0)
    r = separation(vo, mix, SR, vo_offset=-2.0)   # the first run starts before zero and is dropped
    assert r.warnings == ("1 window(s) fall outside the mix; check --vo-offset",)


def test_no_speech_is_an_error_not_a_fail():
    with pytest.raises(NoSpeechError, match="no speech found in the voiceover above -44 dBFS"):
        separation(np.zeros(SR * 10, dtype=np.float32), noise(10), SR)


def test_no_gap_is_an_error():
    vo = noise(10)  # continuous speech, no gap
    with pytest.raises(NoSpeechError, match="no gap of 0.4s or more"):
        separation(vo, vo, SR)


def test_hot_mix_warns():
    vo, mix = mix_at(10.0)
    r = separation(vo, gain(mix, 20.0), SR)   # the g=10 fixture peaks near -16 dBTP; +20 puts it over 0
    assert r.peak_dbtp > -1.0
    assert r.warnings == ("mix true peak above -1 dBTP",)


def test_windows_past_the_end_are_dropped_with_a_warning():
    vo, mix = mix_at(10.0)
    # the mix ends at 20 s: the run at 21-24 s and the gap before it are out
    r = separation(vo, mix[: 20 * SR], SR)
    assert r.warnings == ("2 window(s) fall outside the mix; check --vo-offset",)
    assert abs(r.separation_lu - expected(10.0)) < 1.0
