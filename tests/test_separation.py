import numpy as np
import pytest

from audio_bed_check.checks import NoSpeechError, SeparationResult, gaps_between, separation, speech_runs
from tests.synth import SR, gain, gated, irregular_read, noise, textured

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
    assert r.warnings == ("1 window(s) fall outside the mix; check --vo-offset",
                          "the voiceover seems to start at 0.0s in the mix; check --vo-offset")


def test_no_speech_is_an_error_not_a_fail():
    with pytest.raises(NoSpeechError, match="no speech found in the voiceover above -44 dBFS"):
        separation(np.zeros(SR * 10, dtype=np.float32), noise(10), SR)


def test_no_gap_is_an_error():
    vo = noise(10)  # continuous speech, no gap
    message = ("the voiceover has no pause of 0.9s or more between speech runs (0.4s after a 0.25s guard at "
               "each end); either --vo is the mix rather than the voiceover alone, or its noise floor is "
               "above --gate")
    with pytest.raises(NoSpeechError) as e:
        separation(vo, vo, SR)
    assert str(e.value) == message


def test_the_voiceover_given_as_the_mix_is_an_error():
    # the gaps of a voiceover are silence; measured as the mix it passed at 73 LU
    vo, _ = mix_at(10.0)
    message = r"^the bed-only windows are silent in the mix; is MIX the rendered mix\?$"
    with pytest.raises(NoSpeechError, match=message):
        separation(vo, vo, SR)


def test_pauses_of_exactly_0_9_seconds_are_kept():
    # 2 s runs every 2.9 s: each pause is 0.9 s, so each gap is 0.4 s after the guards, which float
    # rounding once read as 0.39999 and dropped
    runs = [(round(k * 2.9, 2), round(k * 2.9 + 2.0, 2)) for k in range(10)]
    vo = gated(noise(30, 300, rms_dbfs=-30.0), runs)
    found = speech_runs(vo, SR)
    assert found == runs
    assert len(gaps_between(found)) == 9
    assert separation(vo, (vo + noise(30, 301, rms_dbfs=-45.0)).astype(np.float32), SR).gaps == 9


@pytest.mark.parametrize("start", [0.01, 0.03, 0.07, 0.13])
def test_a_0_9_second_pause_off_the_gate_grid_is_kept(start):
    # off the 20 ms grid a run's end rounds up a hop and the next start rounds down, so the pause
    # measures 0.88 s; it was dropped in 9 of 10 phases
    runs = [(round(start + k * 2.9, 3), round(start + k * 2.9 + 2.0, 3)) for k in range(10)]
    found = speech_runs(gated(noise(30, 310, rms_dbfs=-30.0), runs), SR)
    assert len(found) == 10
    assert len(gaps_between(found)) == 9


@pytest.mark.parametrize("start", [0.0, 0.01, 0.03, 0.07, 0.13])
def test_a_0_85_second_pause_is_not_kept(start):
    runs = [(round(start + k * 2.85, 3), round(start + k * 2.85 + 2.0, 3)) for k in range(10)]
    found = speech_runs(gated(noise(30, 311, rms_dbfs=-30.0), runs), SR)
    assert len(found) == 10
    assert gaps_between(found) == []


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def irregular_mix(vo_over_bed_db: float, seed: int, pad: float = 0.0):
    """An irregular read placed `pad` seconds into a textured bed that runs the whole mix."""
    runs = irregular_read(30, seed)
    bed = textured(30 + pad, 600 + seed)
    bed = gain(bed, -30.0 - _rms_db(bed))
    vo = gated(gain(noise(30, 500 + seed, rms_dbfs=-30.0), vo_over_bed_db), runs)
    placed = np.concatenate([np.zeros(int(pad * SR), dtype=np.float32), vo])
    return vo, (bed + placed).astype(np.float32)


@pytest.mark.parametrize("seed", range(4))
def test_a_buried_voice_gives_no_estimate_and_no_warning(seed):
    # 10 dB under a textured bed the envelopes matched best at lags up to 14 s away
    vo, mix = irregular_mix(-10.0, seed)
    r = separation(vo, mix, SR)
    assert r.estimated_offset_s is None
    assert not any("seems to start" in w for w in r.warnings)


@pytest.mark.parametrize("seed", range(4))
def test_a_clear_voice_is_still_found_three_seconds_in(seed):
    vo, mix = irregular_mix(10.0, seed, pad=3.0)
    r = separation(vo, mix, SR)
    assert r.estimated_offset_s == pytest.approx(3.0, abs=0.05)
    assert "the voiceover seems to start at 3.0s in the mix; check --vo-offset" in r.warnings
    assert separation(vo, mix, SR, vo_offset=3.0).warnings == ()


def test_separation_k_weights_each_file_once(monkeypatch):
    import audio_bed_check.checks as checks
    import audio_bed_check.loudness as loudness
    calls = []
    real = loudness.k_weight

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(checks, "k_weight", counting)
    monkeypatch.setattr(loudness, "k_weight", counting)
    vo, mix = mix_at(10.0)
    separation(vo, mix, SR)
    assert len(calls) == 2   # the mix once, the voiceover once


def test_the_offset_is_estimated_and_a_wrong_one_is_named():
    vo, mix = mix_at(10.0)
    assert separation(vo, mix, SR).estimated_offset_s == pytest.approx(0.0, abs=0.05)
    padded = np.concatenate([np.zeros(3 * SR, dtype=np.float32), mix])
    wrong = separation(vo, padded, SR)
    assert wrong.estimated_offset_s == pytest.approx(3.0, abs=0.05)
    assert "the voiceover seems to start at 3.0s in the mix; check --vo-offset" in wrong.warnings
    right = separation(vo, padded, SR, vo_offset=3.0)
    assert right.warnings == ()


def test_the_estimate_holds_over_a_bed_with_its_own_texture():
    vo = gated(gain(noise(30, 400, rms_dbfs=-40.0), 10.0), RUNS)
    mix = np.concatenate([np.zeros(2 * SR, dtype=np.float32), (textured(30, 401) * 0.01 + vo)])
    assert separation(vo, mix.astype(np.float32), SR, vo_offset=2.0).warnings == ()


def test_hot_mix_warns():
    vo, mix = mix_at(10.0)
    r = separation(vo, gain(mix, 20.0), SR)   # the g=10 fixture peaks near -16 dBTP; +20 puts it over 0
    assert r.peak_dbtp > -1.0
    assert r.warnings == ("mix true peak above -1 dBTP",)


def test_windows_past_the_end_are_dropped_with_a_warning():
    vo, mix = mix_at(10.0)
    # the mix ends at 20 s: the run at 21-24 s and the gap before it are out. The read pauses every 5 s,
    # so -5 s matches the truncated mix as well as 0 does (8446 against 8445): the given offset stands
    r = separation(vo, mix[: 20 * SR], SR)
    assert r.warnings == ("2 window(s) fall outside the mix; check --vo-offset",)
    assert abs(r.separation_lu - expected(10.0)) < 1.0


def test_stereo_with_identical_channels_reads_as_mono():
    vo, mix = mix_at(10.0)
    mono = separation(vo, mix, SR)
    both = separation(np.stack([vo, vo], axis=1), np.stack([mix, mix], axis=1), SR)
    assert abs(both.separation_lu - mono.separation_lu) < 0.1
    assert (both.runs, both.gaps) == (mono.runs, mono.gaps)


def test_a_centred_voice_over_a_wide_bed_is_summed_per_channel():
    # L = v + bL, R = v + bR with independent beds. A mono downmix halves the bed's power (the beds do not
    # add coherently) but keeps the voice's, so it read 2.5 LU too much separation at 6 dB
    g = 6.0
    vo = gated(gain(noise(30, 200, rms_dbfs=-40.0), g), RUNS)
    left_bed, right_bed = noise(30, 201, rms_dbfs=-40.0), noise(30, 202, rms_dbfs=-40.0)
    mix = np.stack([vo + left_bed, vo + right_bed], axis=1).astype(np.float32)
    r = separation(vo, mix, SR)

    # the BS.1770 reference by hand: sum the channels' mean squares in each window (weights 1.0). Every
    # signal is white noise, so K-weighting scales voice and bed alike and drops out of the ratio
    def power(windows):
        return np.mean([(mix[int(a * SR):int(b * SR)] ** 2).sum(axis=1).mean() for a, b in windows])
    reference = 10 * np.log10(power(RUNS) / power(gaps_between(RUNS)))
    assert abs(reference - expected(g)) < 0.2           # the reference is the textbook figure
    assert abs(r.separation_lu - reference) < 1.0
    downmix = separation(vo, mix.mean(axis=1), SR).separation_lu
    assert downmix - reference > 2.0                     # and the old downmix really was wrong


def test_speech_runs_gate_on_the_power_of_all_channels():
    vo, _ = mix_at(10.0)
    assert speech_runs(np.stack([vo, vo], axis=1), SR) == speech_runs(vo, SR)
    assert speech_runs(np.stack([vo, -vo], axis=1), SR) == speech_runs(vo, SR)   # anti-phase is not silence
    # voice on one channel only: RMS over every sample of both channels is 3 dB under the mono figure
    one = np.stack([vo, np.zeros_like(vo)], axis=1)
    assert speech_runs(one, SR) == speech_runs(gain(vo, -3.0103), SR)
