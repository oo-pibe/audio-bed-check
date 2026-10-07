import json

import numpy as np
import pytest

from audio_bed_check import cli
from audio_bed_check.decode import DecodeError
from tests.synth import SR, gain, gated, noise, step_gain, textured, tile, write_wav

RUNS = [(1, 4), (6, 9), (11, 14), (16, 19), (21, 24)]


@pytest.fixture
def wavs(tmp_path, monkeypatch):
    """Seven files on disk, and decode() swapped for a WAV reader so no ffmpeg is needed."""
    from tests.synth import read_wav
    monkeypatch.setattr(cli, "decode", lambda path, ffmpeg=None: read_wav(path))
    bed = noise(30, 4, rms_dbfs=-30.0)   # quiet enough that bed + voice stays under 0 dBFS in the WAV
    vo = gated(gain(noise(30, 5, rms_dbfs=-30.0), 14.0), RUNS)   # ~14 LU: clear of 10, under 20
    files = {
        "looped": tile(textured(12, 1), 4),
        "clean": textured(48, 2),
        # +7 dB must stay under 0 dBFS
        "stepped": noise(60, 3, rms_dbfs=-30.0) * step_gain(60 * SR, 30.0, 7.0),
        "bed": bed, "vo": vo, "mix": (bed + vo).astype(np.float32),
        "silent": np.zeros(20 * SR, dtype=np.float32),
    }
    paths = {}
    for name, x in files.items():
        paths[name] = tmp_path / f"{name}.wav"
        write_wav(paths[name], x)
    return paths


def test_clean_bed_passes(wavs, capsys):
    assert cli.main(["bed", str(wavs["clean"])]) == 0
    out = capsys.readouterr().out
    assert "ok    loop" in out and "no repeat at or above 0.75" in out
    assert "ok    step" in out and "info  range" in out


def test_looped_bed_fails_and_names_the_period(wavs, capsys):
    assert cli.main(["loop", str(wavs["looped"])]) == 1
    out = capsys.readouterr().out
    assert "FAIL  loop        1.00 (repeats every 12.0s, at or above 0.75)" in out


def test_batch_keeps_going_and_reports_every_file(wavs, capsys):
    assert cli.main(["bed", str(wavs["stepped"]), str(wavs["clean"])]) == 1
    out = capsys.readouterr().out
    assert str(wavs["stepped"]) in out and str(wavs["clean"]) in out
    assert "FAIL  step        7.0 dB at 30.0s (max 6.0)" in out
    assert out.count("ok    loop") == 2 and out.count("ok    step") == 1


def test_a_silent_bed_renders_without_crashing(wavs, capsys):
    assert cli.main(["bed", str(wavs["silent"])]) == 0
    out = capsys.readouterr().out
    assert "ok    loop        0.00 (level is constant; nothing to correlate)" in out
    assert "ok    step        0.0 dB at" in out


def test_too_short_file_prints_the_note_on_an_ok_line(wavs, capsys, tmp_path):
    short = tmp_path / "short.wav"
    write_wav(short, textured(10, 9))
    assert cli.main(["loop", str(short)]) == 0
    assert "ok    loop        0.00 (too short to test for a repeat longer than 6s)" in capsys.readouterr().out


def test_json_shape_and_rounding(wavs, capsys):
    assert cli.main(["bed", str(wavs["clean"]), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and len(data) == 1
    assert set(data[0]) == {"file", "passed", "loop", "steps"}
    assert set(data[0]["loop"]) == {"score", "period_s", "threshold", "min_period_s", "passed", "notes"}
    assert set(data[0]["steps"]) == {
        "step_db", "step_at_s", "range_db", "transient_db", "peak_dbtp", "max_step", "passed", "notes",
    }
    for section in ("loop", "steps"):
        for value in data[0][section].values():
            if isinstance(value, float):
                assert round(value, 2) == value


def test_separation_passes_at_music_and_fails_at_wcag(wavs, capsys):
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"])]) == 0
    assert "ok    separation" in capsys.readouterr().out
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]), "--profile", "wcag"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  separation" in out and "min 20.0, wcag)" in out


def test_min_separation_overrides_and_says_so(wavs, capsys):
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]), "--profile", "wcag",
                     "--min-separation", "5"]) == 0
    assert "min 5.0, --min-separation)" in capsys.readouterr().out
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]),
                     "--min-separation", "5", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["profile"] == "--min-separation"


def test_vo_offset_and_the_warn_line(wavs, capsys, tmp_path):
    from tests.synth import read_wav
    mix, _ = read_wav(wavs["mix"])
    padded = tmp_path / "padded.wav"
    write_wav(padded, np.concatenate([np.zeros(SR, dtype=np.float32), mix]))
    assert cli.main(["separation", str(padded), "--vo", str(wavs["vo"])]) == 1
    assert cli.main(["separation", str(padded), "--vo", str(wavs["vo"]), "--vo-offset", "1"]) == 0
    capsys.readouterr()
    # a wrong offset on the unpadded 30 s mix pushes the last run and the gap before it past the end;
    # the verdict is whatever the misaligned windows say, the warn line is what the test pins
    cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]), "--vo-offset", "10"])
    assert "  warn  2 window(s) fall outside the mix; check --vo-offset" in capsys.readouterr().out


def test_separation_json_shape(wavs, capsys):
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data[0]) == {"file", "passed", "profile", "separation"}
    assert set(data[0]["separation"]) == {
        "separation_lu", "speech_lkfs", "bed_lkfs", "runs", "gaps", "peak_dbtp", "min_lu", "passed",
        "warnings",
    }


def test_no_speech_is_exit_2(wavs, capsys):
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["silent"])]) == 2
    assert "audio-bed-check: no speech found in the voiceover above -44 dBFS" in capsys.readouterr().err


def test_decode_error_is_exit_2(monkeypatch, capsys, tmp_path):
    def boom(path, ffmpeg=None):
        raise DecodeError("ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG")
    monkeypatch.setattr(cli, "decode", boom)
    assert cli.main(["loop", str(tmp_path / "x.wav")]) == 2
    assert capsys.readouterr().err == (
        "audio-bed-check: ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG\n"
    )


def test_a_library_value_error_is_exit_2_not_a_traceback(wavs, monkeypatch, capsys):
    def guard(*a, **k):
        raise ValueError("min_period must be positive")
    monkeypatch.setattr(cli, "loop_score", guard)
    assert cli.main(["loop", str(wavs["clean"])]) == 2
    assert capsys.readouterr().err == "audio-bed-check: min_period must be positive\n"


@pytest.mark.parametrize("argv", [
    ["loop", "x.wav", "--min-period", "0"],
    ["loop", "x.wav", "--min-period", "nan"],
    ["loop", "x.wav", "--loop-threshold", "1.5"],
    ["steps", "x.wav", "--edge", "-1"],
    ["steps", "x.wav", "--max-step", "-6"],
    ["separation", "m.wav", "--vo", "v.wav", "--vo-offset", "inf"],
    ["separation", "m.wav", "--vo", "v.wav", "--profile", "bogus"],
])
def test_bad_flag_values_are_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("audio-bed-check 0.")


def test_usage_error_is_exit_2(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["separation", "mix.wav"])  # --vo is required
    assert e.value.code == 2
