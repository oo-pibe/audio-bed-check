import json

import numpy as np
import pytest

from audio_bed_check import cli
from audio_bed_check.decode import DecodeError
from tests.synth import SR, gain, gated, noise, step_gain, textured, tile, write_wav


@pytest.fixture
def wavs(tmp_path, monkeypatch):
    """Four files on disk, and decode() swapped for a WAV reader so no ffmpeg is needed."""
    from tests.synth import read_wav
    monkeypatch.setattr(cli, "decode", lambda path, ffmpeg=None: read_wav(path))
    seg = textured(12, 1)
    clean = textured(48, 2)
    stepped = noise(60, 3, rms_dbfs=-30.0) * step_gain(60 * SR, 30.0, 7.0)
    bed = noise(30, 4, rms_dbfs=-30.0)   # quiet enough that bed + voice stays under 0 dBFS in the WAV
    # ~14 LU
    vo = gated(gain(noise(30, 5, rms_dbfs=-30.0), 14.0), [(1, 4), (6, 9), (11, 14), (16, 19), (21, 24)])
    files = {
        "looped": tile(seg, 4), "clean": clean, "stepped": stepped,
        "bed": bed, "vo": vo, "mix": (bed + vo).astype(np.float32),
    }
    paths = {}
    for name, x in files.items():
        paths[name] = tmp_path / f"{name}.wav"
        write_wav(paths[name], x)
    return paths


def test_clean_bed_passes(wavs, capsys):
    assert cli.main(["bed", str(wavs["clean"])]) == 0
    out = capsys.readouterr().out
    assert "ok    loop" in out and "ok    step" in out and "info  range" in out


def test_looped_bed_fails_and_names_the_period(wavs, capsys):
    assert cli.main(["loop", str(wavs["looped"])]) == 1
    out = capsys.readouterr().out
    assert "FAIL  loop" in out and "repeats every 12.0s" in out


def test_batch_keeps_going_and_reports_every_file(wavs, capsys):
    assert cli.main(["bed", str(wavs["stepped"]), str(wavs["clean"])]) == 1
    out = capsys.readouterr().out
    assert "FAIL  step" in out and "7." in out and "at 30.0s" in out
    assert out.count("ok    loop") == 2


def test_json_shape(wavs, capsys):
    assert cli.main(["bed", str(wavs["clean"]), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and len(data) == 1
    assert set(data[0]) == {"file", "passed", "loop", "steps"}
    assert set(data[0]["loop"]) == {"score", "period_s", "threshold", "min_period_s", "passed", "notes"}
    assert set(data[0]["steps"]) == {
        "step_db", "step_at_s", "range_db", "transient_db", "peak_dbtp", "max_step", "passed", "notes",
    }


def test_separation_passes_at_music_and_fails_at_wcag(wavs, capsys):
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"])]) == 0
    assert "ok    separation" in capsys.readouterr().out
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]), "--profile", "wcag"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  separation" in out and "min 20.0, wcag" in out


def test_separation_json_shape(wavs, capsys):
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(wavs["vo"]), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data[0]) == {"file", "passed", "profile", "separation"}
    assert set(data[0]["separation"]) == {
        "separation_lu", "speech_lkfs", "bed_lkfs", "runs", "gaps", "peak_dbtp", "min_lu", "passed",
        "warnings",
    }


def test_no_speech_is_exit_2(wavs, capsys, tmp_path):
    silent = tmp_path / "silent.wav"
    write_wav(silent, np.zeros(SR * 10, dtype=np.float32))
    assert cli.main(["separation", str(wavs["mix"]), "--vo", str(silent)]) == 2
    assert "audio-bed-check: no speech found in the voiceover above -44 dBFS" in capsys.readouterr().err


def test_decode_error_is_exit_2(monkeypatch, capsys, tmp_path):
    def boom(path, ffmpeg=None):
        raise DecodeError("ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG")
    monkeypatch.setattr(cli, "decode", boom)
    assert cli.main(["loop", str(tmp_path / "x.wav")]) == 2
    assert capsys.readouterr().err == (
        "audio-bed-check: ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG\n"
    )


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("audio-bed-check 0.")


def test_usage_error_is_exit_2(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["separation", "mix.wav"])  # --vo is required
    assert e.value.code == 2
