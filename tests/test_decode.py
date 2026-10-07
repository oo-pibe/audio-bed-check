import shutil

import numpy as np
import pytest

from audio_bed_check.decode import DecodeError, decode, find_ffmpeg
from tests.synth import SR, sine, write_wav

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_missing_ffmpeg_is_one_sentence(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO_BED_CHECK_FFMPEG", "/nonexistent/ffmpeg")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(
        DecodeError,
        match="^ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG$",
    ):
        decode(tmp_path / "x.wav")


def test_explicit_argument_wins_over_env(monkeypatch):
    monkeypatch.setenv("AUDIO_BED_CHECK_FFMPEG", "/nonexistent/ffmpeg")
    with pytest.raises(DecodeError, match="ffmpeg not found"):
        find_ffmpeg("/also/nonexistent")


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_decodes_to_48k_mono_float(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, sine(2, 440, 0.5, sr=44100), sr=44100)
    samples, sr = decode(path)
    assert sr == SR
    assert samples.dtype == np.float32
    assert abs(len(samples) - 2 * SR) < SR // 100
    assert abs(float(np.abs(samples).max()) - 0.5) < 0.02


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_missing_file(tmp_path):
    with pytest.raises(DecodeError, match="no such file"):
        decode(tmp_path / "missing.wav")


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_undecodable_file(tmp_path):
    path = tmp_path / "junk.wav"
    path.write_bytes(b"not audio at all")
    with pytest.raises(DecodeError, match="ffmpeg could not decode it"):
        decode(path)


def test_wav_stream_parsing_without_ffmpeg():
    from audio_bed_check.decode import _samples_from_wav
    body = np.array([8192, -16384, 32767], dtype="<i2").tobytes()
    header = b"RIFF" + b"\xff\xff\xff\xff" + b"WAVE" + b"fmt " + (16).to_bytes(4, "little") + bytes(16)
    stream = header + b"data" + b"\xff\xff\xff\xff" + body
    got = _samples_from_wav(stream)
    assert got.dtype == np.float32
    assert np.allclose(got, [0.25, -0.5, 32767 / 32768])
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(b"garbage")
