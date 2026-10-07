import sys

import numpy as np
import pytest

from audio_bed_check.decode import DecodeError, decode, find_ffmpeg
from tests.synth import SR, sine, write_wav

try:
    find_ffmpeg()
    HAVE_FFMPEG = True
except DecodeError:
    HAVE_FFMPEG = False


NOT_ON_PATH = "^ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG$"
NOT_AT_PATH = "^ffmpeg not found at /nonexistent/ffmpeg; install it or set AUDIO_BED_CHECK_FFMPEG$"


def test_missing_ffmpeg_is_one_sentence(monkeypatch, tmp_path):
    monkeypatch.delenv("AUDIO_BED_CHECK_FFMPEG", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(DecodeError, match=NOT_ON_PATH):
        decode(tmp_path / "x.wav")


def test_explicit_argument_wins_over_env(monkeypatch):
    monkeypatch.setenv("AUDIO_BED_CHECK_FFMPEG", "/nonexistent/ffmpeg")
    assert find_ffmpeg(sys.executable) == sys.executable   # any executable proves the argument is used
    with pytest.raises(DecodeError, match=NOT_AT_PATH):
        find_ffmpeg()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_decodes_to_48k_mono_float(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, sine(2, 440, 0.5, sr=44100), sr=44100)   # a real 44.1 kHz file, resampled on decode
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


def _wav_stream(*chunks: tuple[bytes, bytes]) -> bytes:
    """A RIFF/WAVE stream with placeholder sizes, the way ffmpeg writes one to a pipe, plus the chunks."""
    out = b"RIFF" + b"\xff\xff\xff\xff" + b"WAVE"
    for name, body in chunks:
        out += name + len(body).to_bytes(4, "little") + body + (b"\x00" if len(body) % 2 else b"")
    return out


BODY = np.array([8192, -16384, 32767], dtype="<i2").tobytes()


def test_wav_stream_parsing_without_ffmpeg():
    from audio_bed_check.decode import _samples_from_wav
    got = _samples_from_wav(_wav_stream((b"fmt ", bytes(16)), (b"data", BODY)))
    assert got.dtype == np.float32
    assert np.allclose(got, [0.25, -0.5, 32767 / 32768])
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(b"garbage")
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(_wav_stream((b"fmt ", bytes(16))))   # no data chunk at all


def test_tag_text_containing_data_does_not_derail_the_parser():
    # ffmpeg can copy an input's tags into a LIST/INFO chunk; "xdata" at an odd offset once made every
    # sample read one byte off, and every check then reported on noise without an error
    from audio_bed_check.decode import _samples_from_wav
    info = b"INFO" + b"INAM" + (5).to_bytes(4, "little") + b"xdata" + b"\x00"
    got = _samples_from_wav(_wav_stream((b"fmt ", bytes(16)), (b"LIST", info), (b"data", BODY)))
    assert np.allclose(got, [0.25, -0.5, 32767 / 32768])
