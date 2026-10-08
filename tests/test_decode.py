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
NOT_AT_PATH = "^ffmpeg not found at /nonexistent/ffmpeg; check --ffmpeg or AUDIO_BED_CHECK_FFMPEG$"


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
    assert samples.ndim == 1
    assert abs(len(samples) - 2 * SR) < SR // 100
    assert abs(float(np.abs(samples).max()) - 0.5) < 0.02


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_stereo_keeps_its_two_channels(tmp_path):
    path = tmp_path / "stereo.wav"
    left, right = sine(2, 440, 0.5, sr=44100), sine(2, 880, 0.25, sr=44100)
    write_wav(path, np.stack([left, right], axis=1), sr=44100)
    samples, sr = decode(path)
    assert sr == SR and samples.dtype == np.float32
    assert samples.shape[1] == 2 and abs(len(samples) - 2 * SR) < SR // 100
    assert abs(float(np.abs(samples[:, 0]).max()) - 0.5) < 0.02
    assert abs(float(np.abs(samples[:, 1]).max()) - 0.25) < 0.02


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_more_than_two_channels_are_downmixed_to_two(tmp_path):
    path = tmp_path / "six.wav"
    write_wav(path, np.stack([sine(2, 440, 0.1 * (c + 1)) for c in range(6)], axis=1))
    samples, sr = decode(path)
    assert sr == SR and samples.shape == (2 * SR, 2)
    assert float(np.abs(samples).max()) > 0.05


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_a_file_name_that_looks_like_a_protocol_is_read_as_a_file(tmp_path, monkeypatch):
    # ffmpeg reads its input name as a URL: this empty file once decoded a.wav through the concat protocol
    monkeypatch.chdir(tmp_path)
    write_wav(tmp_path / "a.wav", sine(1, 440, 0.5))
    write_wav(tmp_path / "b.wav", sine(1, 440, 0.5))
    (tmp_path / "concat:a.wav|b.wav").write_bytes(b"")
    with pytest.raises(DecodeError, match=r"^concat:a\.wav\|b\.wav: ffmpeg could not decode it \("):
        decode("concat:a.wav|b.wav")


def _fake_run(returncode, stderr):
    import subprocess

    def run(args, **kwargs):
        assert kwargs["stdin"] is subprocess.DEVNULL
        return subprocess.CompletedProcess(args, returncode, b"", stderr)
    return run


def test_a_crashed_ffmpeg_is_not_blamed_on_the_file(tmp_path, monkeypatch):
    path = tmp_path / "x.wav"
    path.write_bytes(b"")
    monkeypatch.setattr("subprocess.run", _fake_run(-6, b"dyld[1]: Library not loaded: libavdevice"))
    with pytest.raises(DecodeError, match=r"x\.wav: ffmpeg failed to start \(signal 6\)$"):
        decode(path, ffmpeg=sys.executable)


@pytest.mark.parametrize("returncode, stderr, detail", [
    (1, b"dyld[42]: Library not loaded: @rpath/libavdevice.dylib\n  Reason: no such file",
     "Reason: no such file"),
    (127, b"ffmpeg: error while loading shared libraries: libavdevice.so.61: cannot open shared object file",
     "ffmpeg: error while loading shared libraries: libavdevice.so.61: cannot open shared object file"),
])
def test_a_missing_library_is_not_blamed_on_the_file(tmp_path, monkeypatch, returncode, stderr, detail):
    path = tmp_path / "x.wav"
    path.write_bytes(b"")
    monkeypatch.setattr("subprocess.run", _fake_run(returncode, stderr))
    with pytest.raises(DecodeError) as e:
        decode(path, ffmpeg=sys.executable)
    assert str(e.value) == f"{path}: ffmpeg failed to start ({detail})"


def test_a_file_named_after_the_loader_is_still_a_decode_error(tmp_path, monkeypatch):
    path = tmp_path / "dyld.wav"
    path.write_bytes(b"")
    monkeypatch.setattr("subprocess.run",
                        _fake_run(1, f"file:{path}: Invalid data found when processing input".encode()))
    with pytest.raises(DecodeError, match="ffmpeg could not decode it"):
        decode(path, ffmpeg=sys.executable)


def test_the_input_is_pinned_to_a_local_file(tmp_path, monkeypatch):
    import subprocess
    path = tmp_path / "x.wav"
    path.write_bytes(b"")
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 1, b"", b"stop here")
    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(DecodeError):
        decode(path, ffmpeg=sys.executable)
    args = seen[0]
    assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
    assert args.index("-protocol_whitelist") < args.index("-i")
    assert args[args.index("-i") + 1] == "file:" + str(path)
    assert "-ac" not in args


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


def _fmt(channels: int) -> bytes:
    """A PCM fmt chunk body: format 1, channels, 48 kHz, byte rate, block align, 16 bits."""
    return b"".join(v.to_bytes(n, "little") for v, n in (
        (1, 2), (channels, 2), (48000, 4), (48000 * 2 * channels, 4), (2 * channels, 2), (16, 2)))


def test_wav_stream_parsing_without_ffmpeg():
    from audio_bed_check.decode import _samples_from_wav
    got, channels = _samples_from_wav(_wav_stream((b"fmt ", _fmt(1)), (b"data", BODY)))
    assert got.dtype == np.float32 and channels == 1 and got.shape == (3,)
    assert np.allclose(got, [0.25, -0.5, 32767 / 32768])
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(b"garbage")
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(_wav_stream((b"fmt ", _fmt(1))))   # no data chunk at all
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(_wav_stream((b"data", BODY)))   # no fmt chunk before the data
    with pytest.raises(DecodeError, match="not a WAV stream"):
        _samples_from_wav(_wav_stream((b"fmt ", _fmt(0)), (b"data", BODY)))


def test_a_stereo_stream_is_de_interleaved():
    from audio_bed_check.decode import _samples_from_wav
    frames = np.array([8192, -8192, 16384, -16384, 1], dtype="<i2").tobytes()   # two frames and a stray half
    got, channels = _samples_from_wav(_wav_stream((b"fmt ", _fmt(2)), (b"data", frames)))
    assert channels == 2 and got.shape == (2, 2)
    assert np.allclose(got, [[0.25, -0.25], [0.5, -0.5]])


def test_tag_text_containing_data_does_not_derail_the_parser():
    # ffmpeg can copy an input's tags into a LIST/INFO chunk; "xdata" at an odd offset once made every
    # sample read one byte off, and every check then reported on noise without an error
    from audio_bed_check.decode import _samples_from_wav
    info = b"INFO" + b"INAM" + (5).to_bytes(4, "little") + b"xdata" + b"\x00"
    got, _ = _samples_from_wav(_wav_stream((b"fmt ", _fmt(1)), (b"LIST", info), (b"data", BODY)))
    assert np.allclose(got, [0.25, -0.5, 32767 / 32768])
