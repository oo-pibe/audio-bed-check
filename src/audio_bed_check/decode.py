"""The only module that runs a subprocess. One ffmpeg call per file, samples piped back, no temp files."""

import os
import re
import shutil
import subprocess

import numpy as np

SR = 48000


class DecodeError(RuntimeError):
    """ffmpeg is missing, or the file could not be read. The message is one sentence for the CLI."""


def find_ffmpeg(ffmpeg: str | None = None) -> str:
    """The argument, then $AUDIO_BED_CHECK_FFMPEG, then PATH. A name or a path; must be executable."""
    candidate = ffmpeg or os.environ.get("AUDIO_BED_CHECK_FFMPEG") or "ffmpeg"
    resolved = shutil.which(candidate)
    if resolved is None:
        if candidate != "ffmpeg":
            raise DecodeError(f"ffmpeg not found at {candidate}; check --ffmpeg or AUDIO_BED_CHECK_FFMPEG")
        raise DecodeError("ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG")
    return resolved


def _samples_from_wav(data: bytes) -> tuple[np.ndarray, int]:
    """(samples, channels) out of a 16-bit WAV stream: the body of the `data` chunk as float32 in
    [-1, 1), shaped `(n,)` for one channel and `(n, channels)` otherwise.

    Walks the RIFF chunks rather than searching for the bytes `data`, because an INFO chunk can carry
    tag text containing that word. The channel count comes from the `fmt ` chunk, which precedes the
    data. ffmpeg writes a WAV to a pipe with placeholder sizes, so the data chunk's length is not
    trusted; the stream ends where the samples end, and a trailing partial frame is dropped.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise DecodeError("ffmpeg returned something that is not a WAV stream")
    at, channels = 12, 0
    while at + 8 <= len(data):
        chunk, size = data[at:at + 4], int.from_bytes(data[at + 4:at + 8], "little")
        if chunk == b"fmt ":
            channels = int.from_bytes(data[at + 10:at + 12], "little")   # fmt body bytes 2-3
        elif chunk == b"data":
            if channels < 1:
                break
            body = data[at + 8:]
            frame = 2 * channels
            samples = np.frombuffer(body[: len(body) - len(body) % frame], dtype="<i2").astype(np.float32)
            samples /= 32768.0
            return (samples if channels == 1 else samples.reshape(-1, channels)), channels
        at += 8 + size + (size & 1)
    raise DecodeError("ffmpeg returned something that is not a WAV stream")


# what the dynamic loader prints when ffmpeg cannot start: macOS dyld, then glibc's ld.so
_LOADER = re.compile(r"^dyld\[|^dyld:|error while loading shared libraries", re.M)


def _run(exe: str, path, downmix: bool) -> bytes:
    """ffmpeg's WAV output for path, or a DecodeError naming the file."""
    args = [exe, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-i", "file:" + os.fspath(path), "-map", "0:a:0", "-map_metadata", "-1", "-fflags", "+bitexact"]
    if downmix:
        args += ["-ac", "2"]
    args += ["-ar", str(SR), "-c:a", "pcm_s16le", "-f", "wav", "-"]
    proc = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        # first: the loader aborts with a signal (dyld raises SIGABRT), and its line names the library
        loader = _LOADER.search(stderr)
        if loader:
            start = stderr.rfind("\n", 0, loader.start()) + 1
            end = stderr.find("\n", loader.start())
            line = stderr[start:end if end >= 0 else len(stderr)].strip()
            raise DecodeError(f"{path}: ffmpeg failed to start ({line})")
        if proc.returncode < 0:
            raise DecodeError(f"{path}: ffmpeg failed to start (signal {-proc.returncode})")
        lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
        detail = lines[-1] if lines else "no detail from ffmpeg"
        if any("matches no streams" in line for line in lines):
            detail = "it has no audio stream"
        raise DecodeError(f"{path}: ffmpeg could not decode it ({detail})")
    return proc.stdout


def decode(path, ffmpeg: str | None = None) -> tuple[np.ndarray, int]:
    """Decode the first audio track of any file ffmpeg reads to (float32 samples at 48 kHz, 48000).

    The samples are `(n,)` for a mono track and `(n, 2)` for stereo. A track with more than two
    channels is decoded a second time with ffmpeg's `-ac 2` downmix and returned as `(n, 2)`.

    The input is always a local file: the path is passed as `file:PATH` with ffmpeg's protocol
    whitelist set to file and pipe, so a name such as `concat:a.wav|b.wav` is opened as that file.

    Asks for a 16-bit WAV rather than raw float: stripped ffmpeg builds bundled with some video
    renderers ship only the WAV muxer and the pcm_s16le encoder, and this tool exists for those
    pipelines. 16 bits put the quantisation floor near -98 dBFS, far below anything measured here;
    the one effect is that a source hotter than 0 dBFS is clipped on the way in, so its true peak
    reads as 0 dBTP (still above the -1 dBTP warning line).
    """
    exe = find_ffmpeg(ffmpeg)
    if not os.path.exists(path):
        raise DecodeError(f"{path}: no such file")
    samples, channels = _samples_from_wav(_run(exe, path, downmix=False))
    if channels > 2:
        samples, channels = _samples_from_wav(_run(exe, path, downmix=True))
    if len(samples) == 0:
        raise DecodeError(f"{path}: decoded to no audio")
    return samples, SR
