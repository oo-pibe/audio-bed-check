"""The only module that runs a subprocess. One ffmpeg call per file, samples piped back, no temp files."""

import os
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
        raise DecodeError("ffmpeg not found on PATH; install it or set AUDIO_BED_CHECK_FFMPEG")
    return resolved


def _samples_from_wav(data: bytes) -> np.ndarray:
    """Samples out of a 16-bit WAV stream, as float32 in [-1, 1): everything after the `data` chunk header.

    ffmpeg writes a WAV to a pipe with placeholder sizes, so the chunk length is not trusted; the
    stream ends where the samples end.
    """
    at = data.find(b"data", 12)
    if at < 0 or data[:4] != b"RIFF":
        raise DecodeError("ffmpeg returned something that is not a WAV stream")
    body = data[at + 8:]
    return np.frombuffer(body[: len(body) - len(body) % 2], dtype="<i2").astype(np.float32) / 32768.0


def decode(path, ffmpeg: str | None = None) -> tuple[np.ndarray, int]:
    """Decode any file ffmpeg reads to (float32 mono samples at 48 kHz, 48000).

    Asks for a 16-bit WAV rather than raw float: stripped ffmpeg builds bundled with some video
    renderers ship only the WAV muxer and the pcm_s16le encoder, and this tool exists for those
    pipelines. 16 bits put the quantisation floor near -98 dBFS, far below anything measured here;
    the one effect is that a source hotter than 0 dBFS is clipped on the way in, so its true peak
    reads as 0 dBTP (still above the -1 dBTP warning line).
    """
    exe = find_ffmpeg(ffmpeg)
    if not os.path.exists(path):
        raise DecodeError(f"{path}: no such file")
    proc = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", "-f", "wav", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        lines = proc.stderr.decode(errors="replace").strip().splitlines()
        detail = lines[-1] if lines else "no detail from ffmpeg"
        raise DecodeError(f"{path}: ffmpeg could not decode it ({detail})")
    samples = _samples_from_wav(proc.stdout)
    if len(samples) == 0:
        raise DecodeError(f"{path}: decoded to no audio")
    return samples, SR
