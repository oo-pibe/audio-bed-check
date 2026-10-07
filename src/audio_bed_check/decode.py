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


def decode(path, ffmpeg: str | None = None) -> tuple[np.ndarray, int]:
    """Decode any file ffmpeg reads to (float32 mono samples at 48 kHz, 48000)."""
    exe = find_ffmpeg(ffmpeg)
    if not os.path.exists(path):
        raise DecodeError(f"{path}: no such file")
    proc = subprocess.run(
        [
            exe, "-nostdin", "-v", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        lines = proc.stderr.decode(errors="replace").strip().splitlines()
        detail = lines[-1] if lines else "no detail from ffmpeg"
        raise DecodeError(f"{path}: ffmpeg could not decode it ({detail})")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if len(samples) == 0:
        raise DecodeError(f"{path}: decoded to no audio")
    return samples, SR
