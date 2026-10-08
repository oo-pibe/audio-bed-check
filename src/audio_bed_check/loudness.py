"""ITU-R BS.1770-4 loudness pieces in numpy: K-weighting, block loudness, momentary series, true peak.

Samples are `(n,)` for mono or `(n, 2)` for stereo, float, finite. Channels are measured separately and
their mean squares summed (BS.1770 weights L and R 1.0), never downmixed: a downmix cancels an
anti-phase pair to silence and halves an uncorrelated bed against a centred voice.

Everything is 48 kHz only. The decoder always produces 48 kHz, and the filter coefficients are the
standard's published 48 kHz set; refusing other rates is cheaper than getting them silently wrong.
"""

import functools

import numpy as np

SR = 48000
HOP = 0.1          # momentary hop, seconds
WINDOW = 0.4       # momentary window, seconds
FLOOR_LKFS = -100.0
FLOOR_DBTP = -99.0

# BS.1770-4 Annex 1, 48 kHz: (b0, b1, b2, a1, a2). Stage 1 is the high shelf, stage 2 the RLB high pass.
_STAGE1 = (1.53512485958697, -2.69169618940638, 1.19839281085285, -1.69065929318241, 0.73248077421585)
_STAGE2 = (1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621)
_TAPS = 4096       # the slowest pole has radius ~0.995; 0.995**4096 is ~1e-9, so truncation is inaudible


def _impulse_response(n: int = _TAPS) -> np.ndarray:
    """Run a unit impulse through both biquads (a tiny Python loop)."""
    x = np.zeros(n)
    x[0] = 1.0
    for b0, b1, b2, a1, a2 in (_STAGE1, _STAGE2):
        y = np.zeros(n)
        for i in range(n):
            y[i] = b0 * x[i]
            if i >= 1:
                y[i] += b1 * x[i - 1] - a1 * y[i - 1]
            if i >= 2:
                y[i] += b2 * x[i - 2] - a2 * y[i - 2]
        x = y
    return x


@functools.cache
def _kernel() -> np.ndarray:
    """The cascade's impulse response, computed once; read-only so no caller can corrupt it."""
    h = _impulse_response()
    h.flags.writeable = False
    return h


def _check_rate(sr: int) -> None:
    if sr != SR:
        raise ValueError("resample to 48 kHz")


def _channels(samples) -> np.ndarray:
    """samples as a float64 (n, ch) array, ch 1 or 2; ValueError for anything else."""
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] not in (1, 2):
        raise ValueError("pass mono or stereo samples")
    if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError("samples must be finite floats")
    return x.astype(np.float64, copy=False)


def fft_convolve(x: np.ndarray, h: np.ndarray, block: int = 1 << 17, full: bool = False) -> np.ndarray:
    """Linear convolution of x with h by overlap-add, so long files stay in memory.

    Truncated to len(x) (a causal filter's output) unless full=True, which returns all len(x)+len(h)-1.
    """
    n, m = len(x), len(h)
    size = 1
    while size < block + m - 1:
        size <<= 1
    spectrum = np.fft.rfft(h, size)
    out = np.zeros(n + m - 1)
    for start in range(0, n, block):
        seg = x[start:start + block]
        piece = np.fft.irfft(np.fft.rfft(seg, size) * spectrum, size)[:len(seg) + m - 1]
        out[start:start + len(piece)] += piece
    return out if full else out[:n]


def k_weight(samples: np.ndarray, sr: int) -> np.ndarray:
    """The K-weighted signal (float64, same shape), each channel filtered on its own.

    The zero-initial-state response of the two BS.1770 biquads, truncated to 4096 taps.
    """
    _check_rate(sr)
    x = _channels(samples)
    out = np.stack([fft_convolve(x[:, c], _kernel()) for c in range(x.shape[1])], axis=1)
    return out if np.ndim(samples) == 2 else out[:, 0]


def block_loudness(weighted: np.ndarray, sr: int, window: float, hop: float) -> np.ndarray:
    """Loudness of an already K-weighted signal per block, in LKFS; floored at -100.

    Mono `(n,)` or stereo `(n, 2)`; the channels' mean squares are summed before the log.
    `window` and `hop` are seconds. Input shorter than one window returns an empty array. The running
    sum of squares is monotone, so no window can go negative; true silence gives 0, then the floor.
    """
    _check_rate(sr)
    x = _channels(weighted)
    w, h = int(round(window * sr)), int(round(hop * sr))
    if h < 1:
        raise ValueError("hop must be at least one sample")
    if len(x) < w:
        return np.zeros(0)
    starts = np.arange(0, len(x) - w + 1, h)
    cumulative = np.concatenate([[0.0], np.cumsum((x * x).sum(axis=1))])
    mean_square = (cumulative[starts + w] - cumulative[starts]) / w
    with np.errstate(divide="ignore"):
        lkfs = -0.691 + 10 * np.log10(mean_square)
    return np.maximum(lkfs, FLOOR_LKFS)


def momentary(samples: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 momentary loudness: 400 ms windows every 100 ms."""
    return block_loudness(k_weight(samples, sr), sr, WINDOW, HOP)


_UP = 4             # oversampling ratio for true peak, per BS.1770-4
_INTERP_HALF = 64   # kernel half-length in the oversampled domain: 16 input samples of context each side


def _interp_kernel() -> np.ndarray:
    """Windowed-sinc low-pass for a 4x zero-stuffed signal: cutoff at the original Nyquist, DC gain 4."""
    n = np.arange(-_INTERP_HALF, _INTERP_HALF + 1)
    h = np.sinc(n / _UP) * np.kaiser(len(n), 9.0)
    return h * (_UP / h.sum())


def true_peak(samples: np.ndarray, sr: int, block: int = 1 << 16) -> float:
    """Peak of the 4x oversampled waveform, in dBTP: the loudest channel.

    Zero-stuff, then a linear (never circular) windowed-sinc interpolation by overlap-add, so the
    answer does not depend on `block` and the file counts as silent beyond its ends.
    """
    _check_rate(sr)
    channels = _channels(samples)
    n = len(channels)
    if n == 0:
        return FLOOR_DBTP
    peak = max(_channel_peak(channels[:, c], block) for c in range(channels.shape[1]))
    return float(20 * np.log10(peak)) if peak > 0 else FLOOR_DBTP


def _channel_peak(x: np.ndarray, block: int) -> float:
    """Linear 4x-oversampled peak of one channel."""
    n = len(x)
    h = _interp_kernel()
    margin = _INTERP_HALF // _UP + 1      # input samples of context a kept sample needs on each side
    peak = 0.0
    for start in range(0, n, block):
        end = min(n, start + block)
        a, b = max(0, start - margin), min(n, end + margin)
        up = np.zeros((b - a) * _UP)
        up[::_UP] = x[a:b]
        y = fft_convolve(up, h, full=True)  # y[j] is the interpolant centred on up[j - _INTERP_HALF]
        lo = (start - a) * _UP + _INTERP_HALF
        hi = (end - a) * _UP + _INTERP_HALF
        peak = max(peak, float(np.abs(y[lo:hi]).max()))
    return peak
