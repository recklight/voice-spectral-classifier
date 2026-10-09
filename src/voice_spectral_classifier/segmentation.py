"""Loading, resampling, silence trimming and fixed-length segmentation.

A convolutional classifier needs inputs of one size, so every recording is cut
into windows of ``segment_s`` seconds (one second by default, the length used
for small-footprint keyword spotting; Warden, 2018). Before cutting:

1. The waveform is read as mono ``float64`` with :func:`speechdsp.read_wav` and
   resampled to one common rate with a polyphase filter
   (:func:`scipy.signal.resample_poly`). A frame length given in
   milliseconds then means the same number of samples, and the same
   frequency resolution, whether the file was recorded at 44.1 kHz or 48 kHz.
2. Optionally, leading and trailing silence is removed with the two-threshold
   endpoint detector of :func:`speechdsp.trim_silence`.

Windows start every ``hop_s`` seconds. A trailing piece shorter than one
window is zero-padded and kept when it holds at least ``min_fraction`` of a
window, and dropped otherwise. Every window keeps its index (``segment``,
counted from 0), and the classifier's scores of all windows of a recording can
later be averaged into one recording-level decision.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import speechdsp
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from .config import SegmentationConfig

_LOGGER = logging.getLogger(__name__)


def resample(x: ArrayLike, source_rate: int, target_rate: int) -> NDArray[np.float64]:
    """Resample a waveform with a polyphase anti-aliasing filter.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Mono waveform.
    source_rate, target_rate : int
        Sampling rates in Hz.

    Returns
    -------
    numpy.ndarray, shape (round(n_samples * target_rate / source_rate),)
        The resampled waveform; a copy of ``x`` when both rates are equal.

    Raises
    ------
    ValueError
        If a rate is not a positive integer or ``x`` is not one-dimensional.

    References
    ----------
    .. [1] R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal
       Processing*. Englewood Cliffs, NJ: Prentice-Hall, 1983.

    Examples
    --------
    >>> resample(np.zeros(44_100), 44_100, 16_000).shape
    (16000,)
    >>> resample(np.zeros(48_000), 48_000, 16_000).shape
    (16000,)
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a one-dimensional waveform, got shape {signal.shape}")
    for name, rate in (("source_rate", source_rate), ("target_rate", target_rate)):
        if isinstance(rate, bool) or int(rate) != rate or rate <= 0:
            raise ValueError(f"{name} must be a positive integer, got {rate!r}")
    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate == target_rate:
        return signal.copy()
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(signal, target_rate // divisor, source_rate // divisor)


def load_recording(path: str | Path, sample_rate: int) -> NDArray[np.float64]:
    """Read a WAV file as mono ``float64`` at ``sample_rate``.

    Parameters
    ----------
    path : str or pathlib.Path
        WAV file (any PCM width or float; several channels are averaged).
    sample_rate : int
        Rate in Hz the waveform is resampled to.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)
        Waveform nominally in ``[-1, 1]``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a WAV file that can be decoded, or if a
        floating-point file holds NaN or infinite samples.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = Path(folder) / "tone.wav"
    ...     speechdsp.write_wav(target, 0.5 * np.ones(48_000), 48_000)
    ...     load_recording(target, 16_000).shape
    (16000,)
    """
    source = Path(path)
    try:
        signal, rate = speechdsp.read_wav(source)
    except FileNotFoundError:
        raise
    except Exception as exc:  # scipy raises several types for a malformed file
        raise ValueError(f"cannot read {source.as_posix()} as a WAV file: {exc}") from exc
    # Checked here so the message names the file; resampling would smear one bad
    # sample over its neighbors.
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{source.as_posix()} holds NaN or infinite samples")
    if rate != sample_rate:
        _LOGGER.debug("resampling %s from %d Hz to %d Hz", source.name, rate, sample_rate)
    return resample(signal, rate, sample_rate)


def split_into_segments(
    x: ArrayLike,
    sample_rate: int,
    *,
    segment_s: float = 1.0,
    hop_s: float = 1.0,
    min_fraction: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Cut a waveform into fixed-length, possibly overlapping windows.

    Window ``i`` starts at sample ``i * hop`` and spans ``segment`` samples,
    where ``segment = round(segment_s * sample_rate)`` and
    ``hop = round(hop_s * sample_rate)``. After the last complete window, the
    window that would start at the next hop position is zero-padded and kept
    when it contains at least ``ceil(min_fraction * segment)`` samples.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Mono waveform.
    sample_rate : int
        Sampling rate in Hz.
    segment_s, hop_s : float, optional
        Window length and window advance in seconds.
    min_fraction : float, optional
        Smallest fraction of a window, in ``(0, 1]``, that a trailing piece
        must fill to be kept. ``1.0`` drops every incomplete piece.

    Returns
    -------
    segments : numpy.ndarray, shape (n_segments, segment)
        One window per row; ``n_segments`` may be 0 for a short input.
    starts : numpy.ndarray of int64, shape (n_segments,)
        First sample of each window.

    Raises
    ------
    ValueError
        For a non-positive length, hop or rate, or ``min_fraction`` outside
        ``(0, 1]``.

    Examples
    --------
    >>> segments, starts = split_into_segments(np.arange(25.0), 10, segment_s=1.0)
    >>> segments.shape, starts.tolist(), segments[-1, -3:].tolist()
    ((3, 10), [0, 10, 20], [0.0, 0.0, 0.0])
    >>> split_into_segments(np.arange(25.0), 10, min_fraction=1.0)[1].tolist()
    [0, 10]
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a one-dimensional waveform, got shape {signal.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must lie in (0, 1], got {min_fraction!r}")
    length = round(segment_s * sample_rate)
    hop = round(hop_s * sample_rate)
    if length < 1 or hop < 1:
        raise ValueError(
            f"segment_s={segment_s!r} and hop_s={hop_s!r} must each span at least one sample"
        )
    n_complete = speechdsp.num_frames(signal.size, length, hop)
    next_start = n_complete * hop
    remainder = signal.size - next_start
    keep_tail = remainder > 0 and remainder >= math.ceil(min_fraction * length)
    if keep_tail:
        signal = np.pad(signal, (0, next_start + length - signal.size))
    n_segments = n_complete + int(keep_tail)
    if n_segments == 0:
        return np.zeros((0, length), dtype=np.float64), np.zeros(0, dtype=np.int64)
    segments = speechdsp.enframe(signal, length, hop)[:n_segments]
    starts = np.arange(n_segments, dtype=np.int64) * hop
    return segments, starts


def select_evenly(n_available: int, n_wanted: int | None) -> NDArray[np.int64]:
    """Indices of at most ``n_wanted`` items spread evenly over ``n_available``.

    Parameters
    ----------
    n_available : int
        Number of items to choose from.
    n_wanted : int or None
        Upper limit; ``None`` keeps every item.

    Returns
    -------
    numpy.ndarray of int64
        Sorted, distinct indices; the first and last item are always included
        when more than one is wanted.

    Examples
    --------
    >>> select_evenly(10, 3).tolist(), select_evenly(4, None).tolist()
    ([0, 4, 9], [0, 1, 2, 3])
    """
    if n_wanted is None or n_wanted >= n_available:
        return np.arange(n_available, dtype=np.int64)
    if n_wanted < 1:
        raise ValueError(f"n_wanted must be at least 1, got {n_wanted!r}")
    if n_wanted == 1:
        return np.array([n_available // 2], dtype=np.int64)
    chosen = np.round(np.linspace(0, n_available - 1, n_wanted)).astype(np.int64)
    return np.unique(chosen)


def segment_recording(
    x: ArrayLike, config: SegmentationConfig
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Trim (optionally), cut and subsample one recording as configured.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Mono waveform already at ``config.sample_rate``.
    config : SegmentationConfig
        The ``segmentation`` section of the configuration.

    Returns
    -------
    segments : numpy.ndarray, shape (n_segments, segment)
        Selected windows.
    indices : numpy.ndarray of int64, shape (n_segments,)
        Index of each selected window among all windows of the recording
        (the ``segment`` column of the outputs).

    References
    ----------
    .. [1] P. Warden, "Speech Commands: A dataset for limited-vocabulary speech
       recognition," arXiv:1804.03209, 2018.

    Examples
    --------
    >>> from voice_spectral_classifier.config import SegmentationConfig
    >>> segmentation_config = SegmentationConfig(
    ...     sample_rate=1000, trim_silence=False, max_segments=2
    ... )
    >>> segments, indices = segment_recording(np.ones(5000), segmentation_config)
    >>> segments.shape, indices.tolist()
    ((2, 1000), [0, 4])
    """
    signal = np.asarray(x, dtype=np.float64)
    if config.trim_silence and signal.size:
        signal = speechdsp.trim_silence(signal, config.sample_rate)
    segments, _ = split_into_segments(
        signal,
        config.sample_rate,
        segment_s=config.segment_s,
        hop_s=config.hop_s,
        min_fraction=config.min_fraction,
    )
    indices = select_evenly(segments.shape[0], config.max_segments)
    return segments[indices], indices
