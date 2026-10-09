"""Fixed-size grey-scale spectrogram images of the segments.

Each segment becomes an ``n_bands x n_frames`` array of 8-bit levels (40 x 98
by default: 40 frequency bands by 98 frames of 25 ms every 10 ms, the number
of frames that fit in one second; Sainath & Parada, 2015; Warden, 2018). The
steps are:

1. Short-time power spectrum (Allen, 1977). The segment is cut into frames
   with :func:`speechdsp.enframe`, weighted by an analysis window (Hamming by
   default; Harris, 1978), zero-padded to ``n_fft`` points and transformed with
   a real FFT. The one-sided power spectral density of frame :math:`t` is

   .. math::

      P_t[k] = \\frac{c_k\\,|X_t[k]|^2}{f_s \\sum_n w[n]^2},

   with :math:`c_k = 2` for bins strictly between DC and the Nyquist
   frequency and :math:`c_k = 1` otherwise.
2. Band pooling between 0 Hz and ``fmax_hz`` into ``n_bands`` rows. The
   ``linear`` scale averages the power over equal-width bands, each FFT bin
   weighted by how much of its width falls inside the band; the ``mel``
   scale applies the triangular mel filterbank of
   :func:`speechdsp.mel_filterbank` (Stevens et al., 1937; Davis &
   Mermelstein, 1980).
3. Decibels relative to the segment's own peak, floored at ``-top_db``:
   :math:`L = \\max(10 \\log_{10}(E / E_{\\max}), -\\mathrm{top\\_db})`.
4. Linear interpolation of the time axis to ``n_frames`` columns (nothing
   changes when the segment already yields that many frames).
5. Linear quantization of ``[-top_db, 0]`` dB onto the integers 0-255.

Row 0 is the *lowest* band and column 0 the earliest frame. Because the last
step is linear, a grey level converts back to decibels; one level is
``top_db / 255`` dB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import speechdsp
from numpy.typing import ArrayLike, NDArray
from speechdsp.framing import resolve_window

if TYPE_CHECKING:
    from .config import SpectrogramConfig

#: Spectrogram scales accepted by :func:`spectrogram_image` and ``spectrogram.scale``.
SPECTROGRAM_SCALES: tuple[str, ...] = ("linear", "mel")
#: Analysis windows accepted by ``spectrogram.window`` (``boxcar`` is rectangular).
WINDOW_NAMES: tuple[str, ...] = ("hamming", "hann", "blackman", "boxcar")


def analysis_sizes(
    sample_rate: int, frame_ms: float, hop_ms: float, n_fft: int | None = None
) -> tuple[int, int, int]:
    """Frame length, hop and FFT size in samples.

    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz.
    frame_ms, hop_ms : float
        Frame length and frame advance in milliseconds.
    n_fft : int or None, optional
        FFT size; ``None`` means the smallest power of two that holds a frame.

    Returns
    -------
    tuple of int
        ``(frame_length, hop, n_fft)``.

    Raises
    ------
    ValueError
        If a size rounds to zero samples or ``n_fft`` is shorter than a frame.

    Examples
    --------
    >>> analysis_sizes(16_000, 25.0, 10.0)
    (400, 160, 512)
    """
    frame_length = round(frame_ms * sample_rate / 1000.0)
    hop = round(hop_ms * sample_rate / 1000.0)
    if frame_length < 1 or hop < 1:
        raise ValueError(
            f"frame_ms={frame_ms!r} and hop_ms={hop_ms!r} must each span at least one sample "
            f"at {sample_rate} Hz"
        )
    if n_fft is None:
        n_fft = 1 << (frame_length - 1).bit_length()
    if n_fft < frame_length:
        raise ValueError(f"n_fft={n_fft} is shorter than the {frame_length}-sample frame")
    return frame_length, hop, int(n_fft)


def power_spectrogram(
    x: ArrayLike,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int | None = None,
    window: str = "hamming",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One-sided short-time power spectral density.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Mono waveform.
    sample_rate : int
        Sampling rate in Hz.
    frame_ms, hop_ms : float, optional
        Frame length and advance in milliseconds; only complete frames are kept.
    n_fft : int or None, optional
        FFT size (zero-padding); ``None`` picks the smallest power of two that
        holds a frame.
    window : {"hamming", "hann", "blackman", "boxcar"}, optional
        Analysis window.

    Returns
    -------
    power : numpy.ndarray, shape (n_frames, n_fft // 2 + 1)
        Power spectral density per frame, in units of ``x**2 / Hz``.
    frequencies : numpy.ndarray, shape (n_fft // 2 + 1,)
        Center frequency of each bin in Hz.

    References
    ----------
    .. [1] J. B. Allen, "Short term spectral analysis, synthesis, and modification
       by discrete Fourier transform," IEEE Transactions on Acoustics, Speech, and
       Signal Processing, vol. 25, no. 3, pp. 235-238, 1977.
    .. [2] F. J. Harris, "On the use of windows for harmonic analysis with the
       discrete Fourier transform," Proceedings of the IEEE, vol. 66, no. 1,
       pp. 51-83, 1978.

    Examples
    --------
    >>> t = np.arange(16_000) / 16_000
    >>> power, frequencies = power_spectrogram(np.sin(2 * np.pi * 1000 * t), 16_000)
    >>> power.shape, float(frequencies[power.mean(axis=0).argmax()])
    ((98, 257), 1000.0)
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a one-dimensional waveform, got shape {signal.shape}")
    if window not in WINDOW_NAMES:
        raise ValueError(f"window={window!r} is not supported; choose one of {list(WINDOW_NAMES)}")
    frame_length, hop, n_fft = analysis_sizes(sample_rate, frame_ms, hop_ms, n_fft)
    taps = resolve_window(window, frame_length)
    frames = speechdsp.enframe(signal, frame_length, hop, taps)
    spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)
    power = np.abs(spectrum) ** 2 / (sample_rate * float(np.sum(taps**2)))
    last = power.shape[1] - 1 if n_fft % 2 == 0 else power.shape[1]
    power[:, 1:last] *= 2.0
    return power, np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)


def band_weights(
    scale: str, n_bands: int, sample_rate: int, n_fft: int, fmax_hz: float
) -> NDArray[np.float64]:
    """Matrix that pools FFT bins into frequency bands.

    Parameters
    ----------
    scale : {"linear", "mel"}
        ``linear``: equal-width bands over ``[0, fmax_hz]``; each row averages
        the bins it overlaps, weighted by the overlapping width, and sums to 1.
        ``mel``: the triangular filters of :func:`speechdsp.mel_filterbank`.
    n_bands : int
        Number of bands (rows).
    sample_rate : int
        Sampling rate in Hz.
    n_fft : int
        FFT size the matrix is applied to.
    fmax_hz : float
        Upper edge of the highest band; at most ``sample_rate / 2``.

    Returns
    -------
    numpy.ndarray, shape (n_bands, n_fft // 2 + 1)
        ``power @ weights.T`` gives the band energies; row 0 is the lowest band.

    References
    ----------
    .. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric representations
       for monosyllabic word recognition in continuously spoken sentences," IEEE
       Transactions on Acoustics, Speech, and Signal Processing, vol. 28, no. 4,
       pp. 357-366, 1980.
    .. [2] S. S. Stevens, J. Volkmann, and E. B. Newman, "A scale for the
       measurement of the psychological magnitude pitch," The Journal of the
       Acoustical Society of America, vol. 8, no. 3, pp. 185-190, 1937.

    Examples
    --------
    >>> weights = band_weights("linear", 40, 16_000, 512, 8000.0)
    >>> weights.shape, bool(np.allclose(weights.sum(axis=1), 1.0))
    ((40, 257), True)
    """
    if scale not in SPECTROGRAM_SCALES:
        raise ValueError(
            f"scale={scale!r} is not supported; choose one of {list(SPECTROGRAM_SCALES)}"
        )
    if n_bands < 1:
        raise ValueError(f"n_bands must be at least 1, got {n_bands!r}")
    nyquist = sample_rate / 2.0
    if not 0.0 < fmax_hz <= nyquist:
        raise ValueError(f"fmax_hz must lie in (0, {nyquist:g}], got {fmax_hz!r}")
    if scale == "mel":
        return speechdsp.mel_filterbank(sample_rate, n_fft, n_mels=n_bands, fmin=0.0, fmax=fmax_hz)
    bin_width = sample_rate / n_fft
    centers = np.arange(n_fft // 2 + 1) * bin_width
    low = np.clip(centers - bin_width / 2.0, 0.0, nyquist)
    high = np.clip(centers + bin_width / 2.0, 0.0, nyquist)
    edges = np.linspace(0.0, fmax_hz, n_bands + 1)
    overlap = np.minimum(high[None, :], edges[1:, None]) - np.maximum(
        low[None, :], edges[:-1, None]
    )
    overlap = np.clip(overlap, 0.0, None)
    return overlap / overlap.sum(axis=1, keepdims=True)


def band_frequencies(scale: str, n_bands: int, fmax_hz: float) -> NDArray[np.float64]:
    """Center frequency of every band in Hz, lowest first.

    Examples
    --------
    >>> band_frequencies("linear", 4, 8000.0).tolist()
    [1000.0, 3000.0, 5000.0, 7000.0]
    """
    if scale not in SPECTROGRAM_SCALES:
        raise ValueError(
            f"scale={scale!r} is not supported; choose one of {list(SPECTROGRAM_SCALES)}"
        )
    if scale == "mel":
        edges = speechdsp.mel_to_hz(
            np.linspace(0.0, float(speechdsp.hz_to_mel(fmax_hz)), n_bands + 2)
        )
        return np.asarray(edges[1:-1], dtype=np.float64)
    edges = np.linspace(0.0, fmax_hz, n_bands + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def to_decibels(energy: ArrayLike, top_db: float = 80.0) -> NDArray[np.float64]:
    """Levels in dB relative to the largest value, floored at ``-top_db``.

    The floor is relative, so a quiet recording keeps the same dynamic range
    as a loud one; an all-zero input (a silent segment) maps to ``-top_db``
    everywhere. The conversion is :func:`speechdsp.power_to_db`.

    Raises
    ------
    ValueError
        If ``top_db`` is not positive, or ``energy`` holds a negative, NaN or
        infinite value.

    Examples
    --------
    >>> to_decibels([1.0, 0.1, 0.0], top_db=60.0).tolist()
    [0.0, -10.0, -60.0]
    >>> to_decibels([1e-9, 1e-10, 1e-20], top_db=60.0).tolist()
    [0.0, -10.0, -60.0]
    """
    return speechdsp.power_to_db(np.asarray(energy, dtype=np.float64), float(top_db))


def fit_frames(levels: ArrayLike, n_frames: int, fill: float = 0.0) -> NDArray[np.float64]:
    """Linearly interpolate the last axis to exactly ``n_frames`` columns.

    Parameters
    ----------
    levels : array_like, shape (..., n_in)
        Values per frame.
    n_frames : int
        Number of output columns.
    fill : float, optional
        Value used when there is no input frame at all.

    Returns
    -------
    numpy.ndarray, shape (..., n_frames)
        Unchanged when ``n_in == n_frames``.

    Examples
    --------
    >>> fit_frames([[0.0, 10.0]], 3).tolist()
    [[0.0, 5.0, 10.0]]
    """
    values = np.asarray(levels, dtype=np.float64)
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames!r}")
    n_in = values.shape[-1]
    if n_in == n_frames:
        return values.copy()
    if n_in == 0:
        return np.full((*values.shape[:-1], n_frames), float(fill))
    position = np.linspace(0.0, n_in - 1, n_frames)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, n_in - 1)
    fraction = position - lower
    return values[..., lower] * (1.0 - fraction) + values[..., upper] * fraction


def quantize(levels_db: ArrayLike, top_db: float = 80.0) -> NDArray[np.uint8]:
    """Map ``[-top_db, 0]`` dB linearly onto the grey levels 0-255.

    Examples
    --------
    >>> quantize([-80.0, -40.0, 0.0]).tolist()
    [0, 128, 255]
    """
    values = np.asarray(levels_db, dtype=np.float64)
    scaled = (values + float(top_db)) * (255.0 / float(top_db))
    return np.round(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)


def spectrogram_images(
    segments: ArrayLike, sample_rate: int, config: SpectrogramConfig | None = None
) -> NDArray[np.uint8]:
    """Grey-scale spectrogram images of equal-length segments.

    Parameters
    ----------
    segments : array_like, shape (n_segments, n_samples)
        One segment per row, at ``sample_rate``.
    sample_rate : int
        Sampling rate in Hz.
    config : SpectrogramConfig, optional
        The ``spectrogram`` section; ``None`` uses the defaults.

    Returns
    -------
    numpy.ndarray of uint8, shape (n_segments, n_bands, n_frames)
        Row 0 of every image is the lowest band, column 0 the earliest frame;
        255 is the segment's spectral peak and 0 is ``top_db`` below it.

    References
    ----------
    .. [1] T. N. Sainath and C. Parada, "Convolutional neural networks for
       small-footprint keyword spotting," in Proc. Interspeech 2015, pp. 1478-1482.
    .. [2] P. Warden, "Speech Commands: A dataset for limited-vocabulary speech
       recognition," arXiv:1804.03209, 2018.

    Examples
    --------
    >>> t = np.arange(16_000) / 16_000
    >>> rows = np.stack([np.sin(2 * np.pi * 1100 * t), np.sin(2 * np.pi * 3100 * t)])
    >>> images = spectrogram_images(rows, 16_000)
    >>> images.shape, images.mean(axis=2).argmax(axis=1).tolist()
    ((2, 40, 98), [5, 15])
    """
    from .config import SpectrogramConfig  # deferred: config imports this module

    spectrogram_config = config if config is not None else SpectrogramConfig()
    batch = np.asarray(segments, dtype=np.float64)
    if batch.ndim != 2:
        raise ValueError(f"expected segments of shape (n_segments, n_samples), got {batch.shape}")
    _frame_length, _hop, n_fft = analysis_sizes(
        sample_rate,
        spectrogram_config.frame_ms,
        spectrogram_config.hop_ms,
        spectrogram_config.n_fft,
    )
    weights = band_weights(
        spectrogram_config.scale,
        spectrogram_config.n_bands,
        sample_rate,
        n_fft,
        spectrogram_config.fmax_hz,
    )
    images = np.empty(
        (batch.shape[0], spectrogram_config.n_bands, spectrogram_config.n_frames), dtype=np.uint8
    )
    for index, segment in enumerate(batch):
        power, _ = power_spectrogram(
            segment,
            sample_rate,
            frame_ms=spectrogram_config.frame_ms,
            hop_ms=spectrogram_config.hop_ms,
            n_fft=n_fft,
            window=spectrogram_config.window,
        )
        levels = to_decibels(power @ weights.T, spectrogram_config.top_db).T  # (n_bands, n_in)
        levels = fit_frames(levels, spectrogram_config.n_frames, fill=-spectrogram_config.top_db)
        images[index] = quantize(levels, spectrogram_config.top_db)
    return images


def spectrogram_image(
    x: ArrayLike, sample_rate: int, config: SpectrogramConfig | None = None
) -> NDArray[np.uint8]:
    """Grey-scale spectrogram image of one segment.

    Parameters
    ----------
    x : array_like, shape (n_samples,)
        Mono segment at ``sample_rate``.
    sample_rate : int
        Sampling rate in Hz.
    config : SpectrogramConfig, optional
        The ``spectrogram`` section; ``None`` uses the defaults.

    Returns
    -------
    numpy.ndarray of uint8, shape (n_bands, n_frames)
        See :func:`spectrogram_images`.

    Examples
    --------
    >>> t = np.arange(16_000) / 16_000
    >>> image = spectrogram_image(np.sin(2 * np.pi * 1100 * t), 16_000)
    >>> image.shape, image.dtype.name, int(image.max()), int(image.mean(axis=1).argmax())
    ((40, 98), 'uint8', 255, 5)
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"expected a one-dimensional segment, got shape {signal.shape}")
    return spectrogram_images(signal[None, :], sample_rate, config)[0]
