"""Synthetic sustained vowels with three distinguishable voice qualities.

The generator writes a dataset in the same on-disk layout as real data (WAV
files plus a ``dataset.csv`` index), so every command can be tried and tested
without a single recording. Each file is a sustained /a/ built from a
harmonic source shaped by four vowel formants (resonance peaks):

* ``modal`` -- a steady harmonic voice with a moderate spectral tilt and a very
  low noise floor;
* ``breathy`` -- a steeper spectral tilt (weak upper harmonics) and strong
  aspiration noise above about 2 kHz;
* ``rough`` -- a 4-6 Hz tremor of both pitch and loudness plus subharmonics at
  0.5 and 1.5 times the fundamental frequency.

I picked these qualities because each leaves a different mark on a
spectrogram (high-frequency noise, temporal modulation, extra spectral
lines); they are not models of any clinical condition. A *severity* between
0 and 1 moves every parameter from the modal voice (0) to the full quality
(1), so mildly affected speakers overlap with modal ones and the task is not
trivial. Each speaker gets its own fundamental frequency, formant scaling,
tremor rate, severity (always 0 for ``modal``) and tilt and noise offsets.
All recordings of a speaker share them, so the cross-validation folds have
to be grouped by speaker; only the fundamental frequency is drawn again for
each take, within 3 % of the speaker's value. Recordings alternate between
the configured sampling rates and start and end with near-silence, which
exercises resampling and silence trimming too. All randomness comes from
one generator, so a seed reproduces the dataset byte for byte.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import speechdsp
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

if TYPE_CHECKING:
    from .config import SyntheticConfig

_LOGGER = logging.getLogger(__name__)

#: Voice qualities the generator can produce (``synthetic.class_names``).
SYNTHETIC_CLASSES: tuple[str, ...] = ("breathy", "modal", "rough")

#: Formant frequencies (Hz) and bandwidths (Hz) of a sustained /a/.
_FORMANTS_HZ = np.array([730.0, 1090.0, 2440.0, 3400.0])
_BANDWIDTHS_HZ = np.array([90.0, 110.0, 170.0, 250.0])
#: Highest harmonic frequency the source contains.
_HARMONIC_CEILING_HZ = 7600.0
#: Slow pitch drift present in every voice (relative depth).
_DRIFT_DEPTH = 0.004
#: Length of the raised-cosine fade at both ends of the voiced part.
_FADE_S = 0.03
#: Level of the near-silence around the voiced part, relative to the voice.
_SILENCE_DB = -55.0


@dataclass(frozen=True)
class _Quality:
    """Generative parameters of one voice quality at full severity."""

    spectral_tilt: float
    noise_db: float
    noise_highpass_hz: float | None
    tremor_depth: float
    loudness_tremor: float
    subharmonic: float


_QUALITIES: dict[str, _Quality] = {
    "breathy": _Quality(1.8, -4.0, 2000.0, 0.0, 0.0, 0.0),
    "modal": _Quality(1.1, -35.0, None, 0.0, 0.0, 0.0),
    "rough": _Quality(1.3, -25.0, None, 0.05, 0.5, 0.6),
}


def _blend(modal: _Quality, target: _Quality, severity: float) -> _Quality:
    """Parameters a fraction ``severity`` of the way from ``modal`` to ``target``."""

    def between(start: float, end: float) -> float:
        return start + severity * (end - start)

    return _Quality(
        spectral_tilt=between(modal.spectral_tilt, target.spectral_tilt),
        noise_db=between(modal.noise_db, target.noise_db),
        noise_highpass_hz=target.noise_highpass_hz,
        tremor_depth=between(modal.tremor_depth, target.tremor_depth),
        loudness_tremor=between(modal.loudness_tremor, target.loudness_tremor),
        subharmonic=between(modal.subharmonic, target.subharmonic),
    )


def _formant_gain(frequencies: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """Sum of resonance peaks (Lorentzian shape) of a vowel, plus a small floor."""
    centers = _FORMANTS_HZ * scale
    detuning = (frequencies[:, None] - centers[None, :]) / (_BANDWIDTHS_HZ[None, :] / 2.0)
    return np.sum(1.0 / (1.0 + detuning**2), axis=1) + 0.05


def make_sustained_vowel(
    quality: str,
    sample_rate: int = 16_000,
    duration_s: float = 1.0,
    *,
    f0_hz: float = 150.0,
    formant_scale: float = 1.0,
    tremor_rate_hz: float = 5.0,
    severity: float = 1.0,
    tilt_offset: float = 0.0,
    noise_offset_db: float = 0.0,
    silence_s: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Synthesize one sustained vowel of the given voice quality.

    Parameters
    ----------
    quality : {"breathy", "modal", "rough"}
        Voice quality (see the module docstring).
    sample_rate : int, optional
        Sampling rate in Hz.
    duration_s : float, optional
        Length of the voiced part in seconds.
    f0_hz : float, optional
        Mean fundamental frequency in Hz.
    formant_scale : float, optional
        Multiplies every formant frequency; stands in for vocal tract length.
    tremor_rate_hz : float, optional
        Rate in Hz of the pitch and loudness tremor of the ``rough`` quality.
    severity : float, optional
        From 0 (a modal voice) to 1 (the full quality); every parameter of
        the quality is interpolated between the two.
    tilt_offset, noise_offset_db : float, optional
        Added to the harmonic roll-off exponent and to the noise level (dB
        relative to the harmonic part), for speaker-to-speaker variation.
    silence_s : float, optional
        Seconds of near-silence (noise 55 dB below the voice) added before
        and after the voiced part.
    rng : numpy.random.Generator or int or None, optional
        Source of the noise and of the modulation phases.

    Returns
    -------
    numpy.ndarray, shape (n_voiced + 2 * n_silence,)
        Waveform with a peak amplitude of 0.8, where
        ``n_voiced = round(duration_s * sample_rate)`` and
        ``n_silence = round(silence_s * sample_rate)``.

    Raises
    ------
    ValueError
        For an unknown quality or non-positive sizes.

    Examples
    --------
    >>> voice = make_sustained_vowel("breathy", 16_000, 0.5, rng=0)
    >>> voice.shape, round(float(np.abs(voice).max()), 6)
    ((8000,), 0.8)
    """
    rng = np.random.default_rng(rng)
    if quality not in _QUALITIES:
        raise ValueError(
            f"unknown voice quality {quality!r}; choose one of {list(SYNTHETIC_CLASSES)}"
        )
    if sample_rate <= 0 or duration_s <= 0 or f0_hz <= 0:
        raise ValueError(
            f"sample_rate, duration_s and f0_hz must be positive, got "
            f"{sample_rate!r}, {duration_s!r}, {f0_hz!r}"
        )
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must lie in [0, 1], got {severity!r}")
    params = _blend(_QUALITIES["modal"], _QUALITIES[quality], severity)
    tilt = params.spectral_tilt + tilt_offset
    level_db = params.noise_db + noise_offset_db
    n_voiced = round(duration_s * sample_rate)
    t = np.arange(n_voiced) / sample_rate

    drift_rate, drift_phase, tremor_phase = rng.uniform([0.4, 0, 0], [1.5, 2 * np.pi, 2 * np.pi])
    tremor = np.sin(2 * np.pi * tremor_rate_hz * t + tremor_phase)
    relative_pitch = (
        _DRIFT_DEPTH * np.sin(2 * np.pi * drift_rate * t + drift_phase)
        + params.tremor_depth * tremor
    )
    phase = 2 * np.pi * np.cumsum(f0_hz * (1.0 + relative_pitch)) / sample_rate

    ceiling = min(_HARMONIC_CEILING_HZ, 0.45 * sample_rate)
    orders = np.arange(1, max(1, int(ceiling // (f0_hz * (1.0 + params.tremor_depth)))) + 1)
    amplitudes = orders.astype(np.float64) ** -tilt * _formant_gain(orders * f0_hz, formant_scale)
    voiced = np.zeros(n_voiced)
    for order, amplitude in zip(orders, amplitudes, strict=True):
        voiced += amplitude * np.sin(order * phase)
    if params.subharmonic > 0:
        voiced += (
            params.subharmonic * amplitudes[0] * (np.sin(0.5 * phase) + 0.5 * np.sin(1.5 * phase))
        )
    voiced *= 1.0 + params.loudness_tremor * tremor

    noise = rng.standard_normal(n_voiced)
    if params.noise_highpass_hz is not None:
        sos = butter(4, params.noise_highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        noise = sosfilt(sos, noise)
    voiced_rms = float(np.sqrt(np.mean(voiced**2)))
    noise *= voiced_rms * 10.0 ** (level_db / 20.0) / float(np.sqrt(np.mean(noise**2)))
    voice = voiced + noise

    n_fade = min(round(_FADE_S * sample_rate), n_voiced // 2)
    if n_fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_fade) / n_fade)
        voice[:n_fade] *= ramp
        voice[n_voiced - n_fade :] *= ramp[::-1]

    n_silence = round(silence_s * sample_rate)
    floor = voiced_rms * 10.0 ** (_SILENCE_DB / 20.0)
    padding = floor * rng.standard_normal((2, n_silence))
    signal = np.concatenate([padding[0], voice, padding[1]])
    return 0.8 * signal / float(np.max(np.abs(signal)))


def write_synthetic_dataset(
    folder: str | Path,
    config: SyntheticConfig | None = None,
    *,
    rng: np.random.Generator | int | None = None,
) -> Path:
    """Write synthetic recordings and their ``dataset.csv`` index.

    Files are named ``<label>/<label>_<NNN>.wav`` (``NNN`` counts the
    recordings of a class from ``000``); the index has the columns ``path``,
    ``label`` and ``group`` (one group per synthetic speaker).

    Parameters
    ----------
    folder : str or pathlib.Path
        Destination folder, created if needed.
    config : SyntheticConfig, optional
        Sizes and generative parameters; ``None`` uses the defaults.
    rng : numpy.random.Generator or int or None, optional
        Source of all randomness; the same seed writes identical files.

    Returns
    -------
    pathlib.Path
        Path of the written ``dataset.csv``.

    Examples
    --------
    >>> import tempfile
    >>> from voice_spectral_classifier.config import SyntheticConfig
    >>> small = SyntheticConfig(n_speakers_per_class=(1, 1, 2), duration_s=0.5)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     index = pd.read_csv(write_synthetic_dataset(folder, small, rng=0))
    >>> len(index), index["path"].iloc[0], index["group"].nunique()
    (8, 'breathy/breathy_000.wav', 4)
    """
    from .config import SyntheticConfig  # deferred: config imports this module

    synthetic_config = config if config is not None else SyntheticConfig()
    rng = np.random.default_rng(rng)
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    speaker = 0
    for label, n_speakers in zip(
        synthetic_config.class_names, synthetic_config.n_speakers_per_class, strict=True
    ):
        take = 0
        for _ in range(n_speakers):
            f0_hz = rng.uniform(110.0, 230.0)
            formant_scale = rng.uniform(0.9, 1.12)
            tremor_rate_hz = rng.uniform(4.0, 6.0)
            severity = rng.uniform(*synthetic_config.severity_range) if label != "modal" else 0.0
            tilt_offset = rng.uniform(-0.15, 0.15)
            noise_offset_db = rng.uniform(-3.0, 3.0)
            for _ in range(synthetic_config.n_recordings_per_speaker):
                rate = synthetic_config.sample_rates[len(rows) % len(synthetic_config.sample_rates)]
                signal = make_sustained_vowel(
                    label,
                    rate,
                    synthetic_config.duration_s,
                    f0_hz=f0_hz * rng.uniform(0.97, 1.03),
                    formant_scale=formant_scale,
                    tremor_rate_hz=tremor_rate_hz,
                    severity=severity,
                    tilt_offset=tilt_offset,
                    noise_offset_db=noise_offset_db,
                    silence_s=synthetic_config.silence_s,
                    rng=rng,
                )
                relative = f"{label}/{label}_{take:03d}.wav"
                speechdsp.write_wav(root / relative, signal * rng.uniform(0.6, 1.0), rate)
                rows.append({"path": relative, "label": label, "group": f"speaker_{speaker:03d}"})
                take += 1
            speaker += 1
    index = root / "dataset.csv"
    pd.DataFrame(rows, columns=["path", "label", "group"]).to_csv(
        index, index=False, lineterminator="\n"
    )
    _LOGGER.info("wrote %d synthetic recordings of %d speakers to %s", len(rows), speaker, root)
    return index
