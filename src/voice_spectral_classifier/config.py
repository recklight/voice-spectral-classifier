"""Run configuration: frozen dataclasses validated on construction, stored as YAML.

There is one dataclass per YAML section (``data``, ``synthetic``,
``segmentation``, ``spectrogram``, ``model``, ``evaluation``, ``output``).
Each section validates itself in ``__post_init__`` and raises
:class:`~voice_spectral_classifier.errors.ConfigError` naming the dotted key,
the value it got and what is allowed; :class:`Config` checks the settings that
span two sections, such as ``spectrogram.fmax_hz`` against the Nyquist
frequency of ``segmentation.sample_rate``. Unknown keys are rejected rather
than ignored, so a typo cannot silently fall back to a default.

``features.npz`` keeps the configuration ``prepare`` used, and ``evaluate``
and ``train`` stop when the current ``segmentation`` or ``spectrogram``
values differ from it (:data:`FEATURE_SECTIONS`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .models import CLASS_WEIGHTS, DEVICES, MODEL_NAMES
from .spectrogram import SPECTROGRAM_SCALES, WINDOW_NAMES
from .synthetic import SYNTHETIC_CLASSES

#: Image formats accepted by ``output.figure_format``.
FIGURE_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def _choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ConfigError(f"{name} must be strictly positive, got {value!r}")


def _number(name: str, value: object) -> None:
    """Reject booleans, strings, NaN and infinities before any comparison.

    NaN slips through every ``<`` and ``>`` check, and an infinity (``.inf`` in
    YAML, ``--set key=Infinity``) passes them and only fails later, when a
    length in seconds is rounded to samples.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")


def _integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer of at least {minimum}, got {value!r}")


def _flag(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives.

    Attributes
    ----------
    dataset : str
        Dataset index (CSV, see :mod:`voice_spectral_classifier.dataset`). A placeholder until you
        point it at your own data.
    """

    dataset: str = "data/dataset.csv"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ConfigError(f"data.dataset must be a non-empty path, got {self.dataset!r}")


@dataclass(frozen=True)
class SyntheticConfig:
    """Size and generative parameters of the synthetic dataset.

    Attributes
    ----------
    class_names : tuple of str
        Voice qualities to synthesize, one class each; see
        :data:`~voice_spectral_classifier.synthetic.SYNTHETIC_CLASSES`.
    n_speakers_per_class : tuple of int
        Number of synthetic speakers of each class, in the order of
        ``class_names``. Unequal counts reproduce a class imbalance.
    n_recordings_per_speaker : int
        Recordings per speaker; they share the speaker's voice, so they must
        stay in the same cross-validation fold.
    severity_range : tuple of float
        Each non-modal speaker draws a severity uniformly from this range:
        0 sounds like the modal voice, 1 is the full voice quality.
    duration_s : float
        Length of the voiced part of every recording.
    silence_s : float
        Near-silence before and after the voiced part.
    sample_rates : tuple of int
        Recording rates in Hz, used in turn, so the pipeline has to resample.
    """

    class_names: tuple[str, ...] = ("breathy", "modal", "rough")
    n_speakers_per_class: tuple[int, ...] = (8, 10, 20)
    n_recordings_per_speaker: int = 2
    severity_range: tuple[float, float] = (0.2, 1.0)
    duration_s: float = 2.5
    silence_s: float = 0.25
    sample_rates: tuple[int, ...] = (22_050, 24_000)

    def __post_init__(self) -> None:
        if not isinstance(self.class_names, tuple) or len(self.class_names) < 2:
            raise ConfigError(
                f"synthetic.class_names must list at least two classes, got {self.class_names!r}"
            )
        for name in self.class_names:
            _choice("synthetic.class_names", name, SYNTHETIC_CLASSES)
        if len(set(self.class_names)) != len(self.class_names):
            raise ConfigError(f"synthetic.class_names repeats a class: {list(self.class_names)}")
        counts = self.n_speakers_per_class
        if not isinstance(counts, tuple) or len(counts) != len(self.class_names):
            raise ConfigError(
                "synthetic.n_speakers_per_class must give one count per class in "
                f"synthetic.class_names ({len(self.class_names)}), got {counts!r}"
            )
        for count in self.n_speakers_per_class:
            _integer("synthetic.n_speakers_per_class", count, 1)
        _integer("synthetic.n_recordings_per_speaker", self.n_recordings_per_speaker, 1)
        if not isinstance(self.severity_range, tuple) or len(self.severity_range) != 2:
            raise ConfigError(
                f"synthetic.severity_range must be [low, high], got {self.severity_range!r}"
            )
        for value in self.severity_range:
            _number("synthetic.severity_range", value)
        low, high = self.severity_range
        if not 0.0 <= low <= high <= 1.0:
            raise ConfigError(
                "synthetic.severity_range must satisfy 0 <= low <= high <= 1, "
                f"got {list(self.severity_range)}"
            )
        _number("synthetic.duration_s", self.duration_s)
        _positive("synthetic.duration_s", self.duration_s)
        _number("synthetic.silence_s", self.silence_s)
        if self.silence_s < 0:
            raise ConfigError(f"synthetic.silence_s must not be negative, got {self.silence_s!r}")
        if not isinstance(self.sample_rates, tuple) or not self.sample_rates:
            raise ConfigError(
                f"synthetic.sample_rates must list at least one rate, got {self.sample_rates!r}"
            )
        for rate in self.sample_rates:
            _integer("synthetic.sample_rates", rate, 16_000)


@dataclass(frozen=True)
class SegmentationConfig:
    """Resampling, silence trimming and fixed-length segmentation.

    Attributes
    ----------
    sample_rate : int
        Every recording is resampled to this rate (Hz) before anything else.
    trim_silence : bool
        Remove leading and trailing silence with an endpoint detector first.
    segment_s : float
        Segment length in seconds.
    hop_s : float
        Segment advance in seconds; smaller than ``segment_s`` for overlap.
    min_fraction : float
        A trailing piece is zero-padded and kept when it fills at least this
        fraction of a segment, in ``(0, 1]``.
    max_segments : int or None
        Keep at most this many segments per recording, spread evenly over it
        (the middle one when only one is kept); ``None`` keeps them all.
    """

    sample_rate: int = 16_000
    trim_silence: bool = True
    segment_s: float = 1.0
    hop_s: float = 1.0
    min_fraction: float = 0.5
    max_segments: int | None = None

    def __post_init__(self) -> None:
        _integer("segmentation.sample_rate", self.sample_rate, 1000)
        _flag("segmentation.trim_silence", self.trim_silence)
        for name in ("segment_s", "hop_s", "min_fraction"):
            _number(f"segmentation.{name}", getattr(self, name))
            _positive(f"segmentation.{name}", getattr(self, name))
        if self.min_fraction > 1:
            raise ConfigError(
                f"segmentation.min_fraction must lie in (0, 1], got {self.min_fraction!r}"
            )
        if self.max_segments is not None:
            _integer("segmentation.max_segments", self.max_segments, 1)

    @property
    def segment_length(self) -> int:
        """Samples per segment."""
        return round(self.segment_s * self.sample_rate)


@dataclass(frozen=True)
class SpectrogramConfig:
    """Spectrogram images: analysis, band layout and quantization.

    Attributes
    ----------
    scale : str
        ``linear`` (equal-width bands) or ``mel`` (mel filterbank).
    n_bands : int
        Image rows (frequency bands between 0 Hz and ``fmax_hz``).
    n_frames : int
        Image columns; the frame sequence is interpolated to this length.
    frame_ms : float
        Analysis frame length in milliseconds.
    hop_ms : float
        Frame advance in milliseconds.
    n_fft : int or None
        FFT size; ``None`` uses the smallest power of two not shorter than the
        frame.
    window : str
        Analysis window: ``hamming``, ``hann``, ``blackman`` or ``boxcar``.
    fmax_hz : float
        Upper edge of the highest band; at most half of ``segmentation.sample_rate``.
    top_db : float
        Dynamic range: grey level 0 is this many dB below the segment's peak.
    """

    scale: str = "linear"
    n_bands: int = 40
    n_frames: int = 98
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int | None = None
    window: str = "hamming"
    fmax_hz: float = 8000.0
    top_db: float = 80.0

    def __post_init__(self) -> None:
        _choice("spectrogram.scale", self.scale, SPECTROGRAM_SCALES)
        _integer("spectrogram.n_bands", self.n_bands, 1)
        _integer("spectrogram.n_frames", self.n_frames, 1)
        for name in ("frame_ms", "hop_ms", "fmax_hz", "top_db"):
            _number(f"spectrogram.{name}", getattr(self, name))
            _positive(f"spectrogram.{name}", getattr(self, name))
        if self.n_fft is not None:
            _integer("spectrogram.n_fft", self.n_fft, 2)
        _choice("spectrogram.window", self.window, WINDOW_NAMES)

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(n_bands, n_frames)``."""
        return (self.n_bands, self.n_frames)


@dataclass(frozen=True)
class ModelConfig:
    """Classifier and training hyperparameters.

    Attributes
    ----------
    name : str
        ``cnn`` (needs the ``dl`` extra), ``mlp`` or ``logistic``.
    n_epochs : int
        Training epochs of the CNN. For the MLP it is an upper limit:
        scikit-learn stops once the training loss has not improved by 1e-4
        for 10 epochs in a row.
    learning_rate : float
        Adam step size (CNN and MLP).
    batch_size : int
        Mini-batch size (CNN and MLP).
    weight_decay : float
        L2 penalty: Adam weight decay of the CNN, ``alpha`` of the MLP.
    class_weight : str or None
        ``balanced`` counters class imbalance (weighted loss for the CNN and
        logistic regression, minority oversampling for the MLP); ``None``
        trains on the data as it is.
    dropout : float
        Dropout probability before the deeper convolution blocks of the CNN.
    hidden_units : tuple of int
        Hidden layer sizes of the MLP.
    n_components : int or None
        Principal components kept before the MLP or logistic regression;
        ``None`` skips the projection.
    logistic_c : float
        Inverse L2 regularization strength of logistic regression.
    device : str
        Torch device for the CNN: ``cpu`` or ``cuda``.
    """

    name: str = "cnn"
    n_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 60
    weight_decay: float = 1e-4
    class_weight: str | None = "balanced"
    dropout: float = 0.2
    hidden_units: tuple[int, ...] = (128,)
    n_components: int | None = None
    logistic_c: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        _choice("model.name", self.name, MODEL_NAMES)
        _integer("model.n_epochs", self.n_epochs, 1)
        _number("model.learning_rate", self.learning_rate)
        _positive("model.learning_rate", self.learning_rate)
        _integer("model.batch_size", self.batch_size, 1)
        _number("model.weight_decay", self.weight_decay)
        if self.weight_decay < 0:
            raise ConfigError(f"model.weight_decay must not be negative, got {self.weight_decay!r}")
        if self.class_weight is not None:
            _choice("model.class_weight", self.class_weight, CLASS_WEIGHTS)
        _number("model.dropout", self.dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model.dropout must lie in [0, 1), got {self.dropout!r}")
        if not isinstance(self.hidden_units, tuple) or not self.hidden_units:
            raise ConfigError(
                f"model.hidden_units must list at least one layer size, got {self.hidden_units!r}"
            )
        for units in self.hidden_units:
            _integer("model.hidden_units", units, 1)
        if self.n_components is not None:
            _integer("model.n_components", self.n_components, 1)
        _number("model.logistic_c", self.logistic_c)
        _positive("model.logistic_c", self.logistic_c)
        _choice("model.device", self.device, DEVICES)


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation.

    Attributes
    ----------
    n_splits : int
        Number of folds; stratified, and grouped by the index's ``group``
        column (by recording when it has none), so the segments of one
        speaker never end up on both sides of a split.
    """

    n_splits: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int):
            raise ConfigError(f"evaluation.n_splits must be an integer, got {self.n_splits!r}")
        if self.n_splits < 2:
            raise ConfigError(f"evaluation.n_splits must be at least 2, got {self.n_splits!r}")


@dataclass(frozen=True)
class OutputConfig:
    """Where and how results are written.

    Attributes
    ----------
    root : str
        Run folders are created here when ``--out`` is not given.
    figure_format : str
        ``png``, ``pdf`` or ``svg``.
    dpi : int
        Resolution of raster figures.
    """

    root: str = "outputs"
    figure_format: str = "png"
    dpi: int = 150

    def __post_init__(self) -> None:
        _choice("output.figure_format", self.figure_format, FIGURE_FORMATS)
        _number("output.dpi", self.dpi)
        _positive("output.dpi", self.dpi)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "synthetic": SyntheticConfig,
    "segmentation": SegmentationConfig,
    "spectrogram": SpectrogramConfig,
    "model": ModelConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
}

#: Sections whose values change what ``prepare`` writes.
FEATURE_SECTIONS: tuple[str, ...] = ("segmentation", "spectrogram")


def _tuplify(value: Any) -> Any:
    """YAML lists become tuples, so configurations stay hashable."""
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value


def _listify(value: Any) -> Any:
    """Tuples become lists again, for YAML and JSON."""
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_listify(item) for item in value]
    return value


def _build_section(name: str, cls: type, raw: Any) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"section {name!r} must be a mapping, got {type(raw).__name__}")
    known = sorted(f.name for f in dataclasses.fields(cls))
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in section {name!r}; valid keys are {known}")
    try:
        return cls(**{key: _tuplify(value) for key, value in raw.items()})
    except TypeError as exc:
        raise ConfigError(f"section {name!r}: {exc}") from None


@dataclass(frozen=True)
class Config:
    """Complete configuration of a run.

    Attributes
    ----------
    seed : int
        Master seed: splits, model initialization and synthetic data derive from it.
    data, synthetic, segmentation, spectrogram, model, evaluation, output
        See the section classes.

    Examples
    --------
    >>> config = Config().override({"evaluation.n_splits": 3})
    >>> config.evaluation.n_splits, Config.from_dict(config.to_dict()) == config
    (3, True)
    """

    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {self.seed!r}")
        rate = self.segmentation.sample_rate
        if self.spectrogram.fmax_hz > rate / 2:
            raise ConfigError(
                f"spectrogram.fmax_hz={self.spectrogram.fmax_hz!r} is above the Nyquist "
                f"frequency of segmentation.sample_rate={rate!r} ({rate / 2:g} Hz)"
            )
        frame_length = round(self.spectrogram.frame_ms * rate / 1000.0)
        hop_length = round(self.spectrogram.hop_ms * rate / 1000.0)
        if frame_length < 1 or hop_length < 1:
            raise ConfigError(
                f"spectrogram.frame_ms={self.spectrogram.frame_ms!r} and "
                f"spectrogram.hop_ms={self.spectrogram.hop_ms!r} must each span at least one "
                f"sample at segmentation.sample_rate={rate!r}"
            )
        if frame_length > self.segmentation.segment_length:
            raise ConfigError(
                f"spectrogram.frame_ms={self.spectrogram.frame_ms!r} is longer than "
                f"segmentation.segment_s={self.segmentation.segment_s!r}"
            )
        if self.spectrogram.n_fft is not None and self.spectrogram.n_fft < frame_length:
            raise ConfigError(
                f"spectrogram.n_fft={self.spectrogram.n_fft!r} is shorter than the "
                f"{frame_length}-sample frame set by spectrogram.frame_ms and "
                "segmentation.sample_rate"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Config:
        """Build a configuration from a nested mapping, rejecting unknown keys."""
        data = dict(raw or {})
        known = sorted(f.name for f in dataclasses.fields(cls))
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ConfigError(f"unknown setting(s) {unknown}; valid top-level keys are {known}")
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name in _SECTIONS:
                kwargs[name] = _build_section(name, _SECTIONS[name], value)
            else:
                kwargs[name] = _tuplify(value)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file; keys it leaves out keep their defaults."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"configuration file not found: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{source} is not valid YAML: {exc}") from None
        if raw is not None and not isinstance(raw, Mapping):
            raise ConfigError(f"{source} must contain a mapping of settings")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested mapping (lists instead of tuples), ready for YAML or JSON."""
        return _listify(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration as YAML and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
        return destination

    def override(self, updates: Mapping[str, Any]) -> Config:
        """Return a validated copy with dotted keys replaced, e.g. ``{"output.dpi": 300}``."""
        tree = self.to_dict()
        for dotted, value in updates.items():
            *parents, leaf = dotted.split(".")
            node = tree
            for part in parents:
                child = node.get(part)
                if not isinstance(child, dict):
                    valid = sorted(key for key, item in node.items() if isinstance(item, dict))
                    raise ConfigError(
                        f"unknown configuration section in {dotted!r}; valid sections are {valid}"
                    )
                node = child
            if leaf not in node:
                raise ConfigError(
                    f"unknown configuration key {dotted!r}; valid keys here are {sorted(node)}"
                )
            node[leaf] = value
        return Config.from_dict(tree)
