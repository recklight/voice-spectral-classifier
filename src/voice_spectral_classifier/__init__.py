"""Pathological-voice classification from fixed-size spectrogram images.

Recordings are resampled, trimmed and cut into one-second segments. Each
segment becomes a 40 x 98 grey-scale spectrogram image, a small CNN (or a
scikit-learn baseline) classifies the images, and grouped, stratified k-fold
cross-validation reports segment-level and recording-level scores.

Where each step lives:

* settings, validated and stored as YAML: :class:`Config`;
* synthetic sustained vowels with three voice qualities:
  :func:`write_synthetic_dataset`, :func:`make_sustained_vowel`;
* the ``dataset.csv`` index: :func:`load_dataset_index`, :func:`index_audio_folder`;
* resampling, silence trimming and fixed-length segmentation:
  :func:`load_recording`, :func:`split_into_segments`, :func:`segment_recording`;
* short-time power spectra pooled into linear or mel bands and quantized to
  8 bits: :func:`power_spectrogram`, :func:`spectrogram_image`,
  :func:`spectrogram_images`;
* the CNN, whose training and inference need the ``dl`` extra:
  :class:`SpectrogramCNNClassifier`, :func:`feature_map_shapes`;
* the ``cnn``, ``mlp`` and ``logistic`` models and their files:
  :func:`build_model`, :func:`save_model`, :func:`load_model`;
* the steps behind the command line: :func:`prepare_features`,
  :func:`train_model`, :func:`predict_inputs`;
* stratified, grouped k-fold evaluation at segment and recording level:
  :func:`cross_validate`, :class:`CrossValidationResult`;
* figures, each saved with a matching table: :func:`plot_confusion_matrix`,
  :func:`plot_class_spectrograms`.
"""

from __future__ import annotations

from .cnn import CNN_CHANNELS, SpectrogramCNNClassifier, feature_map_shapes, parameter_count
from .config import (
    FEATURE_SECTIONS,
    FIGURE_FORMATS,
    Config,
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    OutputConfig,
    SegmentationConfig,
    SpectrogramConfig,
    SyntheticConfig,
)
from .dataset import (
    DATASET_INDEX_NAME,
    FeatureSet,
    index_audio_folder,
    items_table,
    load_dataset_index,
    load_features,
    save_features,
)
from .errors import (
    ConfigError,
    DatasetError,
    MissingDependencyError,
    ModelFileError,
    require_module,
)
from .evaluation import (
    SCALAR_METRICS,
    CrossValidationResult,
    aggregate_recordings,
    classification_scores,
    cross_validate,
    pooled_report,
)
from .figures import (
    SERIES_COLORS,
    class_spectrograms_table,
    confusion_matrix_table,
    new_figure,
    plot_class_spectrograms,
    plot_confusion_matrix,
    save_figure,
)
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import (
    MODEL_NAMES,
    OversampledClassifier,
    build_model,
    load_model,
    model_filename,
    save_model,
)
from .pipeline import predict_inputs, prepare_features, recording_images, train_model
from .segmentation import load_recording, resample, segment_recording, split_into_segments
from .spectrogram import (
    SPECTROGRAM_SCALES,
    WINDOW_NAMES,
    band_weights,
    power_spectrogram,
    spectrogram_image,
    spectrogram_images,
)
from .synthetic import SYNTHETIC_CLASSES, make_sustained_vowel, write_synthetic_dataset

__version__ = "0.1.0"

__all__ = [
    "CNN_CHANNELS",
    "DATASET_INDEX_NAME",
    "FEATURE_SECTIONS",
    "FIGURE_FORMATS",
    "MODEL_NAMES",
    "SCALAR_METRICS",
    "SERIES_COLORS",
    "SPECTROGRAM_SCALES",
    "SYNTHETIC_CLASSES",
    "WINDOW_NAMES",
    "Config",
    "ConfigError",
    "CrossValidationResult",
    "DataConfig",
    "DatasetError",
    "EvaluationConfig",
    "FeatureSet",
    "MissingDependencyError",
    "ModelConfig",
    "ModelFileError",
    "OutputConfig",
    "OversampledClassifier",
    "SegmentationConfig",
    "SpectrogramCNNClassifier",
    "SpectrogramConfig",
    "SyntheticConfig",
    "__version__",
    "aggregate_recordings",
    "band_weights",
    "build_manifest",
    "build_model",
    "class_spectrograms_table",
    "classification_scores",
    "configure_logging",
    "confusion_matrix_table",
    "cross_validate",
    "feature_map_shapes",
    "index_audio_folder",
    "items_table",
    "load_dataset_index",
    "load_features",
    "load_model",
    "load_recording",
    "make_sustained_vowel",
    "model_filename",
    "new_figure",
    "parameter_count",
    "plot_class_spectrograms",
    "plot_confusion_matrix",
    "pooled_report",
    "power_spectrogram",
    "predict_inputs",
    "prepare_features",
    "recording_images",
    "require_module",
    "resample",
    "run_directory",
    "save_features",
    "save_figure",
    "save_model",
    "segment_recording",
    "spectrogram_image",
    "spectrogram_images",
    "split_into_segments",
    "train_model",
    "write_json",
    "write_manifest",
    "write_synthetic_dataset",
]
