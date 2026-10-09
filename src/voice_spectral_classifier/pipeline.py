"""The three high-level steps shared by the command line and the examples.

* :func:`prepare_features` -- dataset index to segment images (``prepare``);
* :func:`train_model` -- one model on every segment (``train``);
* :func:`predict_inputs` -- new recordings through a saved model, using only
  the settings stored in the model file (``predict``).

Each recording goes through :func:`recording_images`: it is read and
resampled, trimmed of silence when ``segmentation.trim_silence`` is set, cut
into segments, and every segment becomes a spectrogram image.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import FeatureSet, load_dataset_index
from .errors import DatasetError
from .evaluation import score_columns
from .models import build_model, fit_estimator
from .segmentation import load_recording, segment_recording
from .spectrogram import spectrogram_images

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from .config import Config

_LOGGER = logging.getLogger(__name__)


def recording_images(
    path: str | Path, config: Config
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Spectrogram images of every selected segment of one recording.

    Parameters
    ----------
    path : str or pathlib.Path
        WAV file.
    config : Config
        Complete configuration; ``segmentation`` and ``spectrogram`` are used.

    Returns
    -------
    images : numpy.ndarray of uint8, shape (n_segments, n_bands, n_frames)
        May hold no image when the recording is shorter than
        ``segmentation.min_fraction`` of a segment.
    segments : numpy.ndarray of int64, shape (n_segments,)
        Index of each segment within the recording.
    """
    signal = load_recording(path, config.segmentation.sample_rate)
    pieces, indices = segment_recording(signal, config.segmentation)
    images = spectrogram_images(pieces, config.segmentation.sample_rate, config.spectrogram)
    return images, indices


def prepare_features(dataset: str | Path, config: Config) -> FeatureSet:
    """Turn every recording of a dataset index into segment images.

    Parameters
    ----------
    dataset : str or pathlib.Path
        Dataset index (``dataset.csv``).
    config : Config
        Complete configuration; it is stored in the result.

    Returns
    -------
    FeatureSet
        One flattened image per segment, in the order of the index.

    Raises
    ------
    FileNotFoundError
        If the index does not exist.
    DatasetError
        If the index is invalid or no recording yields a segment.
    """
    index = load_dataset_index(dataset)
    images: list[NDArray[np.uint8]] = []
    columns: dict[str, list[Any]] = {"y": [], "groups": [], "paths": [], "segments": []}
    skipped: list[str] = []
    for row in index.itertuples(index=False):
        recording, segments = recording_images(row.path, config)
        if recording.shape[0] == 0:
            skipped.append(row.source)
            continue
        images.append(recording.reshape(recording.shape[0], -1))
        columns["y"] += [row.label] * recording.shape[0]
        columns["groups"] += [row.group] * recording.shape[0]
        columns["paths"] += [row.source] * recording.shape[0]
        columns["segments"] += segments.tolist()
    if skipped:
        _LOGGER.warning(
            "%d recording(s) shorter than segmentation.min_fraction of a segment were skipped: %s",
            len(skipped),
            ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""),
        )
    if not images:
        raise DatasetError(
            f"no recording in {Path(dataset).as_posix()} is long enough for one "
            f"{config.segmentation.segment_s} s segment"
        )
    features = FeatureSet(
        X=np.concatenate(images, axis=0),
        y=np.asarray(columns["y"], dtype=str),
        groups=np.asarray(columns["groups"], dtype=str),
        paths=np.asarray(columns["paths"], dtype=str),
        config=config.to_dict(),
        segments=np.asarray(columns["segments"], dtype=np.int64),
    )
    _LOGGER.info(
        "prepared %d segments from %d recordings", features.n_items, len(index) - len(skipped)
    )
    return features


def train_model(features: FeatureSet, config: Config) -> BaseEstimator:
    """Fit the configured model on every segment.

    Parameters
    ----------
    features : FeatureSet
        Segments written by ``prepare``.
    config : Config
        Complete configuration.

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted estimator; store it with
        :func:`~voice_spectral_classifier.models.save_model`.
    """
    labels = np.asarray(features.y).astype(str)
    if np.unique(labels).size < 2:
        raise DatasetError(f"training needs at least two classes, found {sorted(set(labels))}")
    _LOGGER.info("training %s on %d segments", config.model.name, features.n_items)
    return fit_estimator(build_model(config), features.X, labels)


def predict_inputs(inputs: Sequence[str | Path], bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Classify the segments of new recordings with a saved model.

    Only the configuration stored in the model file is used, so the inputs
    are segmented and analyzed exactly as the training data were.

    Parameters
    ----------
    inputs : sequence of str or pathlib.Path
        WAV files.
    bundle : mapping
        Model bundle returned by :func:`~voice_spectral_classifier.models.load_model`.

    Returns
    -------
    pandas.DataFrame
        One row per segment (``predictions.csv``): ``path``, ``segment``,
        ``predicted`` and ``score_<class>``, in the order of ``inputs``.

    Raises
    ------
    DatasetError
        If no input yields a segment.
    """
    from .config import Config  # deferred: config imports this package's modules

    config = Config.from_dict(bundle["config"])
    estimator = bundle["estimator"]
    classes = [str(name) for name in bundle["classes"]]
    frames: list[pd.DataFrame] = []
    for item in inputs:
        source = Path(item)
        images, segments = recording_images(source, config)
        if images.shape[0] == 0:
            _LOGGER.warning("%s is too short for one segment and was skipped", source.as_posix())
            continue
        probabilities = np.asarray(
            estimator.predict_proba(images.reshape(images.shape[0], -1)), dtype=np.float64
        )
        order = [list(estimator.classes_).index(name) for name in classes]
        probabilities = probabilities[:, order]
        frame = pd.DataFrame(
            {
                "path": source.as_posix(),
                "segment": segments,
                "predicted": np.asarray(classes)[np.argmax(probabilities, axis=1)],
            }
        )
        for position, column in enumerate(score_columns(classes)):
            frame[column] = probabilities[:, position]
        frames.append(frame)
    if not frames:
        raise DatasetError("none of the inputs is long enough for one segment")
    return pd.concat(frames, ignore_index=True)
