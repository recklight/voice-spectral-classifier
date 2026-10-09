"""Report figures, each with a table of the values it plots.

Figures are built on :class:`matplotlib.figure.Figure` directly, without
``pyplot``, so they render without a display. Titles and axis labels are in
English, so no CJK font is needed.

* :func:`plot_confusion_matrix` / :func:`confusion_matrix_table` -- the pooled
  out-of-fold confusion matrix; every cell shows its count and its share of
  the true class, so the diagonal is the per-class recall;
* :func:`plot_class_spectrograms` / :func:`class_spectrograms_table` -- the
  mean spectrogram image of every class, to check that the classes differ in
  the bands where they should.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .spectrogram import band_frequencies

if TYPE_CHECKING:
    from .dataset import FeatureSet
    from .evaluation import CrossValidationResult

#: Colors of successive data series.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227")
#: Values of ``level`` accepted by the confusion-matrix functions.
CONFUSION_LEVELS: tuple[str, ...] = ("segments", "recordings")


def new_figure(width: float = 6.0, height: float = 4.0, dpi: int = 150) -> Figure:
    """Blank figure with constrained layout.

    Examples
    --------
    >>> new_figure(4, 3, 100).get_size_inches().tolist()
    [4.0, 3.0]
    """
    return Figure(figsize=(width, height), dpi=dpi, layout="constrained")


def save_figure(figure: Figure, path_stem: str | Path, figure_format: str = "png") -> Path:
    """Save ``figure`` as ``<path_stem>.<figure_format>`` and return the path.

    The extension is appended rather than substituted, so a stem with dots in
    it (``scores.v2``) keeps them.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     save_figure(new_figure(1, 1, 50), Path(folder) / "scores.v2", "svg").name
    'scores.v2.svg'
    """
    stem = Path(path_stem)
    destination = stem.with_name(f"{stem.name}.{figure_format}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format=figure_format)
    return destination


def _pooled(result: CrossValidationResult, level: str) -> dict:
    if level not in CONFUSION_LEVELS:
        raise ValueError(
            f"level={level!r} is not supported; choose one of {list(CONFUSION_LEVELS)}"
        )
    metrics = result.metrics if level == "segments" else result.metrics["recordings"]
    return metrics["pooled"]


def confusion_matrix_table(result: CrossValidationResult, level: str = "segments") -> pd.DataFrame:
    """Pooled confusion matrix in long form.

    Parameters
    ----------
    result : CrossValidationResult
        Output of :func:`~voice_spectral_classifier.evaluation.cross_validate`.
    level : {"segments", "recordings"}, optional
        Which predictions to count.

    Returns
    -------
    pandas.DataFrame
        Columns ``label`` (true class), ``predicted``, ``count`` and
        ``fraction`` (share of the true class's items).
    """
    pooled = _pooled(result, level)
    classes = list(pooled["classes"])
    counts = np.asarray(pooled["confusion_matrix"], dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = np.where(totals > 0, counts / np.maximum(totals, 1), 0.0)
    return pd.DataFrame(
        {
            "label": np.repeat(classes, len(classes)),
            "predicted": np.tile(classes, len(classes)),
            "count": counts.ravel(),
            "fraction": fractions.ravel(),
        }
    )


def plot_confusion_matrix(
    result: CrossValidationResult, level: str = "segments", dpi: int = 150
) -> Figure:
    """Heat map of the pooled confusion matrix with counts and row shares.

    Parameters
    ----------
    result : CrossValidationResult
        Output of :func:`~voice_spectral_classifier.evaluation.cross_validate`.
    level : {"segments", "recordings"}, optional
        Which predictions to count.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = confusion_matrix_table(result, level)
    classes = list(_pooled(result, level)["classes"])
    size = len(classes)
    fractions = table["fraction"].to_numpy().reshape(size, size)
    counts = table["count"].to_numpy().reshape(size, size)
    figure = new_figure(2.4 + 1.1 * size, 1.2 + 1.0 * size, dpi)
    axes = figure.add_subplot()
    image = axes.imshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(size):
        for column in range(size):
            share = fractions[row, column]
            axes.text(
                column,
                row,
                f"{counts[row, column]}\n{share:.0%}",
                ha="center",
                va="center",
                color="white" if share > 0.5 else "black",
                fontsize=9,
            )
    axes.set_xticks(range(size), classes, rotation=30, ha="right")
    axes.set_yticks(range(size), classes)
    axes.set_xlabel("predicted class")
    axes.set_ylabel("true class")
    uar = _pooled(result, level)["uar"]
    axes.set_title(f"{level.capitalize()}: UAR {uar:.3f}")
    figure.colorbar(image, ax=axes, label="share of true class")
    return figure


def class_spectrograms_table(features: FeatureSet) -> pd.DataFrame:
    """Mean grey level of every band and frame, per class, in long form.

    Returns
    -------
    pandas.DataFrame
        Columns ``label``, ``band`` (0 = lowest), ``frequency_hz`` (band
        center), ``frame`` and ``mean_level`` (0-255).
    """
    images = features.images().astype(np.float64)
    labels = np.asarray(features.y).astype(str)
    classes = features.classes
    n_bands, n_frames = features.image_shape
    spectrogram_config = features.config["spectrogram"]
    centers = band_frequencies(
        spectrogram_config["scale"], n_bands, float(spectrogram_config["fmax_hz"])
    )
    means = np.stack([images[labels == name].mean(axis=0) for name in classes])
    return pd.DataFrame(
        {
            "label": np.repeat(classes, n_bands * n_frames),
            "band": np.tile(np.repeat(np.arange(n_bands), n_frames), len(classes)),
            "frequency_hz": np.tile(np.repeat(centers, n_frames), len(classes)),
            "frame": np.tile(np.arange(n_frames), len(classes) * n_bands),
            "mean_level": means.ravel(),
        }
    )


def plot_class_spectrograms(features: FeatureSet, dpi: int = 150) -> Figure:
    """Mean spectrogram image of each class, lowest band at the bottom.

    Parameters
    ----------
    features : FeatureSet
        Segments written by ``prepare``.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = class_spectrograms_table(features)
    classes = features.classes
    n_bands, n_frames = features.image_shape
    spectrogram_config = features.config["spectrogram"]
    figure = new_figure(2.0 + 2.6 * len(classes), 3.2, dpi)
    axes_list = figure.subplots(1, len(classes), sharey=True, squeeze=False)[0]
    # The frames are interpolated to n_frames columns, so the columns span the
    # whole segment rather than n_frames hops.
    segment_ms = 1000.0 * float(features.config["segmentation"]["segment_s"])
    image = None
    for axes, name in zip(axes_list, classes, strict=True):
        mean = table.loc[table["label"] == name, "mean_level"].to_numpy().reshape(n_bands, n_frames)
        image = axes.imshow(
            mean,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=255.0,
            extent=(0.0, segment_ms, -0.5, n_bands - 0.5),
        )
        count = int(np.sum(np.asarray(features.y).astype(str) == name))
        axes.set_title(f"{name} (n={count})")
        axes.set_xlabel("time (ms)")
    axes_list[0].set_ylabel(f"band ({spectrogram_config['scale']}, 0 = lowest)")
    if image is not None:
        figure.colorbar(image, ax=list(axes_list), label="mean grey level")
    return figure
