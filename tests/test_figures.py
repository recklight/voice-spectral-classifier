"""Figures render without a display and come with matching tables."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from voice_spectral_classifier import (
    Config,
    class_spectrograms_table,
    confusion_matrix_table,
    cross_validate,
    new_figure,
    plot_class_spectrograms,
    plot_confusion_matrix,
    save_figure,
)


@pytest.fixture
def result(make_toy, rng):
    config = Config().override({"model.name": "logistic", "evaluation.n_splits": 3})
    return cross_validate(make_toy(rng), config)


@pytest.mark.parametrize("level", ["segments", "recordings"])
def test_confusion_matrix_figure_and_table(result, level):
    assert isinstance(plot_confusion_matrix(result, level, dpi=60), Figure)
    table = confusion_matrix_table(result, level)
    assert list(table.columns) == ["label", "predicted", "count", "fraction"]
    assert len(table) == 9
    assert np.allclose(table.groupby("label")["fraction"].sum(), 1.0)
    pooled = result.metrics if level == "segments" else result.metrics["recordings"]
    assert table["count"].sum() == pooled["n_items"]


def test_an_unknown_level_is_rejected(result):
    with pytest.raises(ValueError, match="level='speakers'"):
        confusion_matrix_table(result, "speakers")


def test_class_spectrogram_figure_and_table(quick_features):
    assert isinstance(plot_class_spectrograms(quick_features, dpi=60), Figure)
    table = class_spectrograms_table(quick_features)
    assert list(table.columns) == ["label", "band", "frequency_hz", "frame", "mean_level"]
    assert len(table) == 3 * 40 * 98
    breathy = table[table["label"] == "breathy"]
    expected = quick_features.images()[quick_features.y == "breathy"].mean(axis=0)
    assert np.allclose(breathy["mean_level"].to_numpy().reshape(40, 98), expected)
    assert breathy["frequency_hz"].iloc[0] == 100.0


def test_class_spectrogram_time_axis_spans_the_segment(make_toy, rng):
    features = make_toy(rng)
    features.config["segmentation"]["segment_s"] = 2.0
    axes = plot_class_spectrograms(features, dpi=60).axes[0]
    assert axes.get_xlim() == (0.0, 2000.0)


def test_saving_keeps_dots_in_the_file_name(tmp_path):
    figure = new_figure(2, 2, 50)
    assert figure.get_size_inches().tolist() == [2.0, 2.0]
    for fmt in ("png", "svg", "pdf"):
        path = save_figure(figure, tmp_path / "run.v1" / "scores.final", fmt)
        assert path.name == f"scores.final.{fmt}" and path.stat().st_size > 0
