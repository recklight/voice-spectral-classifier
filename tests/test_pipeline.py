"""High-level steps: prepare segment images, train, and predict from a model file."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import speechdsp

from voice_spectral_classifier import (
    DatasetError,
    load_model,
    predict_inputs,
    prepare_features,
    recording_images,
    save_model,
    train_model,
)


def test_prepare_turns_every_recording_into_segment_images(quick_dataset, quick_features):
    index = pd.read_csv(quick_dataset)
    features = quick_features
    assert features.X.dtype == np.uint8 and features.X.shape == (36, 40 * 98)
    assert features.images().shape == (36, 40, 98)
    assert set(features.paths) == set(index["path"])
    assert features.paths[0] == "breathy/breathy_000.wav"
    per_recording = pd.Series(features.segments).groupby(features.paths).apply(list)
    assert all(segments == [0, 1] for segments in per_recording)
    groups = dict(zip(index["path"], index["group"], strict=True))
    assert all(
        groups[path] == group for path, group in zip(features.paths, features.groups, strict=True)
    )
    assert features.config["segmentation"]["sample_rate"] == 16_000


def test_recording_images_are_resampled_and_segmented(quick_dataset, fast_config):
    path = quick_dataset.parent / "modal" / "modal_000.wav"
    images, segments = recording_images(path, fast_config)
    assert images.shape == (2, 40, 98) and segments.tolist() == [0, 1]
    assert images.max() == 255


def test_short_recordings_are_skipped_and_reported(tmp_path, fast_config, caplog):
    folder = tmp_path / "data"
    rng = np.random.default_rng(0)
    speechdsp.write_wav(folder / "a" / "long.wav", 0.3 * rng.standard_normal(32_000), 16_000)
    speechdsp.write_wav(folder / "b" / "short.wav", 0.3 * rng.standard_normal(4000), 16_000)
    (folder / "dataset.csv").write_text(
        "path,label\na/long.wav,a\nb/short.wav,b\n", encoding="utf-8"
    )
    config = fast_config.override({"segmentation.trim_silence": False})
    with caplog.at_level(logging.WARNING):
        features = prepare_features(folder / "dataset.csv", config)
    assert features.n_items == 2 and set(features.paths) == {"a/long.wav"}
    assert "b/short.wav" in caplog.text
    (folder / "dataset.csv").write_text("path,label\nb/short.wav,b\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="long enough"):
        prepare_features(folder / "dataset.csv", config)


def test_training_needs_two_classes(quick_features, fast_config):
    keep = quick_features.y == "modal"
    single = type(quick_features)(
        X=quick_features.X[keep],
        y=quick_features.y[keep],
        groups=quick_features.groups[keep],
        paths=quick_features.paths[keep],
        config=quick_features.config,
        segments=quick_features.segments[keep],
    )
    with pytest.raises(DatasetError, match="at least two classes"):
        train_model(single, fast_config)


def test_predictions_use_the_settings_stored_in_the_model(tmp_path, quick_dataset, fast_config):
    config = fast_config.override({"segmentation.max_segments": 1})
    features = prepare_features(quick_dataset, config)
    model = save_model(tmp_path / "model.joblib", train_model(features, config), config)
    bundle = load_model(model)
    folder = quick_dataset.parent
    inputs = [folder / "rough" / "rough_001.wav", folder / "breathy" / "breathy_002.wav"]
    table = predict_inputs(inputs, bundle)
    assert list(table.columns) == [
        "path",
        "segment",
        "predicted",
        "score_breathy",
        "score_modal",
        "score_rough",
    ]
    assert table["path"].tolist() == [path.as_posix() for path in inputs]
    assert table["segment"].tolist() == [1, 1]  # the middle one of two segments
    assert np.allclose(table.filter(like="score_").sum(axis=1), 1.0)
    assert table["predicted"].tolist() == ["rough", "breathy"]


def test_prediction_needs_at_least_one_usable_input(tmp_path, quick_features, fast_config):
    bundle = {
        "config": fast_config.to_dict(),
        "classes": quick_features.classes,
        "estimator": train_model(quick_features, fast_config),
    }
    speechdsp.write_wav(tmp_path / "blip.wav", np.zeros(100), 16_000)
    with pytest.raises(DatasetError, match="long enough"):
        predict_inputs([tmp_path / "blip.wav"], bundle)
