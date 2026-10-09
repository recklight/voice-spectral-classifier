"""The spectrogram CNN: layout arithmetic without PyTorch, training and inference with it."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from voice_spectral_classifier import (
    CNN_CHANNELS,
    Config,
    MissingDependencyError,
    SpectrogramCNNClassifier,
    build_model,
    feature_map_shapes,
    parameter_count,
)

NO_TORCH = importlib.util.find_spec("torch") is None


def test_feature_maps_shrink_to_320_values_for_40_by_98_images():
    shapes = feature_map_shapes((40, 98))
    assert shapes == [
        (16, 20, 49),
        (32, 10, 25),
        (64, 10, 25),
        (64, 5, 13),
        (64, 5, 13),
        (64, 5, 1),
    ]
    assert int(np.prod(shapes[-1])) == 320


def test_parameter_count_matches_the_layer_arithmetic():
    convolutions = 1 * 16 * 9 + 16 + 16 * 32 * 9 + 32 + 32 * 64 * 9 + 64 + 3 * (64 * 64 * 9 + 64)
    normalization = 2 * sum(CNN_CHANNELS)
    assert parameter_count(3) == convolutions + normalization + 320 * 3 + 3 == 135_651


def test_other_image_widths_still_end_in_one_time_step():
    assert feature_map_shapes((40, 120))[-1] == (64, 5, 1)
    assert feature_map_shapes((20, 50))[-1] == (64, 2, 1)


def test_too_small_images_and_wrong_layouts_are_rejected():
    with pytest.raises(ValueError, match="too small"):
        feature_map_shapes((3, 98))
    with pytest.raises(ValueError, match="channel counts"):
        feature_map_shapes((40, 98), channels=(16, 32))


def test_the_estimator_exposes_its_hyperparameters():
    model = build_model(Config()) if not NO_TORCH else None
    params = SpectrogramCNNClassifier(n_epochs=7, dropout=0.1).get_params()
    assert params["n_epochs"] == 7 and params["dropout"] == 0.1 and params["batch_size"] == 60
    if model is not None:
        assert model.get_params()["image_shape"] == (40, 98)


@pytest.mark.skipif(not NO_TORCH, reason="PyTorch is installed")
def test_without_pytorch_the_error_names_the_dl_extra():
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        build_model(Config())
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        SpectrogramCNNClassifier().fit(np.zeros((4, 40 * 98)), ["a", "a", "b", "b"])


def _two_class_images(rng, n_per_class=12):
    """Class ``low`` has energy in the bottom rows, class ``high`` in the top rows."""
    images = rng.integers(0, 60, size=(2 * n_per_class, 40, 98)).astype(np.uint8)
    images[:n_per_class, :10, :] += 150
    images[n_per_class:, 30:, :] += 150
    labels = np.array(["low"] * n_per_class + ["high"] * n_per_class)
    return images, labels


def _small_cnn(**overrides):
    hyperparameters = {"n_epochs": 15, "learning_rate": 1e-2, "batch_size": 8, "random_state": 0}
    hyperparameters.update(overrides)
    return SpectrogramCNNClassifier((40, 98), **hyperparameters)


def test_training_separates_two_obvious_classes(rng):
    pytest.importorskip("torch")
    images, labels = _two_class_images(rng)
    model = _small_cnn().fit(images, labels)
    assert model.classes_.tolist() == ["high", "low"]
    probabilities = model.predict_proba(images.reshape(len(images), -1))
    assert probabilities.shape == (24, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert np.mean(model.predict(images) == labels) >= 0.9


def test_network_layout_and_mean_image_buffer(rng):
    torch = pytest.importorskip("torch")
    images, labels = _two_class_images(rng, n_per_class=4)
    model = _small_cnn(n_epochs=1).fit(images, labels)
    trainable = sum(p.numel() for p in model.network_.parameters() if p.requires_grad)
    assert trainable == parameter_count(2)
    with torch.no_grad():
        hidden = model.network_.features(torch.zeros(1, 1, 40, 98))
    assert tuple(hidden.shape[1:]) == feature_map_shapes((40, 98))[-1]
    expected_mean = images.astype(np.float32).mean(axis=0) / 255.0
    assert np.allclose(model.network_.input_mean.numpy()[0, 0], expected_mean, atol=1e-5)


def test_the_same_seed_trains_the_same_network(rng):
    pytest.importorskip("torch")
    images, labels = _two_class_images(rng, n_per_class=6)
    first = _small_cnn(n_epochs=3).fit(images, labels).predict_proba(images)
    second = _small_cnn(n_epochs=3).fit(images, labels).predict_proba(images)
    assert np.allclose(first, second, atol=1e-6)


def test_state_dict_round_trip_reproduces_the_predictions(rng):
    pytest.importorskip("torch")
    images, labels = _two_class_images(rng, n_per_class=6)
    trained = _small_cnn(n_epochs=3).fit(images, labels)
    restored = _small_cnn().load_state_dict(trained.state_dict(), trained.classes_.tolist())
    assert np.allclose(trained.predict_proba(images), restored.predict_proba(images), atol=1e-6)
    assert "input_mean" in trained.state_dict()


def test_wrong_image_shapes_are_rejected(rng):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="expected images of shape"):
        _small_cnn().fit(np.zeros((4, 100)), ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="at least two classes"):
        _small_cnn().fit(np.zeros((2, 40, 98)), ["a", "a"])
