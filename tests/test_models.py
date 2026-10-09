"""Model registry, class balancing and model files."""

from __future__ import annotations

import importlib.util
import logging

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from voice_spectral_classifier import (
    MODEL_NAMES,
    Config,
    MissingDependencyError,
    ModelFileError,
    OversampledClassifier,
    build_model,
    load_model,
    model_filename,
    save_model,
    train_model,
)
from voice_spectral_classifier.models import MODEL_FORMAT, fit_estimator

NO_TORCH = importlib.util.find_spec("torch") is None


def test_registered_model_names():
    assert MODEL_NAMES == ("cnn", "mlp", "logistic")


def test_mlp_pipeline_steps_follow_the_settings():
    balanced = build_model(Config().override({"model.name": "mlp", "model.n_components": 8}))
    assert [name for name, _ in balanced.steps] == ["standardize", "project", "classify"]
    assert isinstance(balanced.named_steps["classify"], OversampledClassifier)
    network = balanced.named_steps["classify"].estimator
    assert isinstance(network, MLPClassifier) and network.hidden_layer_sizes == (128,)
    plain = build_model(Config().override({"model.name": "mlp", "model.class_weight": None}))
    assert isinstance(plain.named_steps["classify"], MLPClassifier)


def test_logistic_regression_uses_class_weights():
    model = build_model(Config().override({"model.name": "logistic", "model.logistic_c": 0.5}))
    classifier = model.named_steps["classify"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight == "balanced" and classifier.C == 0.5


def test_oversampling_balances_the_classes_seen_by_the_classifier():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.array(["a"] * 8 + ["b"] * 2)
    model = OversampledClassifier(DummyClassifier(strategy="prior"), random_state=0).fit(X, y)
    assert np.allclose(model.estimator_.class_prior_, [0.5, 0.5])
    assert model.classes_.tolist() == ["a", "b"]
    assert model.predict_proba(X[:2]).shape == (2, 2)


def test_model_file_names():
    assert model_filename(Config()) == "model.pt"
    assert model_filename(Config().override({"model.name": "logistic"})) == "model.joblib"


def test_scikit_learn_models_round_trip(tmp_path, quick_features, fast_config):
    estimator = train_model(quick_features, fast_config)
    path = save_model(tmp_path / "model.joblib", estimator, fast_config, quick_features.classes)
    bundle = load_model(path)
    assert list(bundle) == ["format", "package_version", "config", "classes", "estimator"]
    assert bundle["format"] == MODEL_FORMAT
    assert bundle["classes"] == ["breathy", "modal", "rough"]
    assert Config.from_dict(bundle["config"]) == fast_config
    assert np.array_equal(
        bundle["estimator"].predict_proba(quick_features.X),
        estimator.predict_proba(quick_features.X),
    )


def test_foreign_and_incomplete_model_files_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'train' first"):
        load_model(tmp_path / "absent.joblib")
    garbage = tmp_path / "garbage.joblib"
    garbage.write_bytes(b"definitely not a pickle")
    with pytest.raises(ModelFileError, match="not a readable model file"):
        load_model(garbage)
    foreign = tmp_path / "foreign.joblib"
    joblib.dump({"format": "someone-else/v3"}, foreign)
    with pytest.raises(ModelFileError, match="was not written by this package"):
        load_model(foreign)
    partial = tmp_path / "partial.joblib"
    joblib.dump({"format": MODEL_FORMAT, "config": {}}, partial)
    with pytest.raises(ModelFileError, match="incomplete"):
        load_model(partial)


def test_convergence_warnings_go_to_the_log(caplog, rng):
    X = rng.standard_normal((40, 5))
    y = np.where(X[:, 0] > 0, "p", "n")
    with caplog.at_level(logging.WARNING, logger="voice_spectral_classifier.models"):
        fit_estimator(MLPClassifier(max_iter=1, random_state=0), X, y)
    assert "raise model.n_epochs" in caplog.text


@pytest.mark.skipif(not NO_TORCH, reason="PyTorch is installed")
def test_cnn_model_files_need_the_dl_extra(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"\x00")
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        load_model(tmp_path / "model.pt")


def test_cnn_model_round_trip(tmp_path, quick_features, fast_config):
    pytest.importorskip("torch")
    config = fast_config.override({"model.name": "cnn", "model.n_epochs": 2})
    estimator = train_model(quick_features, config)
    path = save_model(tmp_path / model_filename(config), estimator, config)
    bundle = load_model(path)
    assert {"format", "config", "classes", "state_dict", "estimator"} <= set(bundle)
    assert np.allclose(
        bundle["estimator"].predict_proba(quick_features.X),
        estimator.predict_proba(quick_features.X),
        atol=1e-6,
    )


def test_a_cnn_trained_on_cuda_runs_on_the_cpu_when_cuda_is_missing(
    tmp_path, quick_features, fast_config, monkeypatch, caplog
):
    torch = pytest.importorskip("torch")
    config = fast_config.override({"model.name": "cnn", "model.n_epochs": 2})
    estimator = train_model(quick_features, config)
    # What a GPU run writes: the same CPU tensors, with model.device=cuda in the stored settings.
    stored = config.override({"model.device": "cuda"})
    path = save_model(tmp_path / "model.pt", estimator, stored, quick_features.classes)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="voice_spectral_classifier.models"):
        bundle = load_model(path)
    assert "running it on the CPU" in caplog.text
    assert bundle["estimator"].device == "cpu"
    assert bundle["config"]["model"]["device"] == "cuda"
    assert np.allclose(
        bundle["estimator"].predict_proba(quick_features.X),
        estimator.predict_proba(quick_features.X),
        atol=1e-6,
    )
