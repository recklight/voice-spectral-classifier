"""Grouped cross-validation, metrics against scikit-learn, and recording-level scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from voice_spectral_classifier import (
    Config,
    DatasetError,
    aggregate_recordings,
    classification_scores,
    cross_validate,
)
from voice_spectral_classifier.evaluation import SCALAR_METRICS, summarize_folds

LOGISTIC = Config().override({"model.name": "logistic", "evaluation.n_splits": 4})
SCORES = ["score_a", "score_b", "score_c"]


@pytest.fixture
def toy(make_toy, rng):
    return make_toy(rng)


@pytest.fixture
def result(toy):
    return cross_validate(toy, LOGISTIC)


def test_folds_never_split_a_group(result):
    folds_per_group = result.predictions.groupby("group")["fold"].nunique()
    assert folds_per_group.eq(1).all()


def test_every_segment_gets_exactly_one_out_of_fold_prediction(toy, result):
    predictions = result.predictions
    assert len(predictions) == toy.n_items
    assert set(predictions["fold"]) == {0, 1, 2, 3}
    assert result.folds["n_test"].sum() == toy.n_items
    assert (result.folds["n_train"] + result.folds["n_test"]).eq(toy.n_items).all()
    assert np.allclose(predictions[SCORES].sum(axis=1), 1.0)


def test_column_order_of_the_tables(result):
    assert list(result.predictions.columns) == [
        "path",
        "segment",
        "group",
        "fold",
        "label",
        "predicted",
        *SCORES,
    ]
    assert list(result.recording_predictions.columns) == [
        "path",
        "group",
        "fold",
        "label",
        "predicted",
        *SCORES,
    ]
    assert list(result.folds.columns) == ["fold", "n_train", "n_test", *SCALAR_METRICS]
    assert list(result.metrics) == [
        "task",
        "n_items",
        "n_splits",
        "seed",
        "folds",
        "summary",
        "pooled",
        "recordings",
    ]
    assert list(result.metrics["recordings"]) == ["n_items", "folds", "summary", "pooled"]


def test_pooled_metrics_agree_with_scikit_learn(result):
    predictions = result.predictions
    truth, guess = predictions["label"], predictions["predicted"]
    pooled = result.metrics["pooled"]
    assert pooled["uar"] == pytest.approx(balanced_accuracy_score(truth, guess), abs=1e-12)
    assert pooled["accuracy"] == pytest.approx(accuracy_score(truth, guess), abs=1e-12)
    assert pooled["f1_macro"] == pytest.approx(f1_score(truth, guess, average="macro"), abs=1e-12)
    expected_auc = roc_auc_score(truth, predictions[SCORES], multi_class="ovr", average="macro")
    assert pooled["roc_auc_ovr"] == pytest.approx(expected_auc, abs=1e-12)
    matrix = confusion_matrix(truth, guess, labels=["a", "b", "c"])
    assert pooled["confusion_matrix"] == matrix.tolist()
    negatives = matrix.sum() - matrix[0].sum()
    true_negatives = negatives - (matrix[1:, 0]).sum()
    assert pooled["per_class"]["a"]["specificity"] == pytest.approx(true_negatives / negatives)
    assert pooled["per_class"]["a"]["support"] == int(matrix[0].sum())


def test_summary_uses_the_sample_standard_deviation(result):
    for metric in SCALAR_METRICS:
        values = result.folds[metric].dropna().to_numpy()
        summary = result.metrics["summary"][metric]
        assert summary["mean"] == pytest.approx(values.mean())
        assert summary["std"] == pytest.approx(values.std(ddof=1))
    pd.testing.assert_frame_equal(pd.DataFrame(result.metrics["folds"]), result.folds)


def test_undefined_fold_values_are_left_out_of_the_summary():
    table = pd.DataFrame({metric: [0.5, 0.7, np.nan] for metric in SCALAR_METRICS})
    summary = summarize_folds(table)
    assert summary["uar"]["mean"] == pytest.approx(0.6)
    lonely = summarize_folds(pd.DataFrame({metric: [0.5] for metric in SCALAR_METRICS}))
    assert np.isnan(lonely["uar"]["std"])


def test_recording_scores_average_their_segments(result):
    predictions, recordings = result.predictions, result.recording_predictions
    assert (
        len(recordings) == predictions["path"].nunique() == result.metrics["recordings"]["n_items"]
    )
    first = recordings.iloc[0]
    rows = predictions[predictions["path"] == first["path"]]
    assert np.allclose(rows[SCORES].mean().to_numpy(), first[SCORES].to_numpy(dtype=float))
    assert (
        first["predicted"] == ["a", "b", "c"][int(np.argmax(first[SCORES].to_numpy(dtype=float)))]
    )
    assert rows["fold"].nunique() == 1 and rows["fold"].iloc[0] == first["fold"]


def test_aggregation_keeps_first_appearance_order():
    rows = pd.DataFrame(
        {
            "path": ["z.wav", "a.wav", "z.wav"],
            "segment": [0, 0, 1],
            "score_x": [0.2, 0.9, 0.4],
            "score_y": [0.8, 0.1, 0.6],
        }
    )
    table = aggregate_recordings(rows, ["x", "y"])
    assert table["path"].tolist() == ["z.wav", "a.wav"]
    assert table["predicted"].tolist() == ["y", "x"]
    assert table["score_x"].tolist() == pytest.approx([0.3, 0.9])


def test_without_groups_the_recording_is_the_group(make_toy, rng):
    features = make_toy(rng, with_groups=False)
    predictions = cross_validate(features, LOGISTIC).predictions
    assert predictions.groupby("path")["fold"].nunique().eq(1).all()
    assert (predictions["group"] == "").all()


def test_the_same_seed_reproduces_the_folds_and_another_seed_changes_them(toy):
    first = cross_validate(toy, LOGISTIC).predictions
    second = cross_validate(toy, LOGISTIC).predictions
    pd.testing.assert_frame_equal(first, second)
    reshuffled = cross_validate(toy, LOGISTIC.override({"seed": 1})).predictions
    assert not first["fold"].equals(reshuffled["fold"])


def test_scores_with_a_class_absent_from_a_fold():
    scores = classification_scores(["a", "a"], ["a", "b"], [[0.9, 0.1], [0.4, 0.6]], ["a", "b"])
    # speechdsp.uar averages the classes present in y_true, so the predicted-only
    # "b" does not count; the ROC-AUC is undefined without a "b" in the fold.
    assert scores["uar"] == pytest.approx(0.5)
    assert np.isnan(scores["roc_auc_ovr"])


def test_too_few_groups_or_classes_are_reported(make_toy, rng):
    sparse = make_toy(rng, groups_per_class=1)
    with pytest.raises(DatasetError, match=r"only 3 groups for evaluation\.n_splits=4"):
        cross_validate(sparse, LOGISTIC)
    single = make_toy(rng, classes=("a",), groups_per_class=8)
    with pytest.raises(DatasetError, match="at least two classes"):
        cross_validate(single, LOGISTIC)


def test_quick_synthetic_data_is_learnable(quick_features, fast_config):
    result = cross_validate(quick_features, fast_config)
    assert result.metrics["n_splits"] == 3
    assert result.metrics["pooled"]["uar"] > 0.6
    assert result.metrics["recordings"]["n_items"] == 18
