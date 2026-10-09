"""Grouped, stratified k-fold cross-validation at segment and recording level.

Folds are drawn with :class:`~sklearn.model_selection.StratifiedGroupKFold`
(shuffled, ``random_state = seed``; Kohavi, 1995): class proportions are kept
in every fold, and all segments of one group (the ``group`` column of the
dataset index, or the recording itself when there is none) stay on the same
side of each split. A fresh model is built and fitted on the training folds
only, so the scaler, projection or mean image never sees the test fold. Every
segment receives exactly one out-of-fold prediction.

Scores are reported at two levels:

* ``segments``: every segment counts once;
* ``recordings``: the class probabilities of all segments of a recording are
  averaged, and the recording gets the class with the largest mean
  probability.

The main metric is the unweighted average recall (UAR; Schuller et al.,
2009), the mean of the per-class recalls. Unlike accuracy, it is not inflated
by predicting the majority class. Accuracy, macro F1 and the macro
one-vs-rest ROC-AUC are reported alongside; per-class recall (sensitivity),
specificity, precision, F1 and support, and the confusion matrix, are computed
on the pooled out-of-fold predictions. Fold scores are summarized by their
mean and sample standard deviation (``ddof=1``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import speechdsp
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .errors import DatasetError
from .models import build_model, fit_estimator

if TYPE_CHECKING:
    from .config import Config
    from .dataset import FeatureSet

_LOGGER = logging.getLogger(__name__)

#: Scalar metrics, in the order of ``folds.csv`` and ``metrics.json``.
SCALAR_METRICS: tuple[str, ...] = ("uar", "accuracy", "f1_macro", "roc_auc_ovr")


@dataclass(frozen=True, eq=False)
class CrossValidationResult:
    """Out-of-fold predictions and scores of :func:`cross_validate`.

    Attributes
    ----------
    predictions : pandas.DataFrame
        One row per segment (``predictions.csv``): ``path``, ``segment``,
        ``group``, ``fold``, ``label``, ``predicted`` and ``score_<class>``.
    folds : pandas.DataFrame
        One row per fold (``folds.csv``): ``fold``, ``n_train``, ``n_test`` and
        the segment-level scalar metrics.
    metrics : dict
        The contents of ``metrics.json``, including the ``recordings`` section.
    recording_predictions : pandas.DataFrame
        One row per recording (``recording_predictions.csv``): ``path``,
        ``group``, ``fold``, ``label``, ``predicted`` and ``score_<class>``.
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame
    metrics: dict[str, Any]
    recording_predictions: pd.DataFrame

    def as_dict(self) -> dict[str, Any]:
        """The contents of ``metrics.json``."""
        return self.metrics

    @property
    def classes(self) -> list[str]:
        """Class names in sorted order."""
        return list(self.metrics["pooled"]["classes"])


def score_columns(classes: list[str]) -> list[str]:
    """Names of the probability columns, ``score_<class>``.

    Examples
    --------
    >>> score_columns(["a", "b"])
    ['score_a', 'score_b']
    """
    return [f"score_{name}" for name in classes]


def classification_scores(
    y_true: ArrayLike, y_pred: ArrayLike, scores: ArrayLike, classes: list[str]
) -> dict[str, float]:
    """UAR, accuracy, macro F1 and macro one-vs-rest ROC-AUC.

    Parameters
    ----------
    y_true, y_pred : array_like of str, shape (n,)
        True and predicted class names.
    scores : array_like, shape (n, n_classes)
        Class probabilities, columns in the order of ``classes``.
    classes : list of str
        All class names, sorted.

    Returns
    -------
    dict
        ``uar``, ``accuracy``, ``f1_macro`` and ``roc_auc_ovr``. The UAR is
        :func:`speechdsp.uar`, the mean recall over the classes that occur in
        ``y_true``; the macro F1 averages over the classes that occur in
        ``y_true`` or ``y_pred``; the ROC-AUC is ``nan`` when ``y_true`` lacks
        a class.

    References
    ----------
    .. [1] B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion
       Challenge," in Proc. Interspeech 2009, pp. 312-315.

    Examples
    --------
    >>> truth, guess = ["a", "a", "b", "b"], ["a", "b", "b", "b"]
    >>> probabilities = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]]
    >>> {key: round(value, 3) for key, value in classification_scores(
    ...     truth, guess, probabilities, ["a", "b"]).items()}
    {'uar': 0.75, 'accuracy': 0.75, 'f1_macro': 0.733, 'roc_auc_ovr': 1.0}
    """
    truth = np.asarray(y_true).astype(str)
    guess = np.asarray(y_pred).astype(str)
    probabilities = np.asarray(scores, dtype=np.float64)
    return {
        "uar": float(speechdsp.uar(truth, guess)),
        "accuracy": float(accuracy_score(truth, guess)),
        "f1_macro": float(f1_score(truth, guess, average="macro", zero_division=0)),
        "roc_auc_ovr": _roc_auc(truth, probabilities, classes),
    }


def _roc_auc(truth: NDArray[Any], probabilities: NDArray[np.float64], classes: list[str]) -> float:
    """Macro one-vs-rest ROC-AUC, or NaN when a class is absent from ``truth``."""
    if set(truth) != set(classes):
        return float("nan")
    if len(classes) == 2:
        return float(roc_auc_score(truth == classes[1], probabilities[:, 1]))
    return float(
        roc_auc_score(truth, probabilities, multi_class="ovr", average="macro", labels=classes)
    )


def pooled_report(
    y_true: ArrayLike, y_pred: ArrayLike, scores: ArrayLike, classes: list[str]
) -> dict[str, Any]:
    """Scalar metrics plus confusion matrix and per-class figures.

    Returns
    -------
    dict
        The keys of :func:`classification_scores`, then ``classes``,
        ``confusion_matrix`` (rows: true class, columns: predicted class) and
        ``per_class`` (``recall``, ``precision``, ``f1``, ``support`` and the
        one-vs-rest ``specificity`` of every class; the recall is the
        sensitivity).

    Examples
    --------
    >>> scores = [[1, 0], [0, 1], [1, 0]]
    >>> report = pooled_report(["a", "b", "b"], ["a", "b", "a"], scores, ["a", "b"])
    >>> report["confusion_matrix"], report["per_class"]["b"]["specificity"]
    ([[1, 0], [1, 1]], 1.0)
    """
    truth = np.asarray(y_true).astype(str)
    guess = np.asarray(y_pred).astype(str)
    report: dict[str, Any] = classification_scores(truth, guess, scores, classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, guess, labels=classes, zero_division=0
    )
    per_class: dict[str, dict[str, float | int]] = {}
    for position, name in enumerate(classes):
        _, specificity = speechdsp.sensitivity_specificity(truth, guess, pos_label=name)
        per_class[name] = {
            "recall": float(recall[position]),
            "precision": float(precision[position]),
            "f1": float(f1[position]),
            "support": int(support[position]),
            "specificity": float(specificity),
        }
    report["classes"] = list(classes)
    report["confusion_matrix"] = confusion_matrix(truth, guess, labels=classes).tolist()
    report["per_class"] = per_class
    return report


def summarize_folds(folds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation (``ddof=1``) of every scalar metric.

    Folds where a metric is undefined (``nan``) are left out of its summary.

    Examples
    --------
    >>> table = pd.DataFrame({"uar": [0.5, 0.7], "accuracy": [0.6, 0.6],
    ...                       "f1_macro": [0.5, 0.5], "roc_auc_ovr": [0.8, float("nan")]})
    >>> summary = summarize_folds(table)
    >>> round(summary["uar"]["std"], 4), summary["roc_auc_ovr"]["mean"]
    (0.1414, 0.8)
    """
    summary: dict[str, dict[str, float]] = {}
    for metric in SCALAR_METRICS:
        values = folds[metric].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else float("nan")
        std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        summary[metric] = {"mean": mean, "std": std}
    return summary


def aggregate_recordings(predictions: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Average segment probabilities per recording and pick the most probable class.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Segment rows with a ``path`` column and ``score_<class>`` columns; the
        columns ``group``, ``fold`` and ``label`` are carried over when present.
    classes : list of str
        Class names, sorted.

    Returns
    -------
    pandas.DataFrame
        One row per recording, in order of first appearance: ``path``, the
        carried-over columns, ``predicted`` and the mean ``score_<class>``.

    Examples
    --------
    >>> rows = pd.DataFrame({"path": ["r1", "r1", "r2"], "predicted": ["a", "b", "b"],
    ...                      "score_a": [0.9, 0.4, 0.1], "score_b": [0.1, 0.6, 0.9]})
    >>> aggregate_recordings(rows, ["a", "b"])[["path", "predicted"]].values.tolist()
    [['r1', 'a'], ['r2', 'b']]
    """
    columns = score_columns(classes)
    carried = [name for name in ("group", "fold", "label") if name in predictions.columns]
    grouped = predictions.groupby("path", sort=False)
    table = (
        grouped[carried].first().reset_index()
        if carried
        else grouped.size().reset_index()[["path"]]
    )
    means = grouped[columns].mean().reset_index(drop=True)
    table["predicted"] = np.asarray(classes)[np.argmax(means.to_numpy(), axis=1)]
    for name in columns:
        table[name] = means[name].to_numpy()
    return table[["path", *carried, "predicted", *columns]]


def _effective_groups(features: FeatureSet) -> NDArray[np.str_]:
    """The ``group`` of each segment, or its recording when it has none."""
    groups = np.asarray(features.groups).astype(str)
    paths = np.asarray(features.paths).astype(str)
    return np.where(groups == "", paths, groups)


def _check_splittable(labels: NDArray[np.str_], groups: NDArray[np.str_], n_splits: int) -> None:
    classes = np.unique(labels)
    if classes.size < 2:
        raise DatasetError(f"cross-validation needs at least two classes, found {classes.tolist()}")
    n_groups = np.unique(groups).size
    if n_groups < n_splits:
        raise DatasetError(
            f"only {n_groups} groups for evaluation.n_splits={n_splits}; every fold needs at "
            "least one group, so lower evaluation.n_splits or add data"
        )
    frame = pd.DataFrame({"label": labels, "group": groups}).drop_duplicates()
    per_class = frame.groupby("label")["group"].nunique()
    sparse = per_class[per_class < n_splits]
    if not sparse.empty:
        _LOGGER.warning(
            "classes with fewer groups than folds will be missing from some test folds: %s",
            ", ".join(f"{name} ({count})" for name, count in sparse.items()),
        )


def _fold_rows(
    table: pd.DataFrame, train_counts: list[int], classes: list[str]
) -> list[dict[str, Any]]:
    """Per-fold metrics of a prediction table with a ``fold`` column."""
    rows = []
    columns = score_columns(classes)
    for fold, n_train in enumerate(train_counts):
        part = table[table["fold"] == fold]
        scores = classification_scores(part["label"], part["predicted"], part[columns], classes)
        rows.append({"fold": fold, "n_train": n_train, "n_test": len(part), **scores})
    return rows


def _level_metrics(
    table: pd.DataFrame, train_counts: list[int], classes: list[str]
) -> dict[str, Any]:
    folds = pd.DataFrame(_fold_rows(table, train_counts, classes))
    pooled = pooled_report(
        table["label"], table["predicted"], table[score_columns(classes)], classes
    )
    return {
        "n_items": len(table),
        "folds": folds.to_dict(orient="records"),
        "summary": summarize_folds(folds),
        "pooled": pooled,
    }


def cross_validate(features: FeatureSet, config: Config) -> CrossValidationResult:
    """Stratified, grouped k-fold cross-validation of the configured model.

    Parameters
    ----------
    features : FeatureSet
        Segments written by ``prepare``.
    config : Config
        Complete configuration; ``evaluation.n_splits``, ``seed`` and ``model``
        are used.

    Returns
    -------
    CrossValidationResult
        Out-of-fold predictions and metrics at segment and recording level.

    Raises
    ------
    DatasetError
        With fewer than two classes, or fewer groups than folds.

    References
    ----------
    .. [1] R. Kohavi, "A study of cross-validation and bootstrap for accuracy
       estimation and model selection," in Proc. IJCAI 1995, pp. 1137-1143.
    .. [2] B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion
       Challenge," in Proc. Interspeech 2009, pp. 312-315.
    """
    labels = np.asarray(features.y).astype(str)
    groups = _effective_groups(features)
    n_splits = config.evaluation.n_splits
    _check_splittable(labels, groups, n_splits)
    classes = sorted(set(labels.tolist()))
    template = build_model(config)  # fails before any work when a dependency is missing
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    fold_of = np.full(labels.size, -1, dtype=np.int64)
    scores = np.zeros((labels.size, len(classes)))
    segment_train_counts: list[int] = []
    recording_train_counts: list[int] = []
    paths = np.asarray(features.paths).astype(str)
    for fold, (train, test) in enumerate(splitter.split(features.X, labels, groups)):
        _LOGGER.info(
            "fold %d/%d: training on %d segments, testing on %d",
            fold + 1,
            n_splits,
            train.size,
            test.size,
        )
        model = fit_estimator(clone(template), features.X[train], labels[train])
        probabilities = np.asarray(model.predict_proba(features.X[test]), dtype=np.float64)
        known = [classes.index(str(name)) for name in model.classes_]
        scores[np.ix_(test, known)] = probabilities
        fold_of[test] = fold
        segment_train_counts.append(int(train.size))
        recording_train_counts.append(int(np.unique(paths[train]).size))

    columns = score_columns(classes)
    predictions = pd.DataFrame(
        {
            "path": paths,
            "segment": features.segment_indices(),
            "group": np.asarray(features.groups).astype(str),
            "fold": fold_of,
            "label": labels,
            "predicted": np.asarray(classes)[np.argmax(scores, axis=1)],
        }
    )
    for position, name in enumerate(columns):
        predictions[name] = scores[:, position]
    recordings = aggregate_recordings(predictions, classes)

    segment_level = _level_metrics(predictions, segment_train_counts, classes)
    recording_level = _level_metrics(recordings, recording_train_counts, classes)
    folds = pd.DataFrame(segment_level["folds"])
    metrics: dict[str, Any] = {
        "task": "classification",
        "n_items": segment_level["n_items"],
        "n_splits": n_splits,
        "seed": config.seed,
        "folds": segment_level["folds"],
        "summary": segment_level["summary"],
        "pooled": segment_level["pooled"],
        "recordings": recording_level,
    }
    _LOGGER.info(
        "UAR %.3f (segments), %.3f (recordings)",
        segment_level["pooled"]["uar"],
        recording_level["pooled"]["uar"],
    )
    return CrossValidationResult(
        predictions=predictions, folds=folds, metrics=metrics, recording_predictions=recordings
    )
