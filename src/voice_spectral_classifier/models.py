"""Estimator registry and model files.

Three classifiers share one interface (``fit``, ``predict``,
``predict_proba``, ``classes_``), so cross-validation and training never
branch on the model type:

* ``cnn`` -- the spectrogram CNN of :mod:`voice_spectral_classifier.cnn`
  (needs the ``dl`` extra);
* ``mlp`` -- standardization, an optional principal-component projection and
  a multilayer perceptron trained by back-propagation (Rumelhart et al.,
  1986) on the flattened image; it needs nothing beyond scikit-learn and is
  the baseline for the CNN;
* ``logistic`` -- the same preprocessing with multinomial logistic regression.

Everything learned from data (scaler, projection, mean image) lives inside the
estimator, so cross-validation fits it on the training folds only.
``model.class_weight = "balanced"`` counters class imbalance: through a
weighted loss where the model supports one, and by random oversampling of the
minority classes for the MLP, which has no class weights.

Model files are a dictionary with the keys ``format``, ``package_version``,
``config``, ``classes`` and either ``estimator`` (a scikit-learn pipeline,
stored with joblib in ``model.joblib``) or ``state_dict`` (CNN tensors only,
stored with :func:`torch.save` in ``model.pt`` and read back with
``weights_only=True``).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .cnn import SpectrogramCNNClassifier
from .errors import ConfigError, ModelFileError, require_module

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: Values of ``model.class_weight`` other than ``None``.
CLASS_WEIGHTS: tuple[str, ...] = ("balanced",)
#: Torch devices accepted by ``model.device``.
DEVICES: tuple[str, ...] = ("cpu", "cuda")
#: Format tag stored in, and required of, every model file.
MODEL_FORMAT: str = "voice_spectral_classifier/model-v1"

#: Iteration limit of the logistic-regression solver.
_LOGISTIC_MAX_ITER = 2000
_BUNDLE_KEYS = ("format", "package_version", "config", "classes")


class OversampledClassifier(ClassifierMixin, BaseEstimator):
    """Fit a classifier on data in which every class is as frequent as the largest.

    Rows of the minority classes are drawn again at random (with replacement)
    until each class reaches the majority count; prediction is unchanged.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Classifier to fit on the balanced data (cloned, never modified).
    random_state : int, optional
        Seed of the resampling.

    Examples
    --------
    >>> from sklearn.dummy import DummyClassifier
    >>> X, y = np.zeros((5, 1)), np.array(["a", "a", "a", "a", "b"])
    >>> model = OversampledClassifier(DummyClassifier(strategy="prior")).fit(X, y)
    >>> model.predict_proba(X[:1]).round(3).tolist()
    [[0.5, 0.5]]
    """

    def __init__(self, estimator: BaseEstimator, random_state: int = 0) -> None:
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> OversampledClassifier:
        """Oversample the minority classes, then fit a clone of ``estimator``."""
        features = np.asarray(X)
        labels = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        _classes, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
        codes = codes.ravel()
        extra = [
            rng.choice(np.flatnonzero(codes == code), size=counts.max() - count, replace=True)
            for code, count in enumerate(counts)
        ]
        rows = np.concatenate([np.arange(labels.size), *extra]).astype(np.intp)
        self.estimator_ = clone(self.estimator).fit(features[rows], labels[rows])
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict with the wrapped classifier."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Class probabilities from the wrapped classifier."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def _projection_steps(config: Config) -> list[tuple[str, BaseEstimator]]:
    """Standardization, then an optional principal-component projection."""
    steps: list[tuple[str, BaseEstimator]] = [("standardize", StandardScaler())]
    if config.model.n_components is not None:
        steps.append(
            ("project", PCA(n_components=config.model.n_components, random_state=config.seed))
        )
    return steps


def _cnn(config: Config) -> BaseEstimator:
    model_config = config.model
    return SpectrogramCNNClassifier(
        config.spectrogram.image_shape,
        n_epochs=model_config.n_epochs,
        learning_rate=model_config.learning_rate,
        batch_size=model_config.batch_size,
        weight_decay=model_config.weight_decay,
        dropout=model_config.dropout,
        class_weight=model_config.class_weight,
        device=model_config.device,
        random_state=config.seed,
    )


def _mlp(config: Config) -> BaseEstimator:
    model_config = config.model
    network = MLPClassifier(
        hidden_layer_sizes=model_config.hidden_units,
        alpha=model_config.weight_decay,
        batch_size=model_config.batch_size,
        learning_rate_init=model_config.learning_rate,
        max_iter=model_config.n_epochs,
        early_stopping=False,
        random_state=config.seed,
    )
    classifier: BaseEstimator = network
    if model_config.class_weight == "balanced":
        classifier = OversampledClassifier(network, random_state=config.seed)
    return Pipeline([*_projection_steps(config), ("classify", classifier)])


def _logistic(config: Config) -> BaseEstimator:
    classifier = LogisticRegression(
        C=config.model.logistic_c,
        class_weight=config.model.class_weight,
        max_iter=_LOGISTIC_MAX_ITER,
    )
    return Pipeline([*_projection_steps(config), ("classify", classifier)])


_FACTORIES: dict[str, Callable[[Config], BaseEstimator]] = {
    "cnn": _cnn,
    "mlp": _mlp,
    "logistic": _logistic,
}
#: Model names accepted by :func:`build_model` and ``model.name``.
MODEL_NAMES: tuple[str, ...] = tuple(_FACTORIES)


def build_model(config: Config) -> BaseEstimator:
    """Create the untrained estimator named by ``config.model.name``.

    Parameters
    ----------
    config : Config
        Complete configuration; the ``model`` section and ``seed`` are used,
        and the image size comes from ``spectrogram``.

    Returns
    -------
    sklearn.base.BaseEstimator
        A :class:`~sklearn.pipeline.Pipeline` for ``mlp`` and ``logistic``, a
        :class:`~voice_spectral_classifier.cnn.SpectrogramCNNClassifier` for ``cnn``.

    Raises
    ------
    ConfigError
        For an unknown model name.
    MissingDependencyError
        For ``cnn`` when PyTorch is not installed.

    References
    ----------
    .. [1] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning
       representations by back-propagating errors," Nature, vol. 323,
       pp. 533-536, 1986.

    Examples
    --------
    >>> from voice_spectral_classifier.config import Config
    >>> model = build_model(Config().override({"model.name": "mlp"}))
    >>> [name for name, _ in model.steps]
    ['standardize', 'classify']
    """
    name = config.model.name
    if name not in _FACTORIES:
        raise ConfigError(f"unknown model {name!r}; choose one of {list(MODEL_NAMES)}")
    if name == "cnn":
        require_module("torch", extra="dl", feature="the CNN model")
    return _FACTORIES[name](config)


def fit_estimator(estimator: BaseEstimator, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
    """Fit an estimator, routing its warnings into the log instead of the console.

    A solver that stops before converging is reported once, with the setting
    that controls it.

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted estimator.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        _LOGGER.warning(
            "the optimizer stopped before converging; "
            "raise model.n_epochs if the scores look unstable"
        )
    for message in sorted(
        {str(item.message) for item in caught if not issubclass(item.category, ConvergenceWarning)}
    ):
        _LOGGER.warning("%s", message)
    return estimator


def model_filename(config: Config) -> str:
    """``model.pt`` for the CNN, ``model.joblib`` for the scikit-learn models.

    Examples
    --------
    >>> from voice_spectral_classifier.config import Config
    >>> model_filename(Config()), model_filename(Config().override({"model.name": "mlp"}))
    ('model.pt', 'model.joblib')
    """
    return "model.pt" if config.model.name == "cnn" else "model.joblib"


def save_model(
    path: str | Path,
    estimator: Any,
    config: Config,
    classes: Sequence[str] | None = None,
) -> Path:
    """Write a trained estimator with its configuration and class names.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination (``model.pt`` for the CNN, ``model.joblib`` otherwise).
    estimator : Any
        Trained estimator.
    config : Config
        Configuration it was trained with.
    classes : sequence of str, optional
        Class names; read from ``estimator.classes_`` when omitted.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    from . import __version__  # deferred: the package __init__ imports this module

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = [str(name) for name in (estimator.classes_ if classes is None else classes)]
    bundle: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "package_version": __version__,
        "config": config.to_dict(),
        "classes": names,
    }
    if isinstance(estimator, SpectrogramCNNClassifier):
        torch = require_module("torch", extra="dl", feature="saving the CNN model")
        bundle["state_dict"] = estimator.state_dict()
        torch.save(bundle, destination)
    else:
        bundle["estimator"] = estimator
        joblib.dump(bundle, destination)
    _LOGGER.info("model written to %s", destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model file written by :func:`save_model`.

    Only load files from sources you trust: joblib files are pickles and can
    run code when they are read.

    Parameters
    ----------
    path : str or pathlib.Path
        ``model.joblib`` or ``model.pt``.

    Returns
    -------
    dict
        The bundle (``format``, ``package_version``, ``config``, ``classes``)
        with a ready-to-use ``estimator``; CNN files also keep ``state_dict``.
        A CNN trained with ``model.device = "cuda"`` runs on the CPU when
        CUDA is not available, with a warning; ``config`` is left as stored.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ModelFileError
        If the file is unreadable, incomplete or not written by this package.
    """
    from .config import Config  # deferred: config imports this module

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model file not found: {source}. Run 'train' first")
    if source.suffix == ".pt":
        torch = require_module("torch", extra="dl", feature="loading a CNN model")
        try:
            bundle = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as exc:  # torch raises several types for a foreign file
            raise ModelFileError(f"{source} is not a readable model file: {exc}") from exc
        _check_bundle(bundle, source, "state_dict")
        stored_config = Config.from_dict(bundle["config"])
        if stored_config.model.device == "cuda" and not torch.cuda.is_available():
            # The tensors are stored on the CPU, so a model trained on a GPU still
            # runs here; the bundle keeps the configuration it was trained with.
            _LOGGER.warning(
                "%s was trained with model.device='cuda', which is not available here; "
                "running it on the CPU",
                source.as_posix(),
            )
            stored_config = stored_config.override({"model.device": "cpu"})
        cnn = _cnn(stored_config)
        bundle["estimator"] = cnn.load_state_dict(bundle["state_dict"], bundle["classes"])
        return bundle
    try:
        bundle = joblib.load(source)
    except Exception as exc:  # unpickling raises many types for a foreign file
        raise ModelFileError(f"{source} is not a readable model file: {exc}") from exc
    _check_bundle(bundle, source, "estimator")
    return bundle


def _check_bundle(bundle: Any, source: Path, payload: str) -> None:
    """Raise ModelFileError unless ``bundle`` is a complete bundle of this package."""
    if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
        found = bundle.get("format") if isinstance(bundle, dict) else type(bundle).__name__
        raise ModelFileError(
            f"{source} was not written by this package "
            f"(format {found!r}, expected {MODEL_FORMAT!r})"
        )
    missing = [key for key in (*_BUNDLE_KEYS, payload) if key not in bundle]
    if missing:
        raise ModelFileError(f"{source} is incomplete; missing {missing}")
