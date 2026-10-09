"""A small convolutional network for spectrogram images (needs the ``dl`` extra).

The input is the representation used for small-footprint keyword spotting, a
40-band time-frequency image (Sainath & Parada, 2015) of one second, the clip
length of the Speech Commands dataset (Warden, 2018). The network is a plain
stack of small 3x3 convolutions (Simonyan & Zisserman, 2015): six blocks with
"same" padding, each followed by batch normalization (Ioffe & Szegedy, 2015)
and a rectified linear unit (Nair & Hinton, 2010), with dropout
(Srivastava et al., 2014) in front of the four deeper blocks:

::

    block  channels  after the block                      shape for a 40 x 98 image
    1      16        2x2 max pooling                      16 x 20 x 49
    2      32        2x2 max pooling, columns padded by 1 32 x 10 x 25
    3      64        -                                    64 x 10 x 25
    4      64        2x2 max pooling, columns padded by 1 64 x 5 x 13
    5      64        -                                    64 x 5 x 13
    6      64        maximum over the whole time axis     64 x 5 x 1

The 320 remaining values feed a fully connected layer with one output per
class, trained with cross-entropy and Adam (Kingma & Ba, 2015) in mini-batches
that are reshuffled every epoch. The last pooling takes the maximum over all
remaining time steps, so the size of the classifier's input does not depend on
the number of frames and the same layout fits any ``spectrogram.n_frames``; a
trained network accepts only the image size it was trained on. Images are
scaled to ``[0, 1]`` and the per-pixel mean image of the training data is
subtracted; that mean is a buffer of the network, so it is saved and restored
with the weights.

PyTorch is imported only inside the functions that need it, so this module
(and :func:`feature_map_shapes`) works without it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .errors import require_module

_LOGGER = logging.getLogger(__name__)

#: Output channels of the six convolution blocks.
CNN_CHANNELS: tuple[int, ...] = (16, 32, 64, 64, 64, 64)

#: Blocks followed by 2x2 max pooling, with that pooling's (row, column) padding.
_POOLING: dict[int, tuple[int, int]] = {0: (0, 0), 1: (0, 1), 3: (0, 1)}
#: Blocks preceded by dropout.
_DROPOUT_BLOCKS = frozenset({2, 3, 4, 5})
#: Images per forward pass at inference time.
_INFERENCE_BATCH = 256


def _pooled(size: int, padding: int) -> int:
    """Output length of a 2-wide, stride-2 max pooling."""
    return (size + 2 * padding - 2) // 2 + 1


def feature_map_shapes(
    image_shape: tuple[int, int] = (40, 98), channels: Sequence[int] = CNN_CHANNELS
) -> list[tuple[int, int, int]]:
    """Shape ``(channels, rows, columns)`` after each of the six convolution blocks.

    Plain arithmetic that runs without PyTorch; the layout table in the module
    docstring is its output for a 40 x 98 image.

    Parameters
    ----------
    image_shape : tuple of int, optional
        ``(n_bands, n_frames)`` of the input images.
    channels : sequence of int, optional
        Output channels of the six blocks.

    Returns
    -------
    list of tuple of int
        Six shapes; the flattened size of the last one feeds the classifier.

    Raises
    ------
    ValueError
        If ``channels`` does not have six entries or the image is too small
        for the three pooling stages.

    References
    ----------
    .. [1] K. Simonyan and A. Zisserman, "Very deep convolutional networks for
       large-scale image recognition," in Proc. ICLR 2015, arXiv:1409.1556.

    Examples
    --------
    >>> feature_map_shapes((40, 98))
    [(16, 20, 49), (32, 10, 25), (64, 10, 25), (64, 5, 13), (64, 5, 13), (64, 5, 1)]
    """
    if len(channels) != len(CNN_CHANNELS):
        raise ValueError(f"expected {len(CNN_CHANNELS)} channel counts, got {list(channels)}")
    rows, columns = int(image_shape[0]), int(image_shape[1])
    if rows < 1 or columns < 1:
        raise ValueError(f"image_shape must be positive, got {tuple(image_shape)}")
    shapes: list[tuple[int, int, int]] = []
    for block, width in enumerate(channels):
        if block in _POOLING:
            pad_rows, pad_columns = _POOLING[block]
            if rows + 2 * pad_rows < 2 or columns + 2 * pad_columns < 2:
                raise ValueError(
                    f"an image of shape {tuple(image_shape)} is too small for the pooling "
                    f"after block {block + 1}"
                )
            rows, columns = _pooled(rows, pad_rows), _pooled(columns, pad_columns)
        if block == len(channels) - 1:
            columns = 1
        shapes.append((int(width), rows, columns))
    return shapes


def parameter_count(
    n_classes: int,
    image_shape: tuple[int, int] = (40, 98),
    channels: Sequence[int] = CNN_CHANNELS,
) -> int:
    """Number of trainable parameters of the network.

    Examples
    --------
    >>> parameter_count(3)
    135651
    """
    total, previous = 0, 1
    for width in channels:
        total += previous * width * 9 + width  # 3x3 convolution with bias
        total += 2 * width  # batch-normalization scale and shift
        previous = width
    flat = int(np.prod(feature_map_shapes(image_shape, channels)[-1]))
    return total + flat * n_classes + n_classes


def _build_network(
    torch: Any,
    n_classes: int,
    image_shape: tuple[int, int],
    channels: Sequence[int],
    dropout: float,
) -> Any:
    """Build the network.

    The class subclasses ``torch.nn.Module``, so it is defined inside this
    function; importing the module then does not import torch.
    """
    nn = torch.nn
    flat = int(np.prod(feature_map_shapes(image_shape, channels)[-1]))

    class SpectrogramNetwork(nn.Module):
        """Six convolution blocks, a time-axis maximum and a linear classifier."""

        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            previous = 1
            for block, width in enumerate(channels):
                if block in _DROPOUT_BLOCKS:
                    layers.append(nn.Dropout(dropout))
                layers += [
                    nn.Conv2d(previous, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                ]
                if block in _POOLING:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=_POOLING[block]))
                previous = width
            layers.append(nn.AdaptiveMaxPool2d((None, 1)))
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(flat, n_classes)
            self.register_buffer("input_mean", torch.zeros(1, 1, *image_shape))

        def forward(self, images: Any) -> Any:
            hidden = self.features(images - self.input_mean)
            return self.classifier(torch.flatten(hidden, 1))

    return SpectrogramNetwork()


class SpectrogramCNNClassifier(ClassifierMixin, BaseEstimator):
    """The spectrogram CNN as a scikit-learn classifier.

    Cross-validation clones and fits it the same way as the MLP and logistic
    regression models.

    Parameters
    ----------
    image_shape : tuple of int, optional
        ``(n_bands, n_frames)`` of the images.
    n_epochs : int, optional
        Passes over the training data.
    learning_rate : float, optional
        Adam step size.
    batch_size : int, optional
        Mini-batch size; the last batch of an epoch may be smaller.
    weight_decay : float, optional
        L2 penalty added to the gradient by Adam.
    dropout : float, optional
        Dropout probability in front of blocks 3-6.
    class_weight : {"balanced", None}, optional
        ``balanced`` weights the cross-entropy of each class by
        ``n_items / (n_classes * n_items_of_class)``.
    device : {"cpu", "cuda"}, optional
        Where the network is trained and run.
    random_state : int, optional
        Seeds the weight initialization and the per-epoch shuffling.
    log_every : int, optional
        Log the mean training loss every this many epochs.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class names in sorted order (the columns of :meth:`predict_proba`).
    network_ : torch.nn.Module
        The trained network.

    Notes
    -----
    ``X`` is either flattened images, shape ``(n, n_bands * n_frames)``, or
    images, shape ``(n, n_bands, n_frames)``, with grey levels 0-255.

    References
    ----------
    .. [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based
       learning applied to document recognition," Proceedings of the IEEE,
       vol. 86, no. 11, pp. 2278-2324, 1998.
    .. [2] T. N. Sainath and C. Parada, "Convolutional neural networks for
       small-footprint keyword spotting," in Proc. Interspeech 2015, pp. 1478-1482.
    .. [3] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep
       network training by reducing internal covariate shift," in Proc. ICML
       2015, pp. 448-456.
    .. [4] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and
       R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from
       overfitting," Journal of Machine Learning Research, vol. 15,
       pp. 1929-1958, 2014.
    .. [5] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization,"
       in Proc. ICLR 2015.
    .. [6] V. Nair and G. E. Hinton, "Rectified linear units improve restricted
       Boltzmann machines," in Proc. ICML 2010, pp. 807-814.
    .. [7] K. Simonyan and A. Zisserman, "Very deep convolutional networks for
       large-scale image recognition," in Proc. ICLR 2015, arXiv:1409.1556.
    .. [8] P. Warden, "Speech Commands: A dataset for limited-vocabulary speech
       recognition," arXiv:1804.03209, 2018.
    """

    def __init__(
        self,
        image_shape: tuple[int, int] = (40, 98),
        *,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 60,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        class_weight: str | None = "balanced",
        device: str = "cpu",
        random_state: int = 0,
        log_every: int = 10,
    ) -> None:
        self.image_shape = image_shape
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state
        self.log_every = log_every

    def _images(self, X: ArrayLike) -> NDArray[np.float32]:
        """Validated images scaled to [0, 1], shape (n, 1, n_bands, n_frames)."""
        rows, columns = (int(size) for size in self.image_shape)
        array = np.asarray(X)
        if array.ndim == 2 and array.shape[1] == rows * columns:
            array = array.reshape(-1, rows, columns)
        elif not (array.ndim == 3 and array.shape[1:] == (rows, columns)):
            raise ValueError(
                f"expected images of shape (n, {rows}, {columns}) or (n, {rows * columns}), "
                f"got {array.shape}"
            )
        scaled = array.astype(np.float32) / np.float32(255.0)
        return np.ascontiguousarray(scaled[:, None, :, :])

    def _torch(self) -> Any:
        torch = require_module("torch", extra="dl", feature="the CNN model")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("model.device='cuda' was requested but CUDA is not available")
        return torch

    def _network(self, torch: Any) -> Any:
        return _build_network(
            torch,
            len(self.classes_),
            (int(self.image_shape[0]), int(self.image_shape[1])),
            CNN_CHANNELS,
            float(self.dropout),
        ).to(torch.device(self.device))

    def fit(self, X: ArrayLike, y: ArrayLike) -> SpectrogramCNNClassifier:
        """Train the network from scratch.

        Parameters
        ----------
        X : array_like
            Images with grey levels 0-255 (see the class notes).
        y : array_like, shape (n,)
            Class names.

        Returns
        -------
        SpectrogramCNNClassifier
            The fitted estimator.
        """
        torch = self._torch()
        images = self._images(X)
        labels = np.asarray(y)
        if labels.shape != (images.shape[0],):
            raise ValueError(f"y must hold one label per image, got shape {labels.shape}")
        self.classes_, targets = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("the CNN needs at least two classes to train on")
        torch.manual_seed(int(self.random_state))
        device = torch.device(self.device)
        network = self._network(torch)
        inputs = torch.from_numpy(images)
        expected = torch.from_numpy(targets.astype(np.int64).ravel())
        with torch.no_grad():
            network.input_mean.copy_(inputs.mean(dim=0, keepdim=True).to(device))
        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(expected.numpy(), minlength=self.classes_.size)
            balance = len(targets) / (self.classes_.size * np.maximum(counts, 1))
            weight = torch.tensor(balance, dtype=torch.float32, device=device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        shuffler = torch.Generator().manual_seed(int(self.random_state))
        n_items = inputs.shape[0]
        network.train()
        for epoch in range(1, int(self.n_epochs) + 1):
            order = torch.randperm(n_items, generator=shuffler)
            running = 0.0
            for start in range(0, n_items, int(self.batch_size)):
                batch = order[start : start + int(self.batch_size)]
                optimizer.zero_grad()
                logits = network(inputs[batch].to(device))
                loss = loss_function(logits, expected[batch].to(device))
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu()) * int(batch.shape[0])
            if epoch % max(int(self.log_every), 1) == 0 or epoch == int(self.n_epochs):
                _LOGGER.info(
                    "epoch %d/%d: training loss %.4f", epoch, self.n_epochs, running / n_items
                )
        network.eval()
        self.network_ = network
        self.n_features_in_ = int(self.image_shape[0]) * int(self.image_shape[1])
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Softmax class probabilities, columns in the order of ``classes_``."""
        check_is_fitted(self, "network_")
        torch = self._torch()
        images = torch.from_numpy(self._images(X))
        device = torch.device(self.device)
        self.network_.eval()
        chunks = [np.zeros((0, self.classes_.size))]
        with torch.no_grad():
            for start in range(0, images.shape[0], _INFERENCE_BATCH):
                logits = self.network_(images[start : start + _INFERENCE_BATCH].to(device))
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64))
        return np.concatenate(chunks, axis=0)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Most probable class of every image."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def state_dict(self) -> dict[str, Any]:
        """Weights, batch-normalization statistics and mean image, as CPU tensors."""
        check_is_fitted(self, "network_")
        return {key: value.detach().cpu() for key, value in self.network_.state_dict().items()}

    def load_state_dict(
        self, state: Mapping[str, Any], classes: Sequence[str]
    ) -> SpectrogramCNNClassifier:
        """Rebuild a trained network from :meth:`state_dict` and the class names.

        Parameters
        ----------
        state : mapping
            Tensors returned by :meth:`state_dict`.
        classes : sequence of str
            Class names in sorted order.

        Returns
        -------
        SpectrogramCNNClassifier
            ``self``, ready for :meth:`predict`.
        """
        torch = self._torch()
        self.classes_ = np.asarray(list(classes))
        network = self._network(torch)
        network.load_state_dict(dict(state))
        network.eval()
        self.network_ = network
        self.n_features_in_ = int(self.image_shape[0]) * int(self.image_shape[1])
        return self
