"""Dataset index (``dataset.csv``) and the model inputs written by ``prepare``.

The index is one CSV file. Its ``path`` column holds WAV files relative to the
folder of the CSV (POSIX separators), ``label`` the class name and the optional
``group`` column the speaker or patient: every recording of one group stays in
the same cross-validation fold. File names carry no meaning; labels and groups
are read from the columns only.

``prepare`` turns the index into a :class:`FeatureSet`, one flattened
spectrogram image per segment, and stores it as ``features.npz`` with the
keys ``X``, ``y``, ``groups``, ``paths``, ``segments`` and ``config_json``.
The archive is written with fixed time stamps, so the same input and settings
give a byte-identical file.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .errors import ConfigError, DatasetError

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: File name of the dataset index written by ``synthesize`` and ``index``.
DATASET_INDEX_NAME: str = "dataset.csv"
#: Columns every dataset index must have.
REQUIRED_COLUMNS: tuple[str, ...] = ("path", "label")
#: Audio file extensions picked up by :func:`index_audio_folder`.
AUDIO_SUFFIXES: tuple[str, ...] = (".wav",)

#: Time stamp written into every ``features.npz`` member (the zip epoch).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_NPZ_KEYS = ("X", "y", "groups", "paths", "segments", "config_json")


@dataclass(frozen=True, eq=False)
class FeatureSet:
    """Model inputs: one row per segment.

    Attributes
    ----------
    X : numpy.ndarray of uint8, shape (n_items, n_bands * n_frames)
        Flattened spectrogram images (row-major, lowest band first).
    y : numpy.ndarray of str, shape (n_items,)
        Class name of each segment.
    groups : numpy.ndarray of str, shape (n_items,)
        Speaker or patient of each segment; empty strings when the index has
        no ``group`` column.
    paths : numpy.ndarray of str, shape (n_items,)
        Recording of each segment, as written in the dataset index.
    config : dict
        The complete configuration used by ``prepare`` (``Config.to_dict()``).
    segments : numpy.ndarray of int64, shape (n_items,), optional
        Index of the segment within its recording, counted from 0.
    """

    X: NDArray[np.uint8]
    y: NDArray[np.str_]
    groups: NDArray[np.str_]
    paths: NDArray[np.str_]
    config: dict[str, Any]
    segments: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise DatasetError(f"X must be two-dimensional, got shape {self.X.shape}")
        n_items = self.X.shape[0]
        columns = {"y": self.y, "groups": self.groups, "paths": self.paths}
        if self.segments is not None:
            columns["segments"] = self.segments
        for name, values in columns.items():
            if values.shape != (n_items,):
                raise DatasetError(
                    f"{name} must have one entry per row of X ({n_items}), got shape {values.shape}"
                )

    @property
    def n_items(self) -> int:
        """Number of segments."""
        return int(self.X.shape[0])

    @property
    def classes(self) -> list[str]:
        """Class names in sorted order."""
        return sorted({str(label) for label in self.y})

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(n_bands, n_frames)`` of the images, from the stored configuration."""
        spectrogram_config = self.config.get("spectrogram", {})
        return (int(spectrogram_config["n_bands"]), int(spectrogram_config["n_frames"]))

    def images(self) -> NDArray[np.uint8]:
        """The images, shape ``(n_items, n_bands, n_frames)``."""
        return self.X.reshape(self.n_items, *self.image_shape)

    def segment_indices(self) -> NDArray[np.int64]:
        """``segments``, or zeros when the set was built without segmentation."""
        if self.segments is None:
            return np.zeros(self.n_items, dtype=np.int64)
        return self.segments


def load_dataset_index(path: str | Path) -> pd.DataFrame:
    """Read and validate a dataset index.

    Parameters
    ----------
    path : str or pathlib.Path
        The CSV file.

    Returns
    -------
    pandas.DataFrame
        One row per recording with the columns ``path`` (absolute),
        ``source`` (the path as written, POSIX), ``label`` and ``group``
        (empty strings when the index has none), plus any extra columns.

    Raises
    ------
    FileNotFoundError
        If the index does not exist.
    DatasetError
        If a required column is missing, a value is empty or listed files do
        not exist.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     _ = (Path(folder) / "a.wav").write_bytes(b"")
    ...     _ = (Path(folder) / "dataset.csv").write_text("path,label\\na.wav,x\\n")
    ...     index = load_dataset_index(Path(folder) / "dataset.csv")
    >>> index["source"].tolist(), index["group"].tolist(), Path(index["path"][0]).is_absolute()
    (['a.wav'], [''], True)
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"dataset index not found: {source}. Point data.dataset (or --dataset) at your own "
            "dataset.csv, or run 'voice-spectral-classifier synthesize' first"
        )
    try:
        table = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{source} is not a readable CSV file: {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise DatasetError(f"{source} lacks the column(s) {missing}; found {list(table.columns)}")
    if table.empty:
        raise DatasetError(f"{source} lists no recordings")
    for column in REQUIRED_COLUMNS:
        empty = table[column].str.strip() == ""
        if empty.any():
            rows = (np.flatnonzero(empty.to_numpy()) + 2).tolist()[:5]
            raise DatasetError(f"{source}: empty {column!r} on line(s) {rows}")
    if "group" not in table.columns:
        table["group"] = ""
    base = source.resolve().parent
    table["source"] = table["path"].str.replace("\\", "/", regex=False)
    table["path"] = [str((base / entry).resolve()) for entry in table["source"]]
    absent = [
        entry
        for entry, full in zip(table["source"], table["path"], strict=True)
        if not Path(full).is_file()
    ]
    if absent:
        listed = ", ".join(absent[:5])
        more = f" ... and {len(absent) - 5} more" if len(absent) > 5 else ""
        raise DatasetError(
            f"{len(absent)} file(s) listed in {source} do not exist (paths are relative to "
            f"{base.as_posix()}): {listed}{more}"
        )
    ordered = ["path", "source", "label", "group"]
    return table[ordered + [column for column in table.columns if column not in ordered]]


def index_audio_folder(folder: str | Path) -> Path:
    """Write ``dataset.csv`` for a ``<folder>/<label>/<file>.wav`` layout.

    Every sub-folder is a class. Each recording becomes its own group (its
    path without the extension); edit the ``group`` column when one speaker
    has several recordings, so they stay in the same fold.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder holding one sub-folder per class.

    Returns
    -------
    pathlib.Path
        The written index, ``<folder>/dataset.csv``.

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist.
    DatasetError
        If no audio file is found.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     for name in ("low/a.wav", "high/b.wav"):
    ...         (Path(folder) / name).parent.mkdir(exist_ok=True)
    ...         _ = (Path(folder) / name).write_bytes(b"")
    ...     print(index_audio_folder(folder).read_text(), end="")
    path,label,group
    high/b.wav,high,high/b
    low/a.wav,low,low/a
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"folder not found: {root}")
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("*/*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise DatasetError(
            f"no {'/'.join(AUDIO_SUFFIXES)} files found in the class sub-folders of {root}"
        )
    table = pd.DataFrame(
        {
            "path": files,
            "label": [entry.split("/")[0] for entry in files],
            "group": [entry.rsplit(".", 1)[0] for entry in files],
        }
    )
    destination = root / DATASET_INDEX_NAME
    table.to_csv(destination, index=False, lineterminator="\n")
    _LOGGER.info("indexed %d recordings in %d classes", len(table), table["label"].nunique())
    return destination


def items_table(features: FeatureSet) -> pd.DataFrame:
    """One row per segment: ``path``, ``label``, ``group``, ``segment`` (``items.csv``)."""
    return pd.DataFrame(
        {
            "path": features.paths,
            "label": features.y,
            "group": features.groups,
            "segment": features.segment_indices(),
        }
    )


def _write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    """``np.savez_compressed`` with fixed member time stamps (reproducible bytes)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_EPOCH)
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, buffer.getvalue())


def save_features(path: str | Path, features: FeatureSet) -> Path:
    """Write a :class:`FeatureSet` as ``features.npz`` and return the path.

    Examples
    --------
    >>> import tempfile
    >>> data = FeatureSet(
    ...     X=np.zeros((2, 4), dtype=np.uint8), y=np.array(["a", "b"]),
    ...     groups=np.array(["", ""]), paths=np.array(["a.wav", "b.wav"]),
    ...     config={"seed": 0}, segments=np.array([0, 0]),
    ... )
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     loaded = load_features(save_features(Path(folder) / "features.npz", data))
    >>> loaded.y.tolist(), loaded.config, loaded.X.dtype.name
    (['a', 'b'], {'seed': 0}, 'uint8')
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": np.ascontiguousarray(features.X),
        "y": np.asarray(features.y, dtype=str),
        "groups": np.asarray(features.groups, dtype=str),
        "paths": np.asarray(features.paths, dtype=str),
        "segments": features.segment_indices().astype(np.int64),
        "config_json": np.array(json.dumps(features.config, ensure_ascii=False)),
    }
    _write_npz(destination, arrays)
    return destination


def _differing_keys(
    stored: Mapping[str, Any], current: Mapping[str, Any], prefix: str
) -> list[str]:
    keys = sorted(set(stored) | set(current))
    return [f"{prefix}.{key}" for key in keys if stored.get(key) != current.get(key)]


def load_features(path: str | Path, config: Config | None = None) -> FeatureSet:
    """Read ``features.npz``; with ``config``, check it matches the feature settings.

    Parameters
    ----------
    path : str or pathlib.Path
        File written by :func:`save_features`.
    config : Config, optional
        Current configuration. Its ``FEATURE_SECTIONS`` must equal the ones
        stored by ``prepare``.

    Returns
    -------
    FeatureSet

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DatasetError
        If the file is not a features archive of this package.
    ConfigError
        If the feature settings differ from the ones ``prepare`` used.
    """
    from .config import FEATURE_SECTIONS  # deferred: config imports this package's modules

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"features file not found: {source}. Run 'prepare' first")
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = [key for key in _NPZ_KEYS if key not in archive.files]
            if missing:
                raise DatasetError(f"{source} is not a features file; missing {missing}")
            stored_config = json.loads(str(archive["config_json"]))
            features = FeatureSet(
                X=archive["X"],
                y=archive["y"],
                groups=archive["groups"],
                paths=archive["paths"],
                config=stored_config,
                segments=archive["segments"].astype(np.int64),
            )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"{source} is not a readable features file: {exc}") from exc
    if config is not None:
        current = config.to_dict()
        differing: list[str] = []
        for section in FEATURE_SECTIONS:
            differing += _differing_keys(
                stored_config.get(section, {}), current.get(section, {}), section
            )
        if differing:
            raise ConfigError(
                f"feature settings differ from the ones 'prepare' used: {', '.join(differing)}; "
                "run 'prepare' again or use the same --config"
            )
    return features
