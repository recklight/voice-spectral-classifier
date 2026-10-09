"""Shared fixtures. Every fixture is synthesized in code; no recorded data is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_spectral_classifier import (
    Config,
    FeatureSet,
    prepare_features,
    write_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def fast_config():
    return Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")


@pytest.fixture
def synthetic_dataset(tmp_path, fast_config):
    return write_synthetic_dataset(tmp_path / "data", fast_config.synthetic, rng=0)


@pytest.fixture(scope="session")
def quick_dataset(tmp_path_factory):
    """One synthetic dataset (quick settings) shared by the tests that only read it."""
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    return write_synthetic_dataset(
        tmp_path_factory.mktemp("shared") / "data", config.synthetic, rng=0
    )


@pytest.fixture(scope="session")
def quick_features(quick_dataset):
    """Segment images of :func:`quick_dataset`, prepared once."""
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    return prepare_features(quick_dataset, config)


def make_toy_features(
    rng: np.random.Generator,
    *,
    classes: tuple[str, ...] = ("a", "b", "c"),
    groups_per_class: int = 6,
    recordings_per_group: int = 2,
    segments_per_recording: int = 3,
    separation: float = 40.0,
    with_groups: bool = True,
) -> FeatureSet:
    """Small random images whose class shifts the mean level; for fast evaluation tests."""
    config = Config().to_dict()
    config["spectrogram"].update(n_bands=4, n_frames=5)
    rows, labels, groups, paths, segments = [], [], [], [], []
    for position, label in enumerate(classes):
        for group in range(groups_per_class):
            for recording in range(recordings_per_group):
                path = f"{label}/{label}_g{group}_r{recording}.wav"
                for segment in range(segments_per_recording):
                    level = 100.0 + separation * position + rng.normal(0.0, 25.0, size=20)
                    rows.append(np.clip(level, 0, 255).astype(np.uint8))
                    labels.append(label)
                    groups.append(f"{label}_speaker_{group}" if with_groups else "")
                    paths.append(path)
                    segments.append(segment)
    return FeatureSet(
        X=np.stack(rows),
        y=np.asarray(labels),
        groups=np.asarray(groups),
        paths=np.asarray(paths),
        config=config,
        segments=np.asarray(segments, dtype=np.int64),
    )


@pytest.fixture
def make_toy():
    """The :func:`make_toy_features` factory, for tests that need a variant."""
    return make_toy_features
