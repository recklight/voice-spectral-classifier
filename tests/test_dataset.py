"""Dataset index reading, folder indexing and the features.npz archive."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from voice_spectral_classifier import (
    Config,
    ConfigError,
    DatasetError,
    FeatureSet,
    index_audio_folder,
    items_table,
    load_dataset_index,
    load_features,
    save_features,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_paths_are_relative_to_the_index_folder_not_the_working_directory(tmp_path, monkeypatch):
    _touch(tmp_path / "audio" / "x" / "one.wav")
    index = tmp_path / "audio" / "dataset.csv"
    index.write_text("path,label,group\nx/one.wav,x,s1\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    table = load_dataset_index(index)
    assert Path(table.loc[0, "path"]) == (tmp_path / "audio" / "x" / "one.wav").resolve()
    assert table.loc[0, "source"] == "x/one.wav"
    assert list(table.columns[:4]) == ["path", "source", "label", "group"]


def test_windows_separators_are_accepted(tmp_path):
    _touch(tmp_path / "x" / "one.wav")
    (tmp_path / "dataset.csv").write_text("path,label\nx\\one.wav,x\n", encoding="utf-8")
    table = load_dataset_index(tmp_path / "dataset.csv")
    assert table.loc[0, "source"] == "x/one.wav"
    assert table.loc[0, "group"] == ""


def test_missing_columns_are_named_with_the_columns_found(tmp_path):
    (tmp_path / "dataset.csv").write_text("file,class\na.wav,x\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"lacks the column\(s\) \['path', 'label'\]; found"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_missing_files_are_listed_five_at_most(tmp_path):
    rows = "\n".join(f"missing_{number}.wav,x" for number in range(8))
    (tmp_path / "dataset.csv").write_text(f"path,label\n{rows}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"8 file\(s\).*missing_4\.wav \.\.\. and 3 more"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_empty_labels_are_reported_with_their_line(tmp_path):
    _touch(tmp_path / "a.wav")
    (tmp_path / "dataset.csv").write_text("path,label\na.wav,\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"empty 'label' on line\(s\) \[2\]"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_a_missing_index_explains_the_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthesize"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_an_empty_index_is_rejected(tmp_path):
    (tmp_path / "dataset.csv").write_text("path,label\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="lists no recordings"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_folder_layout_is_indexed_with_one_group_per_recording(tmp_path):
    for name in ("palsy/p2.wav", "palsy/p1.WAV", "neoplasm/n1.wav", "notes.txt", "palsy/x.txt"):
        _touch(tmp_path / name)
    index = index_audio_folder(tmp_path)
    table = pd.read_csv(index)
    assert index == tmp_path / "dataset.csv"
    assert table["path"].tolist() == ["neoplasm/n1.wav", "palsy/p1.WAV", "palsy/p2.wav"]
    assert table["label"].tolist() == ["neoplasm", "palsy", "palsy"]
    assert table["group"].tolist() == ["neoplasm/n1", "palsy/p1", "palsy/p2"]
    assert len(load_dataset_index(index)) == 3


def test_indexing_requires_audio_in_class_folders(tmp_path):
    _touch(tmp_path / "loose.wav")
    with pytest.raises(DatasetError, match=r"no \.wav files"):
        index_audio_folder(tmp_path)
    with pytest.raises(FileNotFoundError):
        index_audio_folder(tmp_path / "absent")


def _features() -> FeatureSet:
    config = Config().to_dict()
    config["spectrogram"].update(n_bands=2, n_frames=3)
    return FeatureSet(
        X=np.arange(18, dtype=np.uint8).reshape(3, 6),
        y=np.array(["a", "b", "a"]),
        groups=np.array(["s1", "s2", "s1"]),
        paths=np.array(["a/1.wav", "b/1.wav", "a/1.wav"]),
        config=config,
        segments=np.array([0, 0, 1], dtype=np.int64),
    )


def test_features_round_trip_and_are_byte_identical(tmp_path):
    original = _features()
    first = save_features(tmp_path / "one" / "features.npz", original)
    second = save_features(tmp_path / "two" / "features.npz", original)
    assert first.read_bytes() == second.read_bytes()
    loaded = load_features(first)
    for name in ("X", "y", "groups", "paths", "segments"):
        assert np.array_equal(getattr(loaded, name), getattr(original, name)), name
    assert loaded.config == original.config
    assert loaded.images().shape == (3, 2, 3)
    assert loaded.classes == ["a", "b"]
    with np.load(first, allow_pickle=False) as archive:
        assert sorted(archive.files) == sorted(
            ["X", "y", "groups", "paths", "segments", "config_json"]
        )
        assert archive["config_json"].ndim == 0


def test_items_table_has_one_row_per_segment():
    table = items_table(_features())
    assert list(table.columns) == ["path", "label", "group", "segment"]
    assert table["segment"].tolist() == [0, 0, 1]


def test_changed_feature_settings_are_named(tmp_path):
    path = save_features(tmp_path / "features.npz", _features())
    stored = Config.from_dict(_features().config)
    assert load_features(path, stored).n_items == 3
    changed = stored.override({"spectrogram.n_bands": 4, "model.n_epochs": 3})
    with pytest.raises(
        ConfigError, match=r"differ from the ones 'prepare' used: spectrogram\.n_bands;"
    ):
        load_features(path, changed)


def test_foreign_or_missing_feature_files_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'prepare' first"):
        load_features(tmp_path / "absent.npz")
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip archive")
    with pytest.raises(DatasetError, match="not a readable features file"):
        load_features(garbage)
    partial = tmp_path / "partial.npz"
    np.savez(partial, X=np.zeros((1, 2)))
    with pytest.raises(DatasetError, match="missing"):
        load_features(partial)


def test_feature_set_checks_lengths():
    with pytest.raises(DatasetError, match="one entry per row"):
        FeatureSet(
            X=np.zeros((2, 4), dtype=np.uint8),
            y=np.array(["a"]),
            groups=np.array(["", ""]),
            paths=np.array(["a", "b"]),
            config={},
        )


def test_the_synthetic_index_loads_with_absolute_paths(synthetic_dataset):
    table = load_dataset_index(synthetic_dataset)
    assert len(table) == 18 and table["label"].nunique() == 3
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in table["path"])
    assert table["source"].str.match(r"^(breathy|modal|rough)/\1_\d{3}\.wav$").all()
