"""Command line, end to end on synthetic data."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from voice_spectral_classifier.cli import app
from voice_spectral_classifier.logging_utils import _OWN_HANDLER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUICK = PROJECT_ROOT / "configs" / "quick_demo.yaml"
COMMANDS = ("synthesize", "index", "prepare", "evaluate", "train", "predict")
MANIFEST_KEYS = [
    "command",
    "created_utc",
    "package_version",
    "git_commit",
    "seed",
    "versions",
    "inputs",
    "config",
    "details",
]

RUNNER = CliRunner()


def run(*args):
    return RUNNER.invoke(app, [str(a) for a in args])


def _detach_cli_handlers():
    """The CLI logs to the runner's temporary stderr; detach that handler afterwards."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _OWN_HANDLER, False):
            root.removeHandler(handler)


@pytest.fixture(autouse=True)
def _restore_root_logger():
    level = logging.getLogger().level
    yield
    _detach_cli_handlers()
    logging.getLogger().setLevel(level)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Synthetic data and features shared by the CLI tests."""
    root = tmp_path_factory.mktemp("cli")
    data = root / "data"
    assert run("synthesize", "--config", QUICK, "--out", data).exit_code == 0
    prepare = root / "prepare"
    result = run("prepare", "--config", QUICK, "--dataset", data / "dataset.csv", "--out", prepare)
    assert result.exit_code == 0, result.output
    _detach_cli_handlers()
    return root


def test_the_whole_workflow_writes_every_documented_file(prepared):
    data, root = prepared / "data", prepared
    assert {"dataset.csv", "config.yaml", "manifest.json"} <= {p.name for p in data.iterdir()}
    for name in ("features.npz", "items.csv", "class_spectrograms.png", "class_spectrograms.csv"):
        assert (root / "prepare" / name).is_file(), name
    features = root / "prepare" / "features.npz"

    evaluated = run("evaluate", features, "--config", QUICK, "--out", root / "evaluate")
    assert evaluated.exit_code == 0, evaluated.output
    assert "UAR        : " in evaluated.output and "UAR (rec.) : " in evaluated.output
    for name in (
        "metrics.json",
        "folds.csv",
        "predictions.csv",
        "recording_predictions.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "recording_confusion_matrix.csv",
        "recording_confusion_matrix.png",
        "config.yaml",
        "manifest.json",
    ):
        assert (root / "evaluate" / name).is_file(), name
    manifest = json.loads((root / "evaluate" / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest) == MANIFEST_KEYS and manifest["command"] == "evaluate"
    columns = pd.read_csv(root / "evaluate" / "predictions.csv").columns.tolist()
    assert columns == [
        "path",
        "segment",
        "group",
        "fold",
        "label",
        "predicted",
        "score_breathy",
        "score_modal",
        "score_rough",
    ]

    trained = run("train", features, "--config", QUICK, "--out", root / "train")
    assert trained.exit_code == 0, trained.output
    model = root / "train" / "model.joblib"
    assert model.is_file()

    inputs = [data / "rough" / "rough_000.wav", data / "modal" / "modal_001.wav"]
    predicted = run("predict", *inputs, "--model", model, "--out", root / "predict")
    assert predicted.exit_code == 0, predicted.output
    table = pd.read_csv(root / "predict" / "predictions.csv")
    assert table.columns.tolist() == [
        "path",
        "segment",
        "predicted",
        "score_breathy",
        "score_modal",
        "score_rough",
    ]
    recordings = pd.read_csv(root / "predict" / "recording_predictions.csv")
    assert recordings["path"].tolist() == [path.as_posix() for path in inputs]
    assert predicted.output.strip().splitlines()[-1].startswith("written to ")


def test_evaluation_is_byte_for_byte_reproducible(prepared):
    features = prepared / "prepare" / "features.npz"
    for name in ("first", "second"):
        result = run("evaluate", features, "--config", QUICK, "--seed", 5, "--out", prepared / name)
        assert result.exit_code == 0, result.output
    for name in ("predictions.csv", "recording_predictions.csv", "metrics.json", "folds.csv"):
        assert (prepared / "first" / name).read_bytes() == (prepared / "second" / name).read_bytes()


def test_without_out_a_run_folder_is_created_under_output_root(prepared, tmp_path):
    features = prepared / "prepare" / "features.npz"
    result = run(
        "evaluate", features, "--config", QUICK, "--set", f"output.root={tmp_path.as_posix()}"
    )
    assert result.exit_code == 0, result.output
    folders = [path.name for path in tmp_path.iterdir()]
    assert len(folders) == 1 and folders[0].startswith("evaluate_") and folders[0].endswith("Z")


def test_index_writes_a_dataset_for_a_class_folder_layout(prepared, tmp_path):
    source = prepared / "data"
    copy = tmp_path / "copy"
    for path in source.glob("*/*.wav"):
        target = copy / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    result = run("index", copy)
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(copy / "dataset.csv")) == 18


def test_help_version_and_bare_invocation():
    result = run("--help")
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output
    assert run("--version").output.strip() == "voice-spectral-classifier 0.1.0"
    assert "Usage" in run().output


@pytest.mark.parametrize(
    "args",
    [
        ("evaluate", "{tmp}/absent.npz"),
        ("prepare", "--dataset", "{tmp}/absent.csv"),
        ("prepare", "--set", "spectrogram.n_band=3", "--dataset", "{tmp}/absent.csv"),
        ("prepare", "--set", "model.name=svm"),
        ("prepare", "--set", "model.name"),
        ("prepare", "--set", "segmentation.segment_s=Infinity"),
        ("prepare", "--config", "{tmp}/broken.yaml"),
        ("--log-level", "LOUD", "synthesize", "--out", "{tmp}/unused"),
        ("evaluate", "{features}", "--config", "{quick}", "--set", "spectrogram.n_bands=20"),
        ("predict", "{tmp}/a.wav", "--model", "{tmp}/absent.joblib"),
    ],
)
def test_expected_failures_exit_with_code_2(args, prepared, tmp_path):
    (tmp_path / "broken.yaml").write_text("model: [unclosed\n", encoding="utf-8")
    values = {
        "tmp": tmp_path.as_posix(),
        "features": (prepared / "prepare" / "features.npz").as_posix(),
        "quick": QUICK.as_posix(),
    }
    result = run(*(part.format(**values) for part in args))
    assert result.exit_code == 2, result.output
    assert "error:" in result.output or "Usage" in result.output
    assert isinstance(result.exception, SystemExit)


def test_feature_mismatch_names_the_changed_key(prepared):
    features = prepared / "prepare" / "features.npz"
    result = run("train", features, "--config", QUICK, "--set", "spectrogram.scale=mel")
    assert result.exit_code == 2
    assert "spectrogram.scale" in result.output and "run 'prepare' again" in result.output


def test_module_entry_point_lists_every_command():
    completed = subprocess.run(
        [sys.executable, "-m", "voice_spectral_classifier", "--help"],
        capture_output=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8"),
        check=True,
    )
    for command in COMMANDS:
        assert command in completed.stdout
