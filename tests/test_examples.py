"""Every example script runs end to end on synthetic data (with --quick)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_has_a_test():
    tested = {"end_to_end_synthetic", "compare_spectrogram_scales", "aggregate_segment_scores"}
    assert {path.stem for path in EXAMPLES.glob("*.py")} == tested


def test_end_to_end_synthetic(tmp_path):
    out = tmp_path / "end_to_end_synthetic"
    assert load("end_to_end_synthetic").main(["--quick", "--out", str(out)]) == 0
    for name in (
        "data/dataset.csv",
        "features.npz",
        "items.csv",
        "metrics.json",
        "folds.csv",
        "predictions.csv",
        "recording_predictions.csv",
        "model.joblib",
        "new_predictions.csv",
        "new_recording_predictions.csv",
    ):
        assert (out / name).is_file(), name


def test_compare_spectrogram_scales(tmp_path):
    out = tmp_path / "compare_spectrogram_scales"
    assert load("compare_spectrogram_scales").main(["--quick", "--out", str(out)]) == 0
    table = pd.read_csv(out / "scale_comparison.csv")
    assert set(table["scale"]) == {"linear", "mel"}
    assert {"logistic", "mlp"} <= set(table["model"])
    assert (out / "scale_comparison.png").is_file()


def test_aggregate_segment_scores(tmp_path):
    out = tmp_path / "aggregate_segment_scores"
    assert load("aggregate_segment_scores").main(["--quick", "--out", str(out)]) == 0
    table = pd.read_csv(out / "aggregation_comparison.csv", keep_default_na=False)
    assert set(table["segments_kept"].astype(str)) == {"1", "all"}
    assert set(table["level"]) == {"segment", "recording"}
    assert (out / "per_class_recall.csv").is_file()
