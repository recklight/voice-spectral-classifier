"""Command-line interface.

Commands
--------
synthesize
    Write a synthetic dataset (WAV files and ``dataset.csv``) to try every command.
index
    Write ``dataset.csv`` for a ``<folder>/<label>/<file>.wav`` layout.
prepare
    Segment every recording of the dataset index and compute spectrogram images.
evaluate
    Cross-validate the configured model at segment and recording level.
train
    Train the configured model on every segment and save it.
predict
    Classify new recordings with a saved model.

Commands that use settings take ``--config`` (a YAML file) and any number of
``--set section.key=value`` overrides; ``--seed`` and ``--dataset`` are
shortcuts for ``--set seed=...`` and ``--set data.dataset=...``. Results go to
``--out``, or to a new ``<command>_<UTC time>`` folder under ``output.root``,
always together with ``config.yaml`` and ``manifest.json``; ``index`` only
writes ``<folder>/dataset.csv``. ``predict`` uses only the settings stored in
the model file.

Expected failures (a missing file, an invalid configuration, a missing
optional dependency) end with a one-line message and exit code 2; the
traceback is logged at DEBUG level.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from . import __version__
from .config import Config
from .dataset import index_audio_folder, items_table, load_features, save_features
from .errors import ConfigError
from .evaluation import aggregate_recordings, cross_validate
from .figures import (
    class_spectrograms_table,
    confusion_matrix_table,
    plot_class_spectrograms,
    plot_confusion_matrix,
    save_figure,
)
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import load_model, model_filename, save_model
from .pipeline import predict_inputs, prepare_features, train_model
from .synthetic import write_synthetic_dataset

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Local variables of a failing frame can hold whole datasets; never dump them.
    pretty_exceptions_show_locals=False,
    help=(
        "Classify pathological voices from fixed-size spectrogram images with a CNN, an MLP "
        "or logistic regression."
    ),
)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML configuration file.")
]
SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override one setting: section.key=value (repeatable)."),
]
SeedOption = Annotated[
    int | None, typer.Option("--seed", help="Master seed; overrides the configuration.")
]
DatasetOption = Annotated[
    Path | None,
    typer.Option("--dataset", "-d", help="Dataset index CSV; overrides data.dataset."),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out", "-o", help="Output folder (default: a new run folder under output.root)."
    ),
]
ModelOption = Annotated[Path, typer.Option("--model", "-m", help="Model file written by 'train'.")]


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"voice-spectral-classifier {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    try:
        configure_logging(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from None


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Turn the library's expected failures into a one-line message and exit code 2."""
    try:
        yield
    except (ValueError, FileNotFoundError, ImportError) as exc:
        _LOGGER.debug("command failed", exc_info=True)
        raise _fail(str(exc)) from None


def _coerce(raw: str) -> Any:
    """Parse a ``--set`` value: true/false, none/null (any case), JSON, else the string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _load_config(
    config_file: Path | None,
    overrides: list[str] | None,
    shortcuts: Mapping[str, Any] | None = None,
) -> Config:
    """Apply ``--config``, then every ``--set``, then the shortcuts that were given.

    A shortcut whose value is ``None`` was left off the command line and is skipped.
    """
    config = Config.from_yaml(config_file) if config_file is not None else Config()
    changes: dict[str, Any] = {}
    for item in overrides or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        changes[key.strip()] = _coerce(raw.strip())
    for key, value in (shortcuts or {}).items():
        if value is not None:
            changes[key] = value.as_posix() if isinstance(value, Path) else value
    return config.override(changes) if changes else config


def _report(label: str, value: object) -> None:
    """Print one line of the closing summary, the label padded to 11 characters."""
    typer.echo(f"{label:<11}: {value}")


def _run_folder(config: Config, command: str, out: Path | None) -> Path:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        return out
    return run_directory(config.output.root, command)


def _finish(
    folder: Path,
    command: str,
    config: Config,
    inputs: Sequence[Path],
    details: Mapping[str, Any],
) -> None:
    """Write ``config.yaml`` and ``manifest.json`` and print the closing line."""
    config.to_yaml(folder / "config.yaml")
    manifest = build_manifest(
        command=command, config=config.to_dict(), inputs=inputs, seed=config.seed, extra=details
    )
    write_manifest(folder / "manifest.json", manifest)
    typer.echo(f"written to {folder.as_posix()}")


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")


def _mean_std(summary: Mapping[str, Mapping[str, float]], metric: str) -> str:
    return f"{summary[metric]['mean']:.3f} +/- {summary[metric]['std']:.3f}"


@app.command("synthesize")
def synthesize_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Folder for the synthetic dataset.")
    ] = Path("data/synthetic"),
) -> None:
    """Write a synthetic dataset (WAV files and dataset.csv) to try the other commands on."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        index = write_synthetic_dataset(out, config.synthetic, rng=config.seed)
        table = pd.read_csv(index, dtype=str)
        _report("recordings", len(table))
        _report("speakers", table["group"].nunique())
        counts = table["label"].value_counts().sort_index()
        _report("classes", ", ".join(f"{name} {count}" for name, count in counts.items()))
        _finish(out, "synthesize", config, [], {"n_recordings": len(table)})


@app.command("index")
def index_command(
    folder: Annotated[
        Path, typer.Argument(help="Folder with one sub-folder of WAV files per class.")
    ],
) -> None:
    """Write dataset.csv for a <folder>/<label>/<file>.wav layout."""
    with _reported_errors():
        index = index_audio_folder(folder)
        table = pd.read_csv(index, dtype=str)
        _report("recordings", len(table))
        counts = table["label"].value_counts().sort_index()
        _report("classes", ", ".join(f"{name} {count}" for name, count in counts.items()))
        typer.echo(f"written to {index.as_posix()}")


@app.command("prepare")
def prepare_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
    figures: Annotated[
        bool, typer.Option("--figures/--no-figures", help="Also plot the mean image of each class.")
    ] = True,
) -> None:
    """Segment the recordings of the dataset index and compute spectrogram images."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"data.dataset": dataset})
        features = prepare_features(config.data.dataset, config)
        folder = _run_folder(config, "prepare", out)
        save_features(folder / "features.npz", features)
        _write_csv(items_table(features), folder / "items.csv")
        if figures:
            figure = plot_class_spectrograms(features, dpi=config.output.dpi)
            save_figure(figure, folder / "class_spectrograms", config.output.figure_format)
            _write_csv(class_spectrograms_table(features), folder / "class_spectrograms.csv")
        n_recordings = len(set(features.paths.tolist()))
        _report("recordings", n_recordings)
        _report("segments", features.n_items)
        counts = Counter(features.y.tolist())
        _report("classes", ", ".join(f"{name} {counts[name]}" for name in features.classes))
        rows, columns = features.image_shape
        _report("image", f"{rows} x {columns} ({config.spectrogram.scale})")
        _finish(
            folder,
            "prepare",
            config,
            [Path(config.data.dataset)],
            {"n_recordings": n_recordings, "n_items": features.n_items},
        )


@app.command("evaluate")
def evaluate_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Cross-validate the configured model; write metrics, predictions and figures."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        result = cross_validate(data, config)
        folder = _run_folder(config, "evaluate", out)
        write_json(folder / "metrics.json", result.metrics)
        _write_csv(result.folds, folder / "folds.csv")
        _write_csv(result.predictions, folder / "predictions.csv")
        _write_csv(result.recording_predictions, folder / "recording_predictions.csv")
        fmt, dpi = config.output.figure_format, config.output.dpi
        for level, stem in (
            ("segments", "confusion_matrix"),
            ("recordings", "recording_confusion_matrix"),
        ):
            save_figure(plot_confusion_matrix(result, level, dpi=dpi), folder / stem, fmt)
            _write_csv(confusion_matrix_table(result, level), folder / f"{stem}.csv")
        recordings = result.metrics["recordings"]
        _report("model", config.model.name)
        _report("segments", result.metrics["n_items"])
        _report("recordings", recordings["n_items"])
        _report("folds", config.evaluation.n_splits)
        _report("UAR", _mean_std(result.metrics["summary"], "uar"))
        _report("accuracy", _mean_std(result.metrics["summary"], "accuracy"))
        _report("UAR (rec.)", _mean_std(recordings["summary"], "uar"))
        _finish(
            folder,
            "evaluate",
            config,
            [features],
            {"n_items": result.metrics["n_items"], "n_recordings": recordings["n_items"]},
        )


@app.command("train")
def train_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Train the configured model on every segment and save it."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        estimator = train_model(data, config)
        folder = _run_folder(config, "train", out)
        destination = save_model(folder / model_filename(config), estimator, config, data.classes)
        _report("model", config.model.name)
        _report("segments", data.n_items)
        _report("classes", ", ".join(data.classes))
        _report("file", destination.name)
        _finish(folder, "train", config, [features], {"n_items": data.n_items})


@app.command("predict")
def predict_command(
    inputs: Annotated[list[Path], typer.Argument(help="WAV files to classify.")],
    model: ModelOption,
    out: OutOption = None,
) -> None:
    """Classify new recordings with a saved model, segment by segment and as a whole."""
    with _reported_errors():
        bundle = load_model(model)
        config = Config.from_dict(bundle["config"])
        segments = predict_inputs(inputs, bundle)
        classes = [str(name) for name in bundle["classes"]]
        recordings = aggregate_recordings(segments, classes)
        folder = _run_folder(config, "predict", out)
        _write_csv(segments, folder / "predictions.csv")
        _write_csv(recordings, folder / "recording_predictions.csv")
        _report("recordings", len(recordings))
        _report("segments", len(segments))
        counts = Counter(recordings["predicted"].tolist())
        _report("predicted", ", ".join(f"{name} {counts[name]}" for name in sorted(counts)))
        _finish(folder, "predict", config, [model, *inputs], {"n_items": len(segments)})


def main() -> None:
    """Entry point of the ``voice-spectral-classifier`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
