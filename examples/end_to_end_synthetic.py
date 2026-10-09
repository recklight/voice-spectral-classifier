"""End to end on synthetic data: synthesize, prepare, cross-validate, train, predict.

    python examples/end_to_end_synthetic.py --out examples/output/end_to_end_synthetic

1. Write a synthetic dataset with the same on-disk layout as real data.
2. Cut every recording into one-second segments and compute their 40 x 98
   spectrogram images.
3. Cross-validate the model with grouped, stratified folds and report the
   scores per segment and per recording.
4. Train on every segment and save the model.
5. Load the model and classify a few recordings, segment by segment and as a whole.

The default model is the MLP baseline, which needs nothing beyond
scikit-learn; ``--model cnn`` trains the CNN when PyTorch is installed.
On synthetic data the scores only show that the code works; they say nothing
about real recordings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from voice_spectral_classifier import (
    MODEL_NAMES,
    Config,
    aggregate_recordings,
    configure_logging,
    cross_validate,
    items_table,
    load_model,
    model_filename,
    predict_inputs,
    prepare_features,
    save_features,
    save_model,
    train_model,
    write_json,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("end_to_end_synthetic")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/end_to_end_synthetic"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick", action="store_true", help="Tiny data and models; finishes in seconds."
    )
    parser.add_argument(
        "--model", choices=MODEL_NAMES, default=None, help="Classifier (default: mlp)."
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed, "model.name": args.model or "mlp"})
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic recordings and their dataset.csv index.
    index = write_synthetic_dataset(args.out / "data", config.synthetic, rng=config.seed)

    # 2. Segment images.
    features = prepare_features(index, config)
    save_features(args.out / "features.npz", features)
    _write_csv(items_table(features), args.out / "items.csv")
    LOGGER.info(
        "%d segments of %d recordings, images of %d x %d",
        features.n_items,
        len(set(features.paths.tolist())),
        *features.image_shape,
    )

    # 3. Grouped, stratified cross-validation.
    result = cross_validate(features, config)
    write_json(args.out / "metrics.json", result.metrics)
    _write_csv(result.folds, args.out / "folds.csv")
    _write_csv(result.predictions, args.out / "predictions.csv")
    _write_csv(result.recording_predictions, args.out / "recording_predictions.csv")
    for level, metrics in (
        ("segments", result.metrics),
        ("recordings", result.metrics["recordings"]),
    ):
        summary = metrics["summary"]["uar"]
        LOGGER.info("UAR over %s: %.3f +/- %.3f", level, summary["mean"], summary["std"])

    # 4. One model on every segment.
    model_path = save_model(
        args.out / model_filename(config), train_model(features, config), config, features.classes
    )

    # 5. Reload it and classify the first recording of every class.
    bundle = load_model(model_path)
    data = pd.read_csv(index)
    chosen = data.groupby("label", sort=True)["path"].first()
    inputs = [index.parent / relative for relative in chosen]
    segments = predict_inputs(inputs, bundle)
    recordings = aggregate_recordings(segments, list(bundle["classes"]))
    _write_csv(segments, args.out / "new_predictions.csv")
    _write_csv(recordings, args.out / "new_recording_predictions.csv")
    for (label, _), predicted in zip(chosen.items(), recordings["predicted"], strict=True):
        LOGGER.info("a %s recording was classified as %s", label, predicted)
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
