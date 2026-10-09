"""Compare linear and mel spectrogram images across classifiers.

    python examples/compare_spectrogram_scales.py --out examples/output/compare_spectrogram_scales

1. Write one synthetic dataset.
2. Prepare it twice: 40 equal-width bands (``linear``) and 40 mel bands (``mel``).
3. Cross-validate the logistic-regression and MLP baselines, and the CNN when
   PyTorch is installed, on both image sets with the same folds.
4. Write ``scale_comparison.csv`` (UAR per segment and per recording, mean and
   sample standard deviation over the folds) and a bar chart of it.

On synthetic data the scores only show that the code works; they say nothing
about real recordings.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from voice_spectral_classifier import (
    SERIES_COLORS,
    SPECTROGRAM_SCALES,
    Config,
    configure_logging,
    cross_validate,
    new_figure,
    prepare_features,
    save_figure,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_spectrogram_scales")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_spectrogram_scales"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick", action="store_true", help="Tiny data and models; finishes in seconds."
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    base = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    base = base.override({"seed": args.seed})
    models = ["logistic", "mlp"]
    if importlib.util.find_spec("torch") is not None:
        models.append("cnn")
    else:
        LOGGER.info("PyTorch is not installed; comparing the scikit-learn models only")

    index = write_synthetic_dataset(args.out / "data", base.synthetic, rng=base.seed)
    rows = []
    for scale in SPECTROGRAM_SCALES:
        prepared = base.override({"spectrogram.scale": scale})
        features = prepare_features(index, prepared)
        for name in models:
            config = prepared.override({"model.name": name})
            result = cross_validate(features, config)
            segments = result.metrics["summary"]["uar"]
            recordings = result.metrics["recordings"]["summary"]["uar"]
            rows.append(
                {
                    "scale": scale,
                    "model": name,
                    "uar_mean": segments["mean"],
                    "uar_std": segments["std"],
                    "recording_uar_mean": recordings["mean"],
                    "recording_uar_std": recordings["std"],
                }
            )
            LOGGER.info(
                "%-6s %-8s UAR %.3f +/- %.3f (recordings %.3f +/- %.3f)",
                scale,
                name,
                segments["mean"],
                segments["std"],
                recordings["mean"],
                recordings["std"],
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "scale_comparison.csv", index=False, lineterminator="\n")

    figure = new_figure(5.5, 3.2, 150)
    axes = figure.add_subplot()
    positions = np.arange(len(models))
    for offset, (scale, color) in enumerate(
        zip(SPECTROGRAM_SCALES, SERIES_COLORS[:2], strict=True)
    ):
        part = table[table["scale"] == scale].set_index("model").loc[models]
        axes.bar(
            positions + (offset - 0.5) * 0.38,
            part["uar_mean"],
            width=0.38,
            yerr=part["uar_std"].fillna(0.0),
            color=color,
            label=scale,
            capsize=3,
        )
    axes.set_xticks(positions, models)
    axes.set_ylim(0.0, 1.05)
    axes.set_ylabel("UAR per segment (mean +/- std over folds)")
    axes.legend(title="spectrogram scale", loc="lower right")
    save_figure(figure, args.out / "scale_comparison", "png")
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
