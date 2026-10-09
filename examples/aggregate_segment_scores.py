"""Segment-level versus recording-level scores, with one or with all segments.

    python examples/aggregate_segment_scores.py --out examples/output/aggregate_segment_scores

1. Write one synthetic dataset of multi-second recordings.
2. Prepare it twice: once with only the middle one-second segment of each
   recording, once with every segment.
3. Cross-validate the same model on both, with folds grouped by speaker.
4. Compare the scores of single segments with the scores obtained by
   averaging the class probabilities of all segments of a recording, and
   write ``aggregation_comparison.csv`` and ``per_class_recall.csv``.

On synthetic data the scores only show that the code works; they say nothing
about real recordings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from voice_spectral_classifier import (
    Config,
    configure_logging,
    cross_validate,
    prepare_features,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("aggregate_segment_scores")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"
#: Segments kept per recording: one clip, or all of them.
SEGMENT_LIMITS: tuple[int | None, ...] = (1, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/aggregate_segment_scores"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick", action="store_true", help="Tiny data and models; finishes in seconds."
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    base = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    base = base.override({"seed": args.seed, "model.name": "mlp"})
    index = write_synthetic_dataset(args.out / "data", base.synthetic, rng=base.seed)

    summary_rows = []
    recall_rows = []
    for limit in SEGMENT_LIMITS:
        config = base.override({"segmentation.max_segments": limit})
        features = prepare_features(index, config)
        result = cross_validate(features, config)
        kept = "all" if limit is None else str(limit)
        for level, metrics in (
            ("segment", result.metrics),
            ("recording", result.metrics["recordings"]),
        ):
            summary = metrics["summary"]
            summary_rows.append(
                {
                    "segments_kept": kept,
                    "level": level,
                    "n_items": metrics["n_items"],
                    "uar_mean": summary["uar"]["mean"],
                    "uar_std": summary["uar"]["std"],
                    "accuracy_mean": summary["accuracy"]["mean"],
                    "accuracy_std": summary["accuracy"]["std"],
                    "pooled_uar": metrics["pooled"]["uar"],
                }
            )
            for name, scores in metrics["pooled"]["per_class"].items():
                recall_rows.append(
                    {
                        "segments_kept": kept,
                        "level": level,
                        "class": name,
                        "recall": scores["recall"],
                        "specificity": scores["specificity"],
                        "support": scores["support"],
                    }
                )
            LOGGER.info(
                "%s segment(s) kept, scored per %-9s: %3d items, UAR %.3f +/- %.3f",
                kept,
                level,
                metrics["n_items"],
                summary["uar"]["mean"],
                summary["uar"]["std"],
            )
    pd.DataFrame(summary_rows).to_csv(
        args.out / "aggregation_comparison.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(recall_rows).to_csv(
        args.out / "per_class_recall.csv", index=False, lineterminator="\n"
    )
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
