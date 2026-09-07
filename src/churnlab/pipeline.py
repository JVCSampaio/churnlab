"""End-to-end pipeline runner.

Usage:
    python -m churnlab.pipeline            # full run (train + save artifacts)
    python -m churnlab.pipeline --track    # also record an MLflow run

This is the "run the pipeline" command: load -> clean -> engineer -> encode
-> train all models -> evaluate -> save artifacts. It prints a compact report.
"""
from __future__ import annotations

import argparse
import sys

from . import config
from .data import clean, encode, engineer, load_raw
from .model import save_artifacts, train_all


def run(track: bool = True) -> dict:
    """Execute the full pipeline and return the training report dict."""
    raw = load_raw()
    df = clean(raw)
    df = engineer(df)
    X, y, encoders = encode(df)

    report, models = train_all(X, y, track=track)
    save_artifacts(models, encoders, report)

    print("=== ChurnLab training report ===")
    print(f"train rows: {report.n_train}   test rows: {report.n_test}")
    print(f"class balance (test): {report.class_balance['test']}")
    print(f"best model (by ROC-AUC): {report.best_model}")
    print()
    header = f"{'model':<20} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6} {'time(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in report.results:
        print(
            f"{r.name:<20} {r.accuracy:>6.3f} {r.precision:>6.3f} "
            f"{r.recall:>6.3f} {r.f1:>6.3f} {r.roc_auc:>6.3f} {r.train_time_s:>8.2f}"
        )
    print()
    print(f"artifacts saved to: {config.ARTIFACTS_DIR}")
    return report.as_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ChurnLab pipeline")
    parser.add_argument("--no-track", action="store_true", help="skip MLflow tracking")
    args = parser.parse_args(argv)
    run(track=not args.no_track)
    return 0


if __name__ == "__main__":
    sys.exit(main())
