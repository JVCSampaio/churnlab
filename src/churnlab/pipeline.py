"""End-to-end pipeline runner.

Usage:
    python -m churnlab.pipeline            # train, write metrics, figures, README
    python -m churnlab.pipeline --no-track # skip the MLflow run
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .data import clean, encode, engineer, load_raw
from .model import save_artifacts, train_all
from .plots import write_figures
from .readme import format_cli, render_readme
from .reporting import assemble_metrics


def run(track: bool = True) -> dict:
    """Train, score, and rewrite the metrics file, figures, and README."""
    raw = load_raw()
    cleaned = engineer(clean(raw))
    # encode() rewrites categorical columns in place; keep labels for the EDA tables.
    features, target, encoders = encode(cleaned.copy())
    report, models = train_all(features, target, track=track)
    metrics = assemble_metrics(raw, cleaned, encoders, report, models)
    text = save_artifacts(models, encoders, report, metrics=metrics)
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(text, encoding="utf-8")
    # Render from the parsed file so the README matches what tests reload.
    metrics = json.loads(text)
    write_figures(metrics)
    config.README_PATH.write_text(render_readme(metrics), encoding="utf-8")
    print(format_cli(metrics))
    print()
    print(f"metrics:  {config.METRICS_PATH}")
    print(f"figures:  {config.FIGURES_DIR}")
    print(f"readme:   {config.README_PATH}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ChurnLab pipeline")
    parser.add_argument("--no-track", action="store_true", help="skip MLflow tracking")
    args = parser.parse_args(argv)
    run(track=not args.no_track)
    return 0


if __name__ == "__main__":
    sys.exit(main())
