"""Figures written for the README."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from churnlab import config
from churnlab.plots import write_figures


def _fake_metrics() -> dict:
    curve = [
        {"threshold": threshold, "cost": (threshold - 0.2) ** 2 + 0.1, "precision": threshold, "recall": 1 - threshold}
        for threshold in (0.1, 0.2, 0.5, 0.8)
    ]
    return {
        "best_model": "logistic_regression",
        "models": [
            {"name": "logistic_regression", "roc_auc": 0.81},
            {"name": "random_forest", "roc_auc": 0.77},
            {"name": "xgboost", "roc_auc": 0.74},
        ],
        "split": {"comparison_threshold": 0.5},
        "threshold": {
            "value": 0.2,
            "fn_cost": 5.0,
            "fp_cost": 1.0,
            "test_confusion": [[8, 2], [1, 4]],
            "curve": curve,
        },
        "eda": {
            "by_contract": [
                {"level": "Month-to-month", "n": 30, "churn_rate": 0.4},
                {"level": "One year", "n": 20, "churn_rate": 0.15},
                {"level": "Two year", "n": 10, "churn_rate": 0.05},
            ],
            "by_tenure": [
                {"level": "0–6", "n": 12, "churn_rate": 0.5},
                {"level": "7–12", "n": 8, "churn_rate": 0.25},
            ],
        },
        "effects": {
            "kind": "coefficient",
            "model": "logistic_regression",
            "features": [
                {"name": "tenure", "value": -0.8},
                {"name": "Contract", "value": 0.4},
                {"name": "MonthlyCharges", "value": 0.2},
            ],
        },
    }


def test_write_figures_emits_pngs(tmp_path):
    grid = np.linspace(0, 1, 11)
    curves = {
        name: (grid, np.clip(grid ** (1 + index * 0.35), 0, 1))
        for index, name in enumerate(config.MODELS)
    }
    paths = write_figures(_fake_metrics(), out_dir=tmp_path, roc_curves=curves)
    names = {path.name for path in paths}
    assert names == {Path(rel).name for rel in config.FIGURES.values()}
    for path in paths:
        blob = path.read_bytes()
        assert blob.startswith(b"\x89PNG")
        assert len(blob) > 2_000


def test_committed_figures_exist():
    for rel in config.FIGURES.values():
        path = config.PROJECT_ROOT / rel
        assert path.is_file(), rel
        assert path.stat().st_size > 5_000
