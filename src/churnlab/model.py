"""Model training, evaluation, and MLflow tracking.

Compares a simple baseline (Logistic Regression) against two stronger
models (Random Forest and XGBoost) and reports the metrics that matter for a
churn problem: precision, recall, F1, ROC-AUC, and the confusion matrix.

Churn is class-imbalanced (~26% positive), so accuracy alone is misleading —
we report the per-class metrics and pick the model by ROC-AUC.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]
    train_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "confusion": self.confusion,
            "train_time_s": round(self.train_time_s, 3),
        }


@dataclass
class TrainingReport:
    results: list[ModelResult] = field(default_factory=list)
    best_model: str = ""
    feature_names: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    class_balance: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "best_model": self.best_model,
            "feature_names": self.feature_names,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "class_balance": self.class_balance,
        }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str) -> Pipeline:
    """Return a scikit-learn Pipeline (scaler + estimator) for ``name``.

    A pipeline is used so that the exact same preprocessing (StandardScaler)
    is applied at training and at serving time — no feature drift.
    """
    if name == "logistic_regression":
        est = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    elif name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
    elif name == "xgboost":
        est = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    pipe.name = name
    return pipe


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    """Fit ``model`` and score it on the held-out test set."""
    import time

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    return ModelResult(
        name=model.name,
        accuracy=float(np.mean(y_pred == y_test)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        confusion=cm.tolist(),
        train_time_s=train_time,
    )


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    run_id: str | None = None,
    track: bool = True,
) -> tuple[TrainingReport, dict[str, Pipeline]]:
    """Train every model in ``config.MODELS`` and return report + fitted models.

    ``run_id`` is used as the MLflow run name (defaults to a timestamp).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    report = TrainingReport(
        feature_names=list(X.columns),
        n_train=len(X_train),
        n_test=len(X_test),
        class_balance={
            "train": {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())},
            "test": {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())},
        },
    )

    models: dict[str, Pipeline] = {}
    run_name = run_id or "churnlab-training"

    if track:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_DIR)
        with mlflow.start_run(run_name=run_name):
            for name in config.MODELS:
                model = build_model(name)
                result = evaluate_model(model, X_train, y_train, X_test, y_test)
                report.results.append(result)
                models[name] = model
                mlflow.log_params({"model": name})
                mlflow.log_metrics(
                    {
                        f"{name}_accuracy": result.accuracy,
                        f"{name}_precision": result.precision,
                        f"{name}_recall": result.recall,
                        f"{name}_f1": result.f1,
                        f"{name}_roc_auc": result.roc_auc,
                    }
                )
                mlflow.sklearn.log_model(model, name=name)
    else:
        for name in config.MODELS:
            model = build_model(name)
            result = evaluate_model(model, X_train, y_train, X_test, y_test)
            report.results.append(result)
            models[name] = model

    # Pick the best model by ROC-AUC (robust to class imbalance).
    best = max(report.results, key=lambda r: r.roc_auc)
    report.best_model = best.name
    return report, models


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(
    models: dict[str, Pipeline],
    encoders: dict,
    report: TrainingReport,
    out_dir: str | Path = config.ARTIFACTS_DIR,
) -> Path:
    """Persist the best model, all models, encoders, and the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = models[report.best_model]
    joblib.dump(best, out_dir / f"best_model_{report.best_model}.joblib")
    for name, model in models.items():
        joblib.dump(model, out_dir / f"model_{name}.joblib")
    joblib.dump(encoders, out_dir / "encoders.joblib")
    (out_dir / "report.json").write_text(json.dumps(report.as_dict(), indent=2))
    return out_dir


def load_best_model(
    out_dir: str | Path = config.ARTIFACTS_DIR,
) -> tuple[Pipeline, dict, TrainingReport]:
    """Load the best model, encoders, and report from disk."""
    out_dir = Path(out_dir)
    report = TrainingReport(**json.loads((out_dir / "report.json").read_text()))
    best = joblib.load(out_dir / f"best_model_{report.best_model}.joblib")
    encoders = joblib.load(out_dir / "encoders.joblib")
    return best, encoders, report
