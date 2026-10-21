"""Model training, evaluation, and MLflow tracking.

Compares a logistic-regression baseline with Random Forest and XGBoost.
Churn is class-imbalanced, so the shipped model is the one with the highest
mean ROC-AUC across stratified folds on the training set. Precision and recall
at 0.5 are reported for comparison; the decision threshold is chosen from an
explicit false-negative / false-positive cost ratio on out-of-fold scores.
"""
from __future__ import annotations

import json
import time
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .serialize import to_plain


def cost_assumption(fn_cost: float, fp_cost: float) -> str:
    """State the cost ratio as a planning assumption, not a fitted quantity."""
    fn = f"{fn_cost:g}"
    fp = f"{fp_cost:g}"
    return (
        "A false negative — failing to offer retention to a customer who churns — "
        f"is treated as {fn} times as costly as a false positive — contacting a "
        f"customer who would have stayed. The ratio {fn}:{fp} is a planning "
        "assumption, not a quantity estimated from this dataset. A retention team "
        "should replace it with offer cost and remaining customer value."
    )


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
    cv: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion": self.confusion,
            "train_time_s": self.train_time_s,
            "cv": self.cv,
        }


@dataclass
class TrainingReport:
    results: list[ModelResult] = field(default_factory=list)
    best_model: str = ""
    feature_names: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    class_balance: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    holdout_best_by_roc_auc: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [result.as_dict() for result in self.results],
            "best_model": self.best_model,
            "holdout_best_by_roc_auc": self.holdout_best_by_roc_auc,
            "feature_names": self.feature_names,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "class_balance": self.class_balance,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingReport:
        """Rebuild a report from either ``as_dict()`` or ``docs/metrics.json``."""
        if "models" in payload:
            raw_results = payload["models"]
            split = payload.get("split", {})
            n_train = int(split.get("n_train", 0))
            n_test = int(split.get("n_test", 0))
        else:
            raw_results = payload.get("results", [])
            n_train = int(payload.get("n_train", 0))
            n_test = int(payload.get("n_test", 0))
        results = [
            ModelResult(
                name=raw["name"],
                accuracy=float(raw["accuracy"]),
                precision=float(raw["precision"]),
                recall=float(raw["recall"]),
                f1=float(raw["f1"]),
                roc_auc=float(raw["roc_auc"]),
                confusion=raw["confusion"],
                train_time_s=float(raw["train_time_s"]),
                cv=dict(raw.get("cv", {})),
            )
            for raw in raw_results
        ]
        return cls(
            results=results,
            best_model=payload.get("best_model", ""),
            feature_names=list(payload.get("feature_names", [])),
            n_train=n_train,
            n_test=n_test,
            class_balance=payload.get("class_balance", {}),
            threshold=dict(payload.get("threshold", {})),
            holdout_best_by_roc_auc=payload.get("holdout_best_by_roc_auc", ""),
        )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str) -> Pipeline:
    """Return a scikit-learn Pipeline (scaler + estimator) for ``name``.

    ``n_jobs=1`` keeps tree fits stable enough that a regenerated metrics file
    does not flutter between machines. The scaler is shared so serving applies
    the same transform the logistic regression was trained with.
    """
    if name == "logistic_regression":
        est = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    elif name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
        )
    elif name == "xgboost":
        est = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    pipe.name = name
    return pipe


def make_cv(n_splits: int = config.N_CV_FOLDS) -> StratifiedKFold:
    """Stratified K-fold splitter shared by cross-validation and its tests."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)


def holdout_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """The single stratified holdout used for the published test metrics."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )


def _scores_at(y_true, proba, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": float(np.mean(pred == y)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "confusion": matrix.astype(int).tolist(),
    }


def cross_validate_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = config.N_CV_FOLDS,
) -> tuple[dict[str, Any], np.ndarray]:
    """Stratified CV. Returns mean/std at the comparison cutoff, plus OOF scores."""
    oof = np.zeros(len(y), dtype=float)
    rows: list[dict[str, float]] = []
    for train_idx, valid_idx in make_cv(n_splits).split(X, y):
        model = build_model(name)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = proba
        scored = _scores_at(y.iloc[valid_idx], proba, config.COMPARISON_THRESHOLD)
        rows.append({key: scored[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc")})
    summary: dict[str, Any] = {"n_splits": n_splits}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        vals = np.array([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std(ddof=1))
    return summary, oof


def tune_threshold(
    y_true,
    proba,
    fn_cost: float = config.FN_COST,
    fp_cost: float = config.FP_COST,
    grid: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the threshold that minimises average misclassification cost.

    Cost per customer is ``(fn * fn_cost + fp * fp_cost) / n``. When two
    cutoffs share that cost, the one closer to 0.5 wins.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)
    if grid is None:
        grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    curve: list[dict[str, Any]] = []
    n = len(y)
    for threshold in grid:
        pred = scores >= float(threshold)
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tp = int(np.sum(pred & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append(
            {
                "threshold": round(float(threshold), 2),
                "cost": float((fn * fn_cost + fp * fp_cost) / n),
                "precision": float(precision),
                "recall": float(recall),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": int(n - tp - fp - fn),
            }
        )
    chosen = min(curve, key=lambda row: (row["cost"], abs(row["threshold"] - 0.5)))
    summary = {
        "value": chosen["threshold"],
        "fn_cost": float(fn_cost),
        "fp_cost": float(fp_cost),
        "ratio_label": f"{fn_cost:g}:{fp_cost:g}",
        "assumption": cost_assumption(fn_cost, fp_cost),
        "tuned_on": "out-of-fold predictions on the training set",
        "tie_break": "minimum average cost; ties resolve to the threshold closest to 0.5",
        "oof_cost": chosen["cost"],
        "oof_precision": chosen["precision"],
        "oof_recall": chosen["recall"],
    }
    return summary, curve


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    """Fit ``model`` and score the holdout at the comparison threshold."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    proba = model.predict_proba(X_test)[:, 1]
    scored = _scores_at(y_test, proba, config.COMPARISON_THRESHOLD)
    return ModelResult(
        name=model.name,
        accuracy=scored["accuracy"],
        precision=scored["precision"],
        recall=scored["recall"],
        f1=scored["f1"],
        roc_auc=scored["roc_auc"],
        confusion=scored["confusion"],
        train_time_s=train_time,
    )


def _operating_point(y_true, proba, threshold: float, fn_cost: float, fp_cost: float) -> dict[str, Any]:
    scored = _scores_at(y_true, proba, threshold)
    tn, fp = scored["confusion"][0]
    fn, tp = scored["confusion"][1]
    n = tn + fp + fn + tp
    return {
        "test_precision": scored["precision"],
        "test_recall": scored["recall"],
        "test_f1": scored["f1"],
        "test_confusion": scored["confusion"],
        "test_cost": float((fn * fn_cost + fp * fp_cost) / n),
    }


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    run_id: str | None = None,
    track: bool = True,
) -> tuple[TrainingReport, dict[str, Pipeline]]:
    """Cross-validate, fit, and tune a threshold for every model in ``config.MODELS``."""
    X_train, X_test, y_train, y_test = holdout_split(X, y)
    report = TrainingReport(
        feature_names=list(X.columns),
        n_train=len(X_train),
        n_test=len(X_test),
        class_balance={
            "train": {
                "retained": int((y_train == 0).sum()),
                "churned": int((y_train == 1).sum()),
            },
            "test": {
                "retained": int((y_test == 0).sum()),
                "churned": int((y_test == 1).sum()),
            },
        },
    )

    cv_by_name: dict[str, dict[str, Any]] = {}
    oof_by_name: dict[str, np.ndarray] = {}
    for name in config.MODELS:
        print(f"cross-validating {name}...", flush=True)
        summary, oof = cross_validate_model(name, X_train, y_train)
        cv_by_name[name] = summary
        oof_by_name[name] = oof

    models: dict[str, Pipeline] = {}
    for name in config.MODELS:
        print(f"fitting {name} on the training split...", flush=True)
        model = build_model(name)
        result = evaluate_model(model, X_train, y_train, X_test, y_test)
        result.cv = cv_by_name[name]
        report.results.append(result)
        models[name] = model

    # First maximum wins ties, which is the earlier name in config.MODELS.
    report.best_model = max(config.MODELS, key=lambda name: cv_by_name[name]["roc_auc_mean"])
    report.holdout_best_by_roc_auc = max(report.results, key=lambda result: result.roc_auc).name

    chosen, curve = tune_threshold(y_train.to_numpy(), oof_by_name[report.best_model])
    test_proba = models[report.best_model].predict_proba(X_test)[:, 1]
    operating = _operating_point(
        y_test, test_proba, chosen["value"], config.FN_COST, config.FP_COST
    )
    reference_cost = min(curve, key=lambda row: abs(row["threshold"] - config.COMPARISON_THRESHOLD))
    report.threshold = {
        **chosen,
        **operating,
        "curve": curve,
        "comparison_threshold": config.COMPARISON_THRESHOLD,
        "oof_cost_at_comparison_threshold": reference_cost["cost"],
    }

    if track:
        _log_mlflow(report, models, run_id)
    return report, models


def _log_mlflow(report: TrainingReport, models: dict[str, Pipeline], run_id: str | None) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_DIR)
    with mlflow.start_run(run_name=run_id or "churnlab-training"):
        for result in report.results:
            name = result.name
            mlflow.log_metrics(
                {
                    f"{name}_accuracy": result.accuracy,
                    f"{name}_precision": result.precision,
                    f"{name}_recall": result.recall,
                    f"{name}_f1": result.f1,
                    f"{name}_roc_auc": result.roc_auc,
                    f"{name}_cv_roc_auc_mean": result.cv["roc_auc_mean"],
                    f"{name}_cv_roc_auc_std": result.cv["roc_auc_std"],
                }
            )
            mlflow.sklearn.log_model(
                models[name],
                name=name,
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )
        mlflow.log_params(
            {
                "best_model": report.best_model,
                "n_train": report.n_train,
                "n_test": report.n_test,
                "cv_folds": config.N_CV_FOLDS,
                "fn_cost": config.FN_COST,
                "fp_cost": config.FP_COST,
                "threshold": report.threshold["value"],
            }
        )


def model_hyperparameters(model: Pipeline) -> dict[str, Any]:
    """The handful of settings that define this comparison (no search was run)."""
    keys = {
        "logistic_regression": ["C", "max_iter"],
        "random_forest": ["n_estimators", "max_depth", "min_samples_leaf"],
        "xgboost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
    }[model.name]
    params = model.named_steps["model"].get_params()
    return {key: params[key] for key in keys}


def feature_effects(model: Pipeline, names: list[str]) -> dict[str, Any]:
    """Coefficients for the linear model, impurity importance for the trees."""
    est = model.named_steps["model"]
    if hasattr(est, "coef_"):
        kind = "coefficient"
        values = np.asarray(est.coef_).ravel()
    elif hasattr(est, "feature_importances_"):
        kind = "importance"
        values = np.asarray(est.feature_importances_).ravel()
    else:
        raise TypeError(f"{model.name} exposes neither coefficients nor importances")
    pairs = [{"name": name, "value": float(value)} for name, value in zip(names, values)]
    pairs.sort(key=lambda row: abs(row["value"]), reverse=True)
    return {"kind": kind, "model": model.name, "features": pairs}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(
    models: dict[str, Pipeline],
    encoders: dict,
    report: TrainingReport,
    out_dir: str | Path = config.ARTIFACTS_DIR,
    metrics: dict | None = None,
) -> str:
    """Persist models, encoders, and the metrics document. Returns the JSON text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[report.best_model], out_dir / f"best_model_{report.best_model}.joblib")
    for name, model in models.items():
        joblib.dump(model, out_dir / f"model_{name}.joblib")
    joblib.dump(encoders, out_dir / "encoders.joblib")
    payload = to_plain(metrics if metrics is not None else report.as_dict())
    text = json.dumps(payload, indent=2) + "\n"
    (out_dir / "report.json").write_text(text, encoding="utf-8")
    return text


def load_best_model(
    out_dir: str | Path = config.ARTIFACTS_DIR,
) -> tuple[Pipeline, dict, TrainingReport]:
    """Load the best model, encoders, and report from disk."""
    out_dir = Path(out_dir)
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report = TrainingReport.from_dict(payload)
    best = joblib.load(out_dir / f"best_model_{report.best_model}.joblib")
    encoders = joblib.load(out_dir / "encoders.joblib")
    return best, encoders, report
