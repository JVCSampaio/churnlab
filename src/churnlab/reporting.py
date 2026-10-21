"""Assemble the metrics document the README is rendered from."""
from __future__ import annotations

import pandas as pd

from . import config
from .api import Customer, score_customer
from .data import ENGINEERED_FEATURES
from .eda import CONTRACT_ORDER, grouped_rates, tenure_churn_table
from .model import TrainingReport, feature_effects, model_hyperparameters


def assemble_metrics(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    encoders: dict,
    report: TrainingReport,
    models: dict,
) -> dict:
    """Build the single document behind ``docs/metrics.json`` and the README.

    ``cleaned`` must still carry the original category labels (call this before
    ``encode`` mutates them, or pass a copy).
    """
    if len(raw) != len(cleaned):
        raise ValueError("Cleaning dropped rows; dataset counts would be ambiguous")
    blank = int(pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum())
    target = cleaned[config.TARGET]
    records = []
    for result in report.results:
        record = result.as_dict()
        record["hyperparameters"] = model_hyperparameters(models[result.name])
        records.append(record)

    customer = Customer.model_validate(dict(config.EXAMPLE_CUSTOMER))
    prediction = score_customer(
        customer,
        models[report.best_model],
        encoders,
        report.threshold["value"],
        report.best_model,
    )
    return {
        "project": {
            "author": config.AUTHOR,
            "python": config.PYTHON_VERSION,
            "port": config.API_PORT,
            "repository": config.REPOSITORY,
        },
        "dataset": {
            "name": "IBM Telco Customer Churn",
            "n_rows": int(len(cleaned)),
            "n_churned": int((target == 1).sum()),
            "n_retained": int((target == 0).sum()),
            "churn_rate": float(target.mean()),
            "n_blank_total_charges": blank,
            "n_model_features": len(report.feature_names),
            "n_engineered_features": len(ENGINEERED_FEATURES),
        },
        "split": {
            "test_size": config.TEST_SIZE,
            "random_seed": config.RANDOM_SEED,
            "n_train": report.n_train,
            "n_test": report.n_test,
            "test_retained": report.class_balance["test"]["retained"],
            "test_churned": report.class_balance["test"]["churned"],
            "comparison_threshold": config.COMPARISON_THRESHOLD,
        },
        "cv": {
            "n_splits": config.N_CV_FOLDS,
            "stratified": True,
            "scope": "training set",
        },
        "models": records,
        "best_model": report.best_model,
        "selection_criterion": (
            "highest mean ROC-AUC across stratified folds on the training set"
        ),
        "holdout_best_by_roc_auc": report.holdout_best_by_roc_auc,
        "class_balance": report.class_balance,
        "feature_names": list(report.feature_names),
        "threshold": report.threshold,
        "effects": feature_effects(models[report.best_model], report.feature_names),
        "eda": {
            "by_contract": grouped_rates(cleaned, "Contract", CONTRACT_ORDER),
            "by_tenure": tenure_churn_table(cleaned),
        },
        "example": {
            "request": dict(config.EXAMPLE_CUSTOMER),
            "response": {
                "customer_id": prediction.customer_id,
                "churn_probability": prediction.churn_probability,
                "will_churn": prediction.will_churn,
                "threshold": prediction.threshold,
                "model": prediction.model,
            },
        },
    }
