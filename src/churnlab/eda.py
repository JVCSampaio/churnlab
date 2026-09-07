"""Exploratory data analysis helpers.

These functions return small, summarised frames/dicts (not the full dataset)
so they can be printed in a notebook or shown in the README. They are the
"EDA" layer: churn rate, class balance, feature distributions, and the
churn rate by the most telling categorical features.
"""
from __future__ import annotations

import pandas as pd

from . import config


def churn_rate(df: pd.DataFrame) -> float:
    """Overall churn rate (fraction of customers who churned)."""
    return float(df[config.TARGET].mean())


def class_balance(df: pd.DataFrame) -> dict[str, int]:
    """Count of churned (1) vs retained (0) customers."""
    vc = df[config.TARGET].value_counts()
    return {"churned": int(vc.get(1, 0)), "retained": int(vc.get(0, 0))}


def churn_by_feature(df: pd.DataFrame, col: str) -> pd.Series:
    """Churn rate grouped by a categorical column, sorted descending."""
    g = df.groupby(col)[config.TARGET].mean().sort_values(ascending=False)
    return g


def churn_by_tenure_bucket(df: pd.DataFrame) -> pd.Series:
    """Churn rate by tenure bucket (months)."""
    buckets = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 120])
    return df.groupby(buckets)[config.TARGET].mean()


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact numeric summary of the numeric features."""
    cols = list(config.NUMERIC_COLS)
    if "tenure_years" in df.columns:
        cols.append("tenure_years")
    return df[cols].describe().round(2)


def top_churn_drivers(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Churn rate by the features that most strongly predict churn."""
    interesting = ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]
    return {col: churn_by_feature(df, col) for col in interesting if col in df.columns}
