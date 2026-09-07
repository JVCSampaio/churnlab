"""Data loading, cleaning, feature engineering, and encoding.

The raw IBM Telco Customer Churn CSV has a few real-world quirks we clean:
  * ``TotalCharges`` is stored as a string (blank for some short-tenure rows)
    -> coerced to float, blanks become 0.
  * ``MultipleLines`` contains ``"No phone service"`` (only when there is no
    phone service) -> mapped to ``"No"``.
  * Columns such as ``OnlineSecurity`` contain ``"No internet service"`` when
    the customer has no internet service -> kept as their own category so the
    encoder can distinguish "not applicable" from a genuine "No".
  * ``customerID`` is dropped (it is a key, not a feature).
  * ``Churn`` (Yes/No) -> 1/0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# A small, *meaningful* set of engineered features. Each one encodes a
# business intuition about churn, not an arbitrary transformation.
ENGINEERED_FEATURES = [
    "tenure_years",          # tenure / 12
    "charges_per_tenure_month",  # TotalCharges / tenure (avg spend per month)
    "num_services",          # how many add-on services the customer has
    "has_internet",          # 1 if the customer has an internet service
]

_ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw(path: str | Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV as a DataFrame."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows loaded from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame: types, blanks, and value normalisation."""
    df = df.copy()

    # Drop the customer key.
    df = df.drop(columns=["customerID"])

    # TotalCharges: string -> float; blanks (short tenure) -> 0.0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # MultipleLines: "No phone service" -> "No".
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Target: Yes/No -> 1/0 (also accepts already-numeric 0/1).
    if df[config.TARGET].dtype == object:
        df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0}).astype(int)
    else:
        df[config.TARGET] = df[config.TARGET].astype(int)

    # Sanity: no NaNs in the target, no negative tenure/charges.
    if df[config.TARGET].isna().any():
        raise ValueError("Target contains NaN after cleaning")
    if (df["tenure"] < 0).any() or (df["MonthlyCharges"] < 0).any():
        raise ValueError("Negative tenure/charges present after cleaning")

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features."""
    df = df.copy()
    df["tenure_years"] = df["tenure"] / 12.0
    df["charges_per_tenure_month"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0.0
    )
    df["num_services"] = df[_ADDON_SERVICES].apply(
        lambda r: int(r.map({"Yes": 1, "No": 0}).sum()), axis=1
    )
    df["has_internet"] = (
        df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    )
    return df


def encode(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Label-encode the categorical columns.

    Returns ``(X, y, encoders)`` where ``X`` is the numeric feature matrix and
    ``y`` is the 0/1 target. When ``encoders`` is provided they are reused
    (so the API can apply exactly the same mapping as training); otherwise new
    encoders are fit on ``df``.
    """
    feature_cols = config.FEATURES + ENGINEERED_FEATURES

    if encoders is None:
        encoders = {}
        for col in config.CATEGORICAL_COLS:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc
    else:
        # Reuse the training-time mapping. Unknown categories -> -1 flag.
        for col in config.CATEGORICAL_COLS:
            enc = encoders[col]
            code_map = {str(c): i for i, c in enumerate(enc.classes_)}
            df[col] = df[col].astype(str).map(code_map).fillna(-1).astype(int)

    X = df[feature_cols].astype(float)
    y = df[config.TARGET]
    return X, y, encoders


def feature_names() -> list[str]:
    """The exact feature order used for training and serving."""
    return list(config.FEATURES + ENGINEERED_FEATURES)
