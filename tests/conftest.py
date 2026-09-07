"""Shared test fixtures.

Builds a small, deterministic synthetic churn dataset (a few hundred rows) so
the test suite runs in seconds instead of training on the full 7k-row file.
The synthetic data has a real signal (month-to-month + short tenure -> churn)
so the models can actually learn something.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlab import config


def _make_synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=n)
    month_to_month = rng.random(size=n) < 0.7
    has_internet = rng.random(size=n) < 0.85
    has_security = rng.random(size=n) < 0.3
    has_tech_support = rng.random(size=n) < 0.3

    # Churn probability: high for month-to-month + short tenure + no support.
    p = 0.05
    p += 0.35 * month_to_month  # month-to-month -> +0.35
    p += 0.25 * (tenure < 12)  # short tenure -> +0.25
    p += 0.10 * (~has_security)
    p += 0.08 * (~has_tech_support)
    p = np.clip(p, 0.0, 0.95)
    churn = (rng.random(size=n) < p).astype(int)

    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.integers(0, 2, size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": rng.choice(["Yes", "No"], size=n),
            "InternetService": np.where(has_internet, "DSL", "No"),
            "OnlineSecurity": np.where(has_security, "Yes", "No"),
            "OnlineBackup": rng.choice(["Yes", "No"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No"], size=n),
            "TechSupport": np.where(has_tech_support, "Yes", "No"),
            "StreamingTV": rng.choice(["Yes", "No"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No"], size=n),
            "Contract": np.where(month_to_month, "Month-to-month", "Two year"),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
            ),
            "MonthlyCharges": np.round(rng.uniform(18, 120, size=n), 2),
            "TotalCharges": np.round(rng.uniform(18, 120, size=n) * tenure, 2),
            config.TARGET: churn,
        }
    )
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic()


@pytest.fixture
def synthetic_X_y(synthetic_df: pd.DataFrame):
    from churnlab.data import clean, encode, engineer

    df = engineer(clean(synthetic_df))
    X, y, encoders = encode(df)
    return X, y, encoders
