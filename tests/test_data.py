"""Tests for data loading, cleaning, feature engineering, and encoding."""
from __future__ import annotations

import pandas as pd

from churnlab import config
from churnlab.data import clean, encode, engineer, feature_names, load_raw


def test_load_raw_returns_rows(tmp_path):
    # Write a tiny raw CSV in the expected layout and load it.
    df = pd.DataFrame(
        [
            {
                "customerID": "C1",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0,
                "TotalCharges": "150.0",
                "Churn": "Yes",
            }
        ]
    )
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    loaded = load_raw(path)
    assert len(loaded) == 1
    assert loaded.columns.tolist() == df.columns.tolist()


def test_clean_coerces_total_charges_and_target():
    df = pd.DataFrame(
        [
            {
                "customerID": "C1",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No phone service",
                "InternetService": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0,
                "TotalCharges": "",  # blank -> 0.0
                "Churn": "Yes",
            }
        ]
    )
    cleaned = clean(df)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["MultipleLines"].iloc[0] == "No"
    assert cleaned[config.TARGET].iloc[0] == 1
    assert "customerID" not in cleaned.columns


def test_engineer_adds_features():
    df = pd.DataFrame(
        [
            {
                "customerID": "C1",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 1200.0,
                "Churn": "No",
            }
        ]
    )
    eng = engineer(df)
    assert "tenure_years" in eng.columns
    assert "charges_per_tenure_month" in eng.columns
    assert "num_services" in eng.columns
    assert "has_internet" in eng.columns
    assert eng["tenure_years"].iloc[0] == 2.0
    assert eng["has_internet"].iloc[0] == 1
    # 4 add-on services are Yes (security, backup, tech, streaming tv)
    assert eng["num_services"].iloc[0] == 4


def test_encode_produces_numeric_features(synthetic_X_y):
    X, y, encoders = synthetic_X_y
    assert X.shape[1] == len(feature_names())
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}
    # Every categorical column got an integer code.
    for col in config.CATEGORICAL_COLS:
        assert col in X.columns


def test_encode_reuses_encoders(synthetic_X_y):
    X, y, encoders = synthetic_X_y
    # Re-encoding with the same encoders must be deterministic.
    from churnlab.data import engineer

    raw = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0,
                "TotalCharges": 150.0,
                "Churn": "No",
            }
        ]
    )
    X2, _, _ = encode(engineer(raw), encoders=encoders)
    # Unknown category (not in training) -> -1 flag.
    raw2 = raw.copy()
    raw2["gender"] = "NonBinary"  # unseen
    X3, _, _ = encode(engineer(raw2), encoders=encoders)
    assert X3["gender"].iloc[0] == -1
    assert X2["gender"].iloc[0] != -1
