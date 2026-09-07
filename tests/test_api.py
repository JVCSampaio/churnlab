"""Tests for the FastAPI service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churnlab import config
from churnlab.model import train_all

HIGH_RISK = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
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
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0,
}


@pytest.fixture
def api_client(synthetic_X_y, monkeypatch):
    """Build a TestClient with a model trained on synthetic data."""
    from churnlab import api

    X, y, encoders = synthetic_X_y
    report, models = train_all(X, y, track=False)
    best = models[report.best_model]

    monkeypatch.setattr(
        api, "load_best_model", lambda: (best, encoders, report)
    )
    with TestClient(api.app) as client:
        yield client


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] in config.MODELS


def test_models_endpoint(api_client):
    r = api_client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "best_model" in body
    assert isinstance(body["results"], list) and len(body["results"]) >= 1


def test_predict_high_risk_has_higher_score_than_low(api_client):
    low_risk = dict(
        HIGH_RISK,
        tenure=48,
        Contract="Two year",
        OnlineSecurity="Yes",
        TechSupport="Yes",
        MonthlyCharges=30.0,
        TotalCharges=1440.0,
    )
    hi = api_client.post("/predict", json=HIGH_RISK).json()
    lo = api_client.post("/predict", json=low_risk).json()
    assert hi["churn_probability"] > lo["churn_probability"]
    assert hi["model"] in config.MODELS
    # Response is well-formed.
    assert 0.0 <= hi["churn_probability"] <= 1.0
    assert isinstance(hi["will_churn"], bool)
    assert len(hi["feature_names"]) == len(config.FEATURES) + 4


def test_predict_missing_field_is_422(api_client):
    bad = dict(HIGH_RISK)
    del bad["Contract"]
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422
