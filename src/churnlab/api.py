"""FastAPI service that serves the trained churn model.

Accepts a raw customer record (the same columns as the raw CSV) and returns
the churn probability and class. The request schema mirrors the raw data so
callers do not need to pre-encode anything — the server reuses the exact
encoders and pipeline saved at training time. The class is taken at the
cost-tuned threshold stored in the metrics document, not at 0.5.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, config
from .data import encode, engineer
from .model import TrainingReport, load_best_model


class Customer(BaseModel):
    """A raw customer record, exactly as it appears in the source CSV."""

    customer_id: str | None = None
    gender: str
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(0, ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(0.0, ge=0)
    TotalCharges: float | None = None

    def to_frame(self) -> pd.DataFrame:
        """Build a single-row DataFrame in the raw column layout."""
        total = self.TotalCharges
        if total is None:
            total = self.MonthlyCharges * max(self.tenure, 1)
        return pd.DataFrame(
            [
                {
                    "gender": self.gender,
                    "SeniorCitizen": self.SeniorCitizen,
                    "Partner": self.Partner,
                    "Dependents": self.Dependents,
                    "tenure": self.tenure,
                    "PhoneService": self.PhoneService,
                    "MultipleLines": self.MultipleLines,
                    "InternetService": self.InternetService,
                    "OnlineSecurity": self.OnlineSecurity,
                    "OnlineBackup": self.OnlineBackup,
                    "DeviceProtection": self.DeviceProtection,
                    "TechSupport": self.TechSupport,
                    "StreamingTV": self.StreamingTV,
                    "StreamingMovies": self.StreamingMovies,
                    "Contract": self.Contract,
                    "PaperlessBilling": self.PaperlessBilling,
                    "PaymentMethod": self.PaymentMethod,
                    "MonthlyCharges": self.MonthlyCharges,
                    "TotalCharges": total,
                    config.TARGET: 0,  # placeholder; ignored at serving time
                }
            ]
        )


class Prediction(BaseModel):
    customer_id: str | None = None
    churn_probability: float
    will_churn: bool
    threshold: float
    model: str
    feature_names: list[str]


def operating_threshold(report: TrainingReport | None) -> float:
    """Threshold saved with the training report, or the comparison cutoff."""
    if report is None or not report.threshold:
        return float(config.COMPARISON_THRESHOLD)
    return float(report.threshold.get("value", config.COMPARISON_THRESHOLD))


def score_customer(
    customer: Customer,
    pipeline,
    encoders: dict,
    threshold: float,
    model_name: str,
) -> Prediction:
    """Score one raw record with the training encoders and pipeline."""
    frame = customer.to_frame()
    frame["TotalCharges"] = frame["TotalCharges"].astype(float)
    frame["MultipleLines"] = frame["MultipleLines"].replace("No phone service", "No")
    frame = engineer(frame)
    features, _, _ = encode(frame, encoders=encoders)
    proba = float(pipeline.predict_proba(features)[0, 1])
    return Prediction(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        will_churn=bool(proba >= threshold),
        threshold=round(float(threshold), 4),
        model=model_name,
        feature_names=list(features.columns),
    )


class State:
    pipeline = None
    encoders = None
    report: TrainingReport | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    best, encoders, report = load_best_model()
    State.pipeline = best
    State.encoders = encoders
    State.report = report
    yield


app = FastAPI(
    title="ChurnLab API",
    version=__version__,
    description="Telecom customer churn prediction (trained on the IBM Telco dataset).",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    threshold = operating_threshold(State.report)
    return {
        "status": "ok",
        "model": State.report.best_model if State.report else None,
        "threshold": threshold,
    }


@app.get("/models")
def models() -> dict:
    if State.report is None:
        raise HTTPException(503, "Model not loaded")
    payload = State.report.as_dict()
    threshold = dict(payload.get("threshold") or {})
    threshold.pop("curve", None)
    payload["threshold"] = threshold
    return payload


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    if State.pipeline is None or State.encoders is None or State.report is None:
        raise HTTPException(503, "Model not loaded")
    return score_customer(
        customer,
        State.pipeline,
        State.encoders,
        operating_threshold(State.report),
        State.report.best_model,
    )
