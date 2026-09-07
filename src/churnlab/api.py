"""FastAPI service that serves the trained churn model.

Accepts a raw customer record (the same columns as the raw CSV) and returns
the churn probability and class. The request schema mirrors the raw data so
callers do not need to pre-encode anything — the server reuses the exact
encoders and pipeline saved at training time.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, config
from .data import encode, engineer
from .model import TrainingReport, load_best_model


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class Customer(BaseModel):
    """A raw customer record, exactly as it appears in the source CSV."""

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
    model: str
    feature_names: list[str]


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
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
    return {"status": "ok", "model": State.report.best_model if State.report else None}


@app.get("/models")
def models() -> dict:
    if State.report is None:
        raise HTTPException(503, "Model not loaded")
    return State.report.as_dict()


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    if State.pipeline is None or State.encoders is None:
        raise HTTPException(503, "Model not loaded")

    frame = customer.to_frame()
    # Apply the same cleaning/feature engineering as training.
    frame["TotalCharges"] = frame["TotalCharges"].astype(float)
    frame["MultipleLines"] = frame["MultipleLines"].replace("No phone service", "No")
    frame = engineer(frame)
    X, _, _ = encode(frame, encoders=State.encoders)

    proba = float(State.pipeline.predict_proba(X)[0, 1])
    will_churn = proba >= 0.5

    return Prediction(
        customer_id=None,
        churn_probability=round(proba, 4),
        will_churn=will_churn,
        model=State.report.best_model,
        feature_names=list(X.columns),
    )
