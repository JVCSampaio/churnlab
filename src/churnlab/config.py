"""Central configuration for the ChurnLab pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """Locate the repository root for both editable installs and Docker.

    An editable install keeps this file under ``src/churnlab``. A regular
    install copies it into site-packages, so the image sets ``CHURNLAB_ROOT``.
    """
    override = os.environ.get("CHURNLAB_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "data" / "raw").is_dir():
            return candidate
    return here.parents[2]


PROJECT_ROOT = _project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "img"
METRICS_PATH = DOCS_DIR / "metrics.json"
README_PATH = PROJECT_ROOT / "README.md"

RAW_CSV = DATA_RAW / "Telco-Customer-Churn.csv"
CLEAN_CSV = DATA_CLEAN / "churn_clean.csv"

# Repo-relative figure paths. The README links to these; plots write the
# matching filenames under FIGURES_DIR.
FIGURES = {
    "roc_curves": "docs/img/roc_curves.png",
    "confusion_matrix": "docs/img/confusion_matrix.png",
    "churn_by_contract": "docs/img/churn_by_contract.png",
    "churn_by_tenure": "docs/img/churn_by_tenure.png",
    "feature_importance": "docs/img/feature_importance.png",
    "cost_vs_threshold": "docs/img/cost_vs_threshold.png",
}

# ---------------------------------------------------------------------------
# Data / modelling constants
# ---------------------------------------------------------------------------
TARGET = "Churn"

# Categorical columns (Yes/No or multi-category). Encoded with LabelEncoder.
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numeric columns kept as-is (tenure, charges).
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns that are already 0/1 (no encoding needed).
BINARY_COLS = ["SeniorCitizen"]

# Feature order used everywhere (deterministic).
FEATURES = BINARY_COLS + NUMERIC_COLS + CATEGORICAL_COLS

# ---------------------------------------------------------------------------
# Splits / reproducibility
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_CV_FOLDS = 5
# Cutoff used only to compare models on precision/recall/F1. The business
# threshold is chosen separately from the cost ratio below.
COMPARISON_THRESHOLD = 0.5

# Planning assumption, not an estimate from this dataset: missing a churner
# (false negative) costs five times an unnecessary retention contact
# (false positive). Change these two numbers and retrain to retune.
FN_COST = 5.0
FP_COST = 1.0

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = ["logistic_regression", "random_forest", "xgboost"]
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
MODEL_LABELS_PT = {
    "logistic_regression": "regressão logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

FEATURE_LABELS = {
    "tenure": "Tenure (months)",
    "tenure_years": "Tenure (years)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "charges_per_tenure_month": "Avg. charge per tenure month",
    "num_services": "Add-on services",
    "has_internet": "Has internet",
    "SeniorCitizen": "Senior citizen",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "InternetService": "Internet service",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "gender": "Gender",
    "Partner": "Partner",
    "Dependents": "Dependents",
}

# The best model is chosen at training time by mean CV ROC-AUC.
BEST_MODEL = "logistic_regression"

# ---------------------------------------------------------------------------
# README / API identity copied into docs/metrics.json at train time
# ---------------------------------------------------------------------------
AUTHOR = "João Vitor Sampaio"
PYTHON_VERSION = "3.11"
API_PORT = 8000
REPOSITORY = "JVCSampaio/churnlab"

# Illustrative request. Every categorical value is a level that appears in
# the IBM extract, including the full payment-method label.
EXAMPLE_CUSTOMER = {
    "customer_id": "CUST_001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 70.35,
    "TotalCharges": 211.05,
}
