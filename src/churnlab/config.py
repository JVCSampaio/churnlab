"""Central configuration for the ChurnLab pipeline."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"

RAW_CSV = DATA_RAW / "Telco-Customer-Churn.csv"
CLEAN_CSV = DATA_CLEAN / "churn_clean.csv"

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

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = ["logistic_regression", "random_forest", "xgboost"]
# The best model is chosen at training time by ROC-AUC (see model.train_all);
# this is a fallback only used before the first training run.
BEST_MODEL = "logistic_regression"
