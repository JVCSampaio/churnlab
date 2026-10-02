# ChurnLab

[![CI](https://github.com/JVCSampaio/churnlab/actions/workflows/ci.yml/badge.svg)](https://github.com/JVCSampaio/churnlab/actions/workflows/ci.yml) · [Portfólio](https://github.com/JVCSampaio)

**Predicting telecom customer churn from real data — a complete, reproducible ML pipeline with a serving API.**

ChurnLab takes a real-world dataset (IBM's *Telco Customer Churn*, 7,043 customers), cleans it, engineers features, trains and compares three models (a logistic-regression baseline against Random Forest and XGBoost), evaluates them with the metrics that actually matter for an imbalanced binary problem, tracks the runs in **MLflow**, and serves the best model through a **FastAPI** endpoint.

The repository documents data preparation, model evaluation, experiment tracking and API serving, with instructions to reproduce the pipeline.

---

## Problem

A telecom company loses customers ("churns") — and winning back a lost customer costs far more than keeping an existing one. The task: **given a customer's contract, services, usage, and billing history, predict whether they will churn**, so the retention team can target the highest-risk accounts.

Key characteristics of the problem:

- **Class imbalance.** ~26.5% of customers churn. A model that always says "won't churn" scores 73.5% accuracy — useless. So accuracy alone is a trap; we report **precision, recall, F1, and ROC-AUC**.
- **Noisy, messy source data.** `TotalCharges` is stored as a *string* with blanks; several columns contain `"No internet service"`; `Churn` is `"Yes"`/`"No"`. Cleaning is part of the job.
- **A decision threshold, not just a score.** The business wants *which* customers to call, so the output is a probability plus a class at a chosen threshold.

---

## Architecture

```
 data/raw/Telco-Customer-Churn.csv          (IBM Telco Customer Churn, 7,043 rows)
        │
        ▼
  data.py  load_raw → clean → engineer → encode
        │        (types, missing, derived features, label encoding)
        ▼
  model.py  train_all → {logistic_regression, random_forest, xgboost}
        │        (StandardScaler + estimator, 80/20 split, evaluate)
        ▼
        ├─► TrainingReport  (accuracy, precision, recall, F1, ROC-AUC, confusion)
        ├─► MLflow run      (metrics + parameters + artifacts)
        └─► artifacts/      (best model, encoders, report.json, feature_names)
        │
        ▼
  api.py    FastAPI  POST /predict  (loads best model at startup)
        │
        ▼
  docker  (single container, uvicorn on :8000)
```

**Modules**

| File | Responsibility |
|---|---|
| `config.py` | Paths, column lists, hyperparameters, model registry |
| `data.py` | Load → clean (types, missing, target) → engineer (derived features) → encode (label-encode categoricals, one-hot-free numeric features) |
| `eda.py` | Small summarised EDA frames (class balance, churn by category/tenure, numeric summary) |
| `model.py` | Build pipelines, train, evaluate, MLflow tracking, persist artifacts |
| `api.py` | FastAPI service — `POST /predict`, `GET /health`, `GET /model` |
| `pipeline.py` | CLI entry point that runs the whole thing and prints the report |
| `notebook/eda.ipynb` | Executed EDA notebook with real outputs |

---

## Technical Decisions

**1. Why logistic regression is the baseline (and why it wins here).**
On a 7k-row tabular problem with ~20 features, a well-scaled linear model is a strong, interpretable baseline. It trains in milliseconds, its coefficients are inspectable, and it beats the tree models on ROC-AUC (below). Choosing it as the *baseline* — not the final answer — is the point: it gives a reference bar the fancier models must beat.

**2. Why not Spark?**
The dataset is 7,043 rows. Pandas/NumPy handles it in memory in under a second. Spark's whole value is *out-of-core* and *distributed* processing — it adds a cluster, a JVM, and serialization overhead for **zero** benefit at this scale. If the data grew to hundreds of millions of rows, Spark would become the right tool; at this size it is cargo-culting. (This is the exact answer to "why Spark?" an interviewer asks.)

**3. Why 80/20 and a fixed seed.**
A single deterministic split (seed `42`) makes results reproducible and comparable across runs. For a final model you'd use stratified k-fold to reduce variance — noted in *Limitations*.

**4. Why `scale` before trees but not strictly needed.**
Random Forest and XGBoost are invariant to monotonic feature scaling, but the shared `StandardScaler` keeps the pipeline uniform and correct for the logistic-regression baseline. One pipeline, three models, no special-casing.

**5. Why label-encode categoricals (not one-hot).**
With low-cardinality columns (2–4 categories) and tree models, integer label codes are fine and keep the feature matrix narrow. For the linear model the ordering is arbitrary but the model treats each code as a distinct feature. One-hot would be the more conservative choice for the linear model; this is a deliberate trade-off documented here.

**6. Why MLflow.**
It records *what* was trained (hyperparameters), *how well* it did (metrics), and *where* the model lives (artifacts) in one place — the audit trail an interviewer (or a future you) needs to reproduce the result.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The raw CSV is already in `data/raw/` (committed). No credentials or API keys are required.

---

## Usage

**Run the full pipeline (train + save artifacts):**

```bash
PYTHONPATH=src python -m churnlab.pipeline
# add --no-track to skip the MLflow run
```

Example output (real run, seed 42):

```
=== ChurnLab training report ===
train rows: 5634   test rows: 1409
class balance (test): {'churned': 374, 'retained': 1035}
best model (by ROC-AUC): logistic_regression

model                 acc   prec    rec     f1    auc  time(s)
-------------------- ----- ------ ------ ------ ------ --------
logistic_regression 0.796 0.636  0.543  0.586  0.840     0.02
random_forest       0.797 0.651  0.508  0.571  0.837     0.35
xgboost             0.787 0.622  0.505  0.558  0.830     0.30
```

**Start the API:**

```bash
PYTHONPATH=src uvicorn churnlab.api:app --host 0.0.0.0 --port 8000
```

**Predict a customer:**

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "CUST_001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Credit card",
    "MonthlyCharges": 20.0,
    "TotalCharges": 60.0
  }'
```

```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.61,
  "will_churn": true,
  "threshold": 0.5
}
```

**Docker:**

```bash
docker compose up --build
# then: curl http://localhost:8000/health
```

---

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

14 tests, three areas:

- **`test_data.py`** — cleaning (types, missing, target), feature engineering (derived columns), encoding (deterministic, re-usable encoders).
- **`test_model.py`** — each model trains and produces a full metric set; ROC-AUC in [0,1]; confusion matrix shape; best model is selected by ROC-AUC; artifacts round-trip through `joblib`.
- **`test_api.py`** — health endpoint, model loaded after startup, a high-risk record predicts churn, a low-risk record predicts retention, a malformed request returns 422.

Tests run on a small deterministic synthetic dataset so the suite finishes in seconds while still exercising the real pipeline code.

---

## Benchmarks / Metrics

Real results (80/20 split, seed 42, 5,634 train / 1,409 test):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.796 | 0.636 | 0.543 | 0.586 | **0.840** |
| Random Forest | 0.797 | 0.651 | 0.508 | 0.571 | 0.837 |
| XGBoost | 0.787 | 0.622 | 0.505 | 0.558 | 0.830 |

**Reading the table (the part an interviewer wants to hear):**

- All three models land in a tight band (ROC-AUC 0.83–0.84). On this data, **model choice is not the bottleneck** — feature quality and the decision threshold are.
- **Logistic regression wins on ROC-AUC.** That is the honest, defensible answer to "why did you pick this model?": the simpler model is at least as good as the fancier ones here, trains 15× faster, and is interpretable. There is no free lunch from reaching for XGBoost.
- **Accuracy is nearly identical** (0.79–0.80) but recall and F1 differ — the models trade off differently between catching churners and flagging false alarms. For a retention campaign you'd tune the threshold to hit a target recall on the *churn* class, not chase accuracy.
- The **baseline matters**: it proves the hard models earn their keep (they don't, here), which is exactly the evidence a good data-science answer needs.

---

## Limitations

- **Single 80/20 split.** No stratified k-fold, so the metrics carry run-to-run variance. A production version should report mean ± std over folds.
- **Threshold fixed at 0.5.** The business cost of a false negative (missing a churner) is usually higher than a false positive (calling a loyal customer). The API exposes the threshold but does not yet tune it against a cost matrix.
- **Label-encoding of categoricals** is fine for trees but suboptimal for the linear model (one-hot would be more conservative).
- **No feature selection.** All engineered features are used; a pruning step (e.g., mutual information or L1) could shrink the model.
- **Static snapshot.** This is a one-time prediction, not a live system that retrains as new data arrives.
- **Docker daemon** could not be validated on the author's local machine (license-blocked); the CI build is the verification path.

---

## Roadmap

- **Stratified k-fold** evaluation with confidence intervals.
- **Cost-aware threshold tuning** (precision/recall operating point from a business cost matrix).
- **Feature engineering pass** — usage ratios, charge velocity, rolling aggregates.
- **One-hot encoding** variant for the linear model; compare against label-encoding.
- **MLflow model registry** + a second run for hyperparameter search (RF/XGBoost `n_estimators`, `max_depth`, `learning_rate`).
- **Retraining loop** — a scheduled job that retrains on fresh data and registers the new model.
- **Airflow / dbt** upstream if the data source grows beyond a single CSV.

---

## Repository layout

```
churnlab/
├── data/raw/Telco-Customer-Churn.csv   # real dataset (committed)
├── src/churnlab/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── model.py
│   ├── api.py
│   └── pipeline.py
├── notebook/eda.ipynb                  # executed EDA (real outputs)
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ruff.toml
├── .gitignore
└── LICENSE
```

---

## License

MIT — see [LICENSE](LICENSE).
