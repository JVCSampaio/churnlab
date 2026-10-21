"""The README is generated from docs/metrics.json, including the curl example."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from churnlab import config
from churnlab.api import Customer
from churnlab.readme import render_readme

# Fields the published example must carry. Several of these have no default
# on the API schema; InternetService and PaperlessBilling are the ones a
# short curl example is most likely to drop.
EXAMPLE_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]


def _readme_text() -> str:
    return Path(config.README_PATH).read_text(encoding="utf-8")


def _metrics() -> dict:
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def readme_predict_payload() -> dict:
    match = re.search(r"<<'JSON'\n(.*?)\nJSON", _readme_text(), re.S)
    assert match, "README is missing the predict payload heredoc"
    return json.loads(match.group(1))


def test_readme_is_rendered_from_metrics():
    metrics = _metrics()
    rendered = render_readme(metrics)
    text = _readme_text()
    if text != rendered:
        import difflib

        diff = "".join(
            difflib.unified_diff(
                text.splitlines(True),
                rendered.splitlines(True),
                fromfile="README.md",
                tofile="render_readme",
                n=2,
            )
        )
        raise AssertionError(diff[:4000])


def test_render_tracks_metric_values():
    metrics = _metrics()
    original = render_readme(metrics)
    mutated = json.loads(json.dumps(metrics))
    mutated["models"][0]["roc_auc"] = 0.111
    updated = render_readme(mutated)
    assert "0.111" in updated
    assert updated != original


def test_readme_example_payload_matches_schema_and_dataset():
    payload = readme_predict_payload()
    missing = [field for field in EXAMPLE_FIELDS if field not in payload]
    assert missing == []
    customer = Customer.model_validate(payload)
    assert customer.InternetService
    assert customer.PaperlessBilling
    assert customer.PaymentMethod == "Credit card (automatic)"

    raw = pd.read_csv(config.RAW_CSV)
    for column in config.CATEGORICAL_COLS:
        observed = set(raw[column].astype(str))
        assert str(payload[column]) in observed, column
    assert int(payload["SeniorCitizen"]) in set(raw["SeniorCitizen"].astype(int))
