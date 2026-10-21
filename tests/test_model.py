"""Tests for model training, evaluation, and persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlab import config
from churnlab.model import (
    build_model,
    evaluate_model,
    load_best_model,
    save_artifacts,
    train_all,
)


def test_build_model_known_and_unknown():
    for name in config.MODELS:
        pipe = build_model(name)
        assert pipe.name == name
    import pytest

    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_train_all_reports_all_models(synthetic_X_y):
    X, y, encoders = synthetic_X_y
    report, models = train_all(X, y, track=False)
    # Every configured model is reported.
    assert {r.name for r in report.results} == set(config.MODELS)
    # The shipped model is the best mean CV ROC-AUC, not the holdout score.
    assert report.best_model in config.MODELS
    best = max(report.results, key=lambda r: r.cv["roc_auc_mean"])
    assert best.name == report.best_model
    assert report.holdout_best_by_roc_auc in config.MODELS
    for r in report.results:
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.roc_auc <= 1.0
        assert len(r.confusion) == 2 and len(r.confusion[0]) == 2
        assert r.cv["n_splits"] == config.N_CV_FOLDS
        assert r.cv["roc_auc_std"] >= 0.0
    # Cost ratio is explicit, and the chosen cutoff is at least as cheap as 0.5
    # on the out-of-fold scores used to tune it.
    assert report.threshold["fn_cost"] == config.FN_COST
    assert report.threshold["fp_cost"] == config.FP_COST
    assert "planning assumption" in report.threshold["assumption"]
    assert report.threshold["oof_cost"] <= report.threshold["oof_cost_at_comparison_threshold"] + 1e-9
    matrix = report.threshold["test_confusion"]
    assert sum(sum(row) for row in matrix) == report.n_test


def test_models_learn_the_signal(synthetic_X_y):
    # The synthetic data has a real signal, so every model should beat
    # the trivial "always predict majority class" baseline.
    X, y, encoders = synthetic_X_y
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    # Majority-class accuracy baseline.
    baseline_acc = max(yte.mean(), 1 - yte.mean())
    report, _ = train_all(X, y, track=False)
    best = max(report.results, key=lambda r: r.roc_auc)
    assert best.roc_auc > 0.5  # better than random
    assert best.accuracy >= baseline_acc - 1e-9


def test_save_and_load_roundtrip(synthetic_X_y, tmp_path):
    X, y, encoders = synthetic_X_y
    report, models = train_all(X, y, track=False)
    save_artifacts(models, encoders, report, out_dir=tmp_path)

    best, loaded_encoders, loaded_report = load_best_model(tmp_path)
    assert loaded_report.best_model == report.best_model
    assert loaded_report.threshold["value"] == pytest.approx(report.threshold["value"])
    assert loaded_report.results[0].cv["n_splits"] == config.N_CV_FOLDS
    assert best.name == report.best_model
    # Encoders round-trip.
    for col in config.CATEGORICAL_COLS:
        assert set(loaded_encoders[col].classes_) == set(encoders[col].classes_)
    # The saved best model can predict.
    pred = best.predict(X)
    assert pred.shape == (len(X),)
    assert set(pred) <= {0, 1}


def test_evaluate_model_confusion_matrix(synthetic_X_y):
    X, y, encoders = synthetic_X_y
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    model = build_model("logistic_regression")
    result = evaluate_model(model, Xtr, ytr, Xte, yte)
    # Confusion matrix entries are non-negative ints summing to n_test.
    total = sum(sum(row) for row in result.confusion)
    assert total == len(Xte)
    assert all(isinstance(v, int) for row in result.confusion for v in row)


def test_cv_splitter_is_stratified():
    from churnlab.model import make_cv

    y = pd.Series([0] * 80 + [1] * 20)
    rates = []
    for _, valid_idx in make_cv(5).split(np.zeros((len(y), 1)), y):
        rates.append(float(y.iloc[valid_idx].mean()))
    assert rates == [0.2, 0.2, 0.2, 0.2, 0.2]


def test_tune_threshold_known_case_and_tie_break():
    from churnlab.model import tune_threshold

    chosen, _curve = tune_threshold(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        fn_cost=5,
        fp_cost=1,
        grid=np.array([0.10, 0.50, 0.90]),
    )
    assert chosen["value"] == 0.5
    assert chosen["oof_cost"] == 0.0
    assert chosen["ratio_label"] == "5:1"
    assert chosen["fn_cost"] == 5
    assert chosen["fp_cost"] == 1

    tied, curve = tune_threshold(
        np.array([0, 1]),
        np.array([0.3, 0.8]),
        fn_cost=5,
        fp_cost=1,
        grid=np.array([0.10, 0.20]),
    )
    assert len({row["cost"] for row in curve}) == 1
    assert tied["value"] == 0.2
