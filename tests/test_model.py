"""Tests for model training, evaluation, and persistence."""
from __future__ import annotations

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
    # A best model is chosen.
    assert report.best_model in config.MODELS
    # Metrics are in a sane range.
    for r in report.results:
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.roc_auc <= 1.0
        assert len(r.confusion) == 2 and len(r.confusion[0]) == 2
    # The best model is the max ROC-AUC.
    best = max(report.results, key=lambda r: r.roc_auc)
    assert best.name == report.best_model


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
