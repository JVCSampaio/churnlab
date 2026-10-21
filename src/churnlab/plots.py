"""Figures for the README, drawn from docs/metrics.json.

ROC curves are the one view that needs the saved models: the curve itself is
not stored in the metrics file. Every other figure is a direct plot of fields
the training run already wrote down.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from . import config

INK = "#1F2933"
MUTED = "#7B8794"
GRID = "#E4E7EB"
BAR = "#3E4C59"
POS = "#334E68"
NEG = "#8D6A5A"
REF = "#9AA5B1"
MODEL_COLORS = {
    "logistic_regression": "#334E68",
    "random_forest": "#7B8794",
    "xgboost": "#8D6A5A",
}
CMAP = LinearSegmentedColormap.from_list("churnlab", ["#F7F8FA", "#243B53"])


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "axes.labelsize": 11,
            "axes.edgecolor": "#D9E2EC",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _finish(ax, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _figure_path(out_dir: Path, key: str) -> Path:
    return out_dir / Path(config.FIGURES[key]).name


def roc_curves_from_artifacts(artifacts_dir: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rebuild holdout ROC curves from the joblib models written by training."""
    import joblib
    from sklearn.metrics import roc_curve

    from .data import clean, encode, engineer, load_raw
    from .model import holdout_split

    artifacts_dir = Path(artifacts_dir or config.ARTIFACTS_DIR)
    frame = engineer(clean(load_raw()))
    features, target, _ = encode(frame)
    _, x_test, _, y_test = holdout_split(features, target)
    y_np = y_test.to_numpy()
    curves = {}
    for name in config.MODELS:
        path = artifacts_dir / f"model_{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run `make train` first.")
        model = joblib.load(path)
        proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_np, proba)
        curves[name] = (np.asarray(fpr), np.asarray(tpr))
    return curves


def _plot_roc(metrics: dict, curves: dict[str, tuple[np.ndarray, np.ndarray]]):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    aucs = {model["name"]: model["roc_auc"] for model in metrics["models"]}
    for name in config.MODELS:
        fpr, tpr = curves[name]
        label = f"{config.MODEL_LABELS[name]}  ({aucs[name]:.3f})"
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2.2, label=label)
    ax.plot([0, 1], [0, 1], color=REF, lw=1, ls="--", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _finish(ax, "Held-out ROC curves", ylabel="True positive rate", xlabel="False positive rate")
    ax.legend(frameon=False, loc="lower right")
    return fig


def _plot_confusion(metrics: dict):
    matrix = np.asarray(metrics["threshold"]["test_confusion"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(matrix, cmap=CMAP, vmin=0)
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    peak = matrix.max() if matrix.size else 1
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > peak / 2 else INK
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=color, fontsize=16)
    threshold = metrics["threshold"]["value"]
    model = config.MODEL_LABELS[metrics["best_model"]]
    ax.set_title(f"{model} on the test set, threshold {threshold:.2f}", loc="left", pad=10, color=INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(False)
    return fig


def _plot_rates(rows: list[dict], title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    labels = [f"{row['level']}    n={row['n']:,}" for row in rows]
    rates = [float(row["churn_rate"]) for row in rows]
    ax.barh(labels, rates, color=BAR, height=0.62)
    ax.invert_yaxis()
    peak = max(rates) if rates else 1.0
    ceiling = min(1.0, max(peak + 0.16, peak * 1.35))
    ax.set_xlim(0, ceiling)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for index, rate in enumerate(rates):
        ax.text(
            rate + ceiling * 0.02,
            index,
            f"{rate:.1%}",
            va="center",
            ha="left",
            color=INK,
            fontsize=10,
        )
    _finish(ax, title, xlabel=xlabel)
    ax.set_ylabel("")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", length=0)
    return fig


def _plot_effects(metrics: dict):
    effects = metrics["effects"]
    features = list(reversed(effects["features"]))
    labels = [config.FEATURE_LABELS.get(row["name"], row["name"]) for row in features]
    values = [row["value"] for row in features]
    if effects["kind"] == "coefficient":
        colors = [POS if value >= 0 else NEG for value in values]
        xlabel = "Standardized coefficient (log-odds of churn)"
        title = f"Coefficients — {config.MODEL_LABELS[effects['model']]}"
    else:
        colors = [BAR for _ in values]
        xlabel = "Impurity importance"
        title = f"Feature importance — {config.MODEL_LABELS[effects['model']]}"
    height = max(4.4, 0.32 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(7.4, height))
    ax.barh(labels, values, color=colors, height=0.7)
    if effects["kind"] == "coefficient":
        ax.axvline(0, color=REF, lw=0.9)
    span = max(abs(value) for value in values) if values else 1
    if effects["kind"] == "coefficient":
        ax.set_xlim(-span * 1.18, span * 1.18)
    else:
        ax.set_xlim(0, span * 1.18)
    _finish(ax, title, xlabel=xlabel)
    ax.set_ylabel("")
    return fig


def _plot_cost(metrics: dict):
    curve = metrics["threshold"]["curve"]
    chosen = float(metrics["threshold"]["value"])
    reference = float(metrics["split"]["comparison_threshold"])
    fn_cost = metrics["threshold"]["fn_cost"]
    fp_cost = metrics["threshold"]["fp_cost"]
    thresholds = [row["threshold"] for row in curve]
    costs = [row["cost"] for row in curve]
    chosen_cost = min(curve, key=lambda row: abs(row["threshold"] - chosen))["cost"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(thresholds, costs, color=POS, lw=2.2, label="Average cost")
    ax.axvline(chosen, color=NEG, lw=1.6, label=f"Chosen {chosen:.2f}")
    ax.axvline(reference, color=REF, lw=1.2, ls="--", label=f"Reference {reference:.2f}")
    ax.scatter([chosen], [chosen_cost], color=NEG, zorder=3, s=28)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(costs) * 1.08 if costs else 1)
    ratio = f"{fn_cost:g}:{fp_cost:g}"
    _finish(
        ax,
        f"Cost versus threshold  (FN:FP = {ratio})",
        ylabel="Average cost per customer",
        xlabel="Decision threshold",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.set_axisbelow(True)
    return fig


def write_figures(
    metrics: dict,
    out_dir: Path | None = None,
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[Path]:
    """Write the six README figures. Pass ``roc_curves`` to skip loading models."""
    out_dir = Path(out_dir or config.FIGURES_DIR)
    if roc_curves is None:
        roc_curves = roc_curves_from_artifacts()
    _apply_style()
    contract = _plot_rates(
        metrics["eda"]["by_contract"],
        "Churn rate by contract",
        "Churn rate",
    )
    tenure = _plot_rates(
        metrics["eda"]["by_tenure"],
        "Churn rate by tenure",
        "Churn rate",
    )
    paths = [
        _save(_plot_roc(metrics, roc_curves), _figure_path(out_dir, "roc_curves")),
        _save(_plot_confusion(metrics), _figure_path(out_dir, "confusion_matrix")),
        _save(contract, _figure_path(out_dir, "churn_by_contract")),
        _save(tenure, _figure_path(out_dir, "churn_by_tenure")),
        _save(_plot_effects(metrics), _figure_path(out_dir, "feature_importance")),
        _save(_plot_cost(metrics), _figure_path(out_dir, "cost_vs_threshold")),
    ]
    return paths


def main() -> int:
    if not config.METRICS_PATH.is_file():
        raise SystemExit("docs/metrics.json is missing. Run `make train` first.")
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    for path in write_figures(metrics):
        print(path.relative_to(config.PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
