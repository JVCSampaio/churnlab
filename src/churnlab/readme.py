"""Render README.md from the metrics document.

Every quantity in the markdown is formatted from that document. ``make train``
rewrites both files together; the test suite checks they still match.
"""
from __future__ import annotations

import json

from . import config


def _models(metrics: dict) -> dict[str, dict]:
    return {model["name"]: model for model in metrics["models"]}


def _int(value) -> str:
    return f"{int(value):,}"


def _int_pt(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def _n3(value) -> str:
    return f"{float(value):.3f}"


def _n2(value) -> str:
    return f"{float(value):.2f}"


def _pct(value) -> str:
    return f"{100 * float(value):.1f}%"


def _pct_pt(value) -> str:
    return _pct(value).replace(".", ",")


def _pm(mean, std) -> str:
    return f"{float(mean):.3f} ± {float(std):.3f}"


def _cell(text: str, emphasize: bool) -> str:
    return f"**{text}**" if emphasize else text


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _param(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _split_label(test_size: float) -> str:
    test_pct = int(round(float(test_size) * 100))
    train_pct = 100 - test_pct
    return f"{train_pct}/{test_pct}"


def format_cli(metrics: dict) -> str:
    """Terminal summary. The same figures are in the README tables."""
    dataset = metrics["dataset"]
    split = metrics["split"]
    threshold = metrics["threshold"]
    models = _models(metrics)
    ref = _n2(split["comparison_threshold"])
    lines = [
        "=== ChurnLab training report ===",
        (
            f"rows: {_int(dataset['n_rows'])}   "
            f"train: {_int(split['n_train'])}   "
            f"test: {_int(split['n_test'])}   "
            f"churn rate: {_pct(dataset['churn_rate'])}"
        ),
        f"best model (mean CV ROC-AUC): {metrics['best_model']}",
        (
            f"threshold: {_n2(threshold['value'])}   "
            f"FN:FP = {threshold['ratio_label']}   "
            f"test precision: {_n3(threshold['test_precision'])}   "
            f"test recall: {_n3(threshold['test_recall'])}"
        ),
        "",
    ]
    header = (
        f"{'model':<22} {'cv ROC-AUC':>16} {'test ROC-AUC':>13} "
        f"{'prec@' + ref:>9} {'recall@' + ref:>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name in config.MODELS:
        model = models[name]
        lines.append(
            f"{name:<22} {_pm(model['cv']['roc_auc_mean'], model['cv']['roc_auc_std']):>16} "
            f"{_n3(model['roc_auc']):>13} {_n3(model['precision']):>9} {_n3(model['recall']):>11}"
        )
    return "\n".join(lines)


def _header(metrics: dict) -> str:
    project = metrics["project"]
    dataset = metrics["dataset"]
    repo = project["repository"]
    python = project["python"]
    author = project["author"]
    folds = metrics["cv"]["n_splits"]
    chosen = config.MODEL_LABELS_PT[metrics["best_model"]]
    resumo = (
        f"**Resumo.** O ChurnLab estima a chance de um cliente de telecom cancelar o serviço "
        f"a partir do conjunto real *Telco Customer Churn* da IBM "
        f"({_int_pt(dataset['n_rows'])} clientes; taxa de churn de {_pct_pt(dataset['churn_rate'])}). "
        f"O pipeline limpa os dados, compara regressão logística, Random Forest e XGBoost "
        f"com validação cruzada estratificada ({folds} folds), registra os experimentos no MLflow "
        f"e serve o modelo escolhido em uma API FastAPI. "
        f"Pela média de ROC-AUC na validação cruzada, o modelo escolhido é {chosen}. "
        f"Autor: {author}."
    )
    pitch = (
        f"ChurnLab predicts whether a telecom customer will leave, using IBM's "
        f"Telco Customer Churn extract ({_int(dataset['n_rows'])} customers, "
        f"{_pct(dataset['churn_rate'])} churn). It cleans the billing fields, compares a "
        f"logistic regression with Random Forest and XGBoost, records each run in MLflow, "
        f"and serves the selected model through FastAPI."
    )
    stay = dataset["n_retained"] / dataset["n_rows"]
    problem = (
        f"A retention offer is cheaper than winning the customer back later, so the team "
        f"needs a ranked list and a cutoff it can afford. {_pct(dataset['churn_rate'])} of "
        f"these customers churn ({_int(dataset['n_churned'])} of {_int(dataset['n_rows'])}). "
        f"A classifier that always predicts \"stay\" is {_pct(stay)} accurate and never "
        f"identifies anyone. Ranking quality is ROC-AUC. Who actually gets a call is "
        f"precision and recall at a threshold."
    )
    badges = (
        f"[![CI](https://github.com/{repo}/actions/workflows/ci.yml/badge.svg)]"
        f"(https://github.com/{repo}/actions/workflows/ci.yml) "
        f"[![Python {python}](https://img.shields.io/badge/python-{python}-3776AB?logo=python&logoColor=white)]"
        f"(https://www.python.org/)"
    )
    return f"# ChurnLab\n\n{badges}\n\n> {resumo}\n\nA portfolio project by {author}.\n\n{pitch}\n\n{problem}\n"


def _how_to(metrics: dict) -> str:
    port = metrics["project"]["port"]
    return f"""## How to run

```bash
python -m venv .venv
source .venv/bin/activate
make setup    # editable install; no PYTHONPATH
make train    # rewrites docs/metrics.json, docs/img/, and this README
make test
make plots    # redraw figures from the saved run
make api      # http://127.0.0.1:{port}
```

`make train` is the source of the numbers below. Edit `docs/metrics.json` or the tables by hand and the test suite fails.

Docker (CI builds this image on every pull request):

```bash
docker compose up --build
curl http://127.0.0.1:{port}/health
```
"""


def _architecture() -> str:
    return """## Architecture

```mermaid
flowchart LR
    data["Data<br/>Telco Customer Churn"] --> training["Training<br/>clean, compare, tune"]
    training --> mlflow["MLflow<br/>metrics and models"]
    mlflow --> api["API<br/>FastAPI /predict"]
```

Training writes a joblib pipeline and `docs/metrics.json`. The API loads that pipeline and applies the threshold stored in the same run. MLflow records parameters, metrics, and the model artifacts for the run.

| Module | Role |
|---|---|
| `config.py` | Paths, columns, split, cost ratio, the example request |
| `data.py` | Load, clean, engineer, label-encode |
| `eda.py` | Churn rate by contract and by tenure bucket |
| `model.py` | Fit, stratified cross-validation, threshold, MLflow |
| `reporting.py` | Assemble `docs/metrics.json` |
| `plots.py` | Write `docs/img/` |
| `readme.py` | Render this file from the metrics document |
| `api.py` | `GET /health`, `GET /models`, `POST /predict` |
| `pipeline.py` | `python -m churnlab.pipeline` |
"""


def _verdict(metrics: dict, models: dict[str, dict]) -> str:
    best = metrics["best_model"]
    holdout = metrics["holdout_best_by_roc_auc"]
    folds = metrics["cv"]["n_splits"]
    best_label = config.MODEL_LABELS[best]
    cv_text = _pm(models[best]["cv"]["roc_auc_mean"], models[best]["cv"]["roc_auc_std"])
    test_auc = _n3(models[best]["roc_auc"])
    if best == holdout:
        sentence = (
            f"{best_label} is the best model on both criteria: stratified {folds}-fold "
            f"mean ROC-AUC ({cv_text}) and held-out ROC-AUC ({test_auc})."
        )
    else:
        other = config.MODEL_LABELS[holdout]
        sentence = (
            f"Cross-validation selects {best_label} (mean ROC-AUC {cv_text}). "
            f"The held-out split ranks {other} higher "
            f"(test ROC-AUC {_n3(models[holdout]['roc_auc'])} versus {test_auc}). "
            f"The shipped model is {best_label}, chosen on the training folds."
        )
    if best == "logistic_regression" and best == holdout:
        sentence += (
            " Random forest and XGBoost do not beat the linear baseline here. "
            "That is the result."
        )
    return sentence


def _rate_sentence(rows: list[dict], kind: str) -> str:
    ranked = sorted(rows, key=lambda row: row["churn_rate"], reverse=True)
    top, bottom = ranked[0], ranked[-1]
    if kind == "contract":
        return (
            f"{top['level']} is the highest, {_pct(top['churn_rate'])} of "
            f"{_int(top['n'])} customers. {bottom['level']} is the lowest, "
            f"{_pct(bottom['churn_rate'])} of {_int(bottom['n'])}."
        )
    return (
        f"The {top['level']} month bucket is the highest, {_pct(top['churn_rate'])} "
        f"of {_int(top['n'])} customers. The {bottom['level']} month bucket is the lowest, "
        f"{_pct(bottom['churn_rate'])} of {_int(bottom['n'])}."
    )


def _results(metrics: dict) -> str:
    models = _models(metrics)
    best = metrics["best_model"]
    folds = metrics["cv"]["n_splits"]
    ref = _n2(metrics["split"]["comparison_threshold"])
    dataset = metrics["dataset"]
    split = metrics["split"]
    threshold = metrics["threshold"]
    aucs = [models[name]["roc_auc"] for name in config.MODELS]
    spread = (
        f"Holdout ROC-AUC runs from {_n3(min(aucs))} to {_n3(max(aucs))}."
    )

    cv_rows = []
    test_rows = []
    for name in config.MODELS:
        model = models[name]
        emphasize = name == best
        cv = model["cv"]
        cv_rows.append(
            [
                _cell(config.MODEL_LABELS[name], emphasize),
                _cell(_pm(cv["roc_auc_mean"], cv["roc_auc_std"]), emphasize),
                _cell(_pm(cv["precision_mean"], cv["precision_std"]), emphasize),
                _cell(_pm(cv["recall_mean"], cv["recall_std"]), emphasize),
                _cell(_pm(cv["f1_mean"], cv["f1_std"]), emphasize),
            ]
        )
        test_rows.append(
            [
                _cell(config.MODEL_LABELS[name], emphasize),
                _cell(_n3(model["roc_auc"]), emphasize),
                _cell(_n3(model["precision"]), emphasize),
                _cell(_n3(model["recall"]), emphasize),
                _cell(_n3(model["f1"]), emphasize),
                _cell(_n3(model["accuracy"]), emphasize),
            ]
        )
    cv_table = _md_table(
        ["Model", "ROC-AUC", "Precision", "Recall", "F1"],
        cv_rows,
    )
    test_table = _md_table(
        ["Model", "ROC-AUC", "Precision", "Recall", "F1", "Accuracy"],
        test_rows,
    )
    tn, fp = threshold["test_confusion"][0]
    fn, tp = threshold["test_confusion"][1]
    effects = _effects_sentence(metrics)
    contract = _rate_sentence(metrics["eda"]["by_contract"], "contract")
    tenure = _rate_sentence(metrics["eda"]["by_tenure"], "tenure")
    return f"""## Results

Every figure and metric in this section is rendered from [`docs/metrics.json`](docs/metrics.json). The training run rewrites that file, the images, and this README together.

The extract has {_int(dataset['n_rows'])} customers and {_int(dataset['n_blank_total_charges'])} blank `TotalCharges` values (new accounts; filled with 0). The model sees {_int(dataset['n_model_features'])} columns, {_int(dataset['n_engineered_features'])} of them engineered: tenure in years, average charge per tenure month, count of add-on services, and an internet flag.

![Churn rate by contract type]({config.FIGURES['churn_by_contract']})

{contract}

![Churn rate by tenure bucket]({config.FIGURES['churn_by_tenure']})

{tenure}

### Cross-validation

Stratified {folds}-fold cross-validation on the training split ({_int(split['n_train'])} rows, seed {split['random_seed']}). Cells are mean ± sample standard deviation. Precision, recall, and F1 use the comparison cutoff {_ref_phrase(ref)}. ROC-AUC does not.

{cv_table}

The shipped model is the one with the {metrics['selection_criterion']}. {_verdict(metrics, models)}

### Held-out test

One stratified {_split_label(split['test_size'])} holdout ({_int(split['n_test'])} rows: {_int(split['test_churned'])} churned, {_int(split['test_retained'])} retained). This split is not used to pick the model or the threshold. {spread}

{test_table}

![ROC curves of logistic regression, random forest, and XGBoost]({config.FIGURES['roc_curves']})

### Decision threshold

{threshold['assumption']}

Average cost per customer is `(false negatives × {threshold['fn_cost']:g} + false positives × {threshold['fp_cost']:g}) / n`. The minimum is taken on out-of-fold scores of the selected model. Ties resolve to the cutoff closest to {_n2(threshold['comparison_threshold'])}.

The chosen threshold is **{_n2(threshold['value'])}**. On the held-out test set, precision is **{_n3(threshold['test_precision'])}** and recall is **{_n3(threshold['test_recall'])}** (F1 {_n3(threshold['test_f1'])}). Out-of-fold, where the cutoff was selected, precision was {_n3(threshold['oof_precision'])} and recall was {_n3(threshold['oof_recall'])}, at an average cost of {_n3(threshold['oof_cost'])} per customer. The same out-of-fold cost at the {_n2(threshold['comparison_threshold'])} reference cutoff is {_n3(threshold['oof_cost_at_comparison_threshold'])}. Test-set average cost at the chosen threshold is {_n3(threshold['test_cost'])}; that figure was not used to pick the cutoff.

![Average retention cost versus decision threshold]({config.FIGURES['cost_vs_threshold']})

![Confusion matrix of the selected model at the cost-tuned threshold]({config.FIGURES['confusion_matrix']})

On the test set the matrix is {_int(tn)} true negatives, {_int(fp)} false positives, {_int(fn)} false negatives, and {_int(tp)} true positives.

### What the selected model uses

![Feature coefficients or importances of the selected model]({config.FIGURES['feature_importance']})

{effects}
"""


def _ref_phrase(ref: str) -> str:
    return f"{ref} (the same cutoff for every model)"


def _effects_sentence(metrics: dict) -> str:
    effects = metrics["effects"]
    bits = []
    for row in effects["features"][:3]:
        label = config.FEATURE_LABELS.get(row["name"], row["name"])
        bits.append(f"{label} ({float(row['value']):+.3f})")
    joined = _join(bits)
    if effects["kind"] == "coefficient":
        shared = ""
        names = {row["name"] for row in effects["features"]}
        if "has_internet" in names and "InternetService" in names:
            shared = (
                " `has_internet` is 1 when Internet service is DSL or fiber, so those "
                "two coefficients share one signal."
            )
        return (
            f"Largest absolute standardized coefficients: {joined}. "
            "A positive coefficient raises the log-odds of churn after scaling. "
            "Each categorical column is a single label code, so the Contract coefficient "
            "is not a separate effect for month-to-month, one-year, and two-year."
            f"{shared}"
        )
    return (
        f"Largest importances: {joined}. "
        "Importances are non-negative; they do not say which way a feature pushes churn."
    )


def _serve(metrics: dict) -> str:
    port = metrics["project"]["port"]
    request = json.dumps(metrics["example"]["request"], indent=2)
    response = json.dumps(metrics["example"]["response"], indent=2)
    body = metrics["example"]["response"]
    if body["will_churn"]:
        decision = "flagged as likely to churn"
    else:
        decision = "left unflagged"
    return f"""## Serve a prediction

The request below is the example stored in the metrics file. It includes every field the API schema requires, and each category is a level from the CSV (`Credit card (automatic)`, not a shortened label). The response is what the trained model returns for that payload. `feature_names` is also included, in model order.

```bash
curl -X POST http://127.0.0.1:{port}/predict \\
  -H 'Content-Type: application/json' \\
  -d @- <<'JSON'
{request}
JSON
```

```json
{response}
```

Scored by `{body['model']}`. This customer is {decision}.
"""


def _decisions(metrics: dict) -> str:
    dataset = metrics["dataset"]
    split = metrics["split"]
    folds = metrics["cv"]["n_splits"]
    models = _models(metrics)
    param_lines = []
    for name in config.MODELS:
        pairs = ", ".join(
            f"`{key}={_param(value)}`" for key, value in models[name]["hyperparameters"].items()
        )
        param_lines.append(f"- {config.MODEL_LABELS[name]}: {pairs}")
    params = "\n".join(param_lines)
    return f"""## Technical decisions

**Logistic regression is the baseline, and it has to earn the deployment.** On {_int(dataset['n_rows'])} rows and {_int(dataset['n_model_features'])} columns a scaled linear model is a serious answer: it trains quickly, the coefficients can be read, and the tree models have to beat its ROC-AUC. The Results section states which one actually won under the settings below.

**No hyperparameter search.** The comparison uses fixed settings, so the ranking is conditional on them:

{params}

**Why the data stays in pandas.** {_int(dataset['n_rows'])} rows fit in memory and the clean step is a few vectorized casts. Spark pays for a cluster and a JVM when the table no longer fits on one machine. At this size that overhead has nothing to process.

**Holdout plus stratified {folds}-fold CV.** A single {_split_label(split['test_size'])} split (seed {split['random_seed']}) is the untouched report card. Model choice uses mean ROC-AUC on stratified folds of the training portion, so a lucky test split cannot pick the winner. Fold precision and recall are still taken at {_n2(split['comparison_threshold'])} so the three models are compared at one operating point.

**One scaler for all three estimators.** Trees do not need standardized inputs. The logistic regression does. Keeping `StandardScaler` inside every pipeline means serving cannot forget a transform, and the three models see the same matrix.

**Label encoding, with the limitation written down.** The categorical columns have two to four levels. Integer codes keep the matrix narrow and are a reasonable input to trees. They are a worse input to a linear model, because the codes invent an order. One-hot encoding would be the more conservative linear setup; it is called out under Limitations rather than hidden.

**The threshold is a cost, not 0.5.** {metrics['threshold']['assumption']} The curve in Results is that cost on out-of-fold scores. Test precision and recall are measured once, after the cutoff is frozen.

**MLflow records the run.** Parameters, per-model metrics (including the cross-validation mean and standard deviation), and the fitted pipelines go to the local file store. The API serves the joblib artifact from the same training command.
"""


def _limitations(metrics: dict) -> str:
    ratio = metrics["threshold"]["ratio_label"]
    return f"""## Limitations

- The {ratio} cost ratio is a planning assumption. Replace `FN_COST` and `FP_COST` in `config.py` and rerun `make train` before treating the threshold as a business decision.
- Label-encoded categoricals make the linear coefficients harder to read. A one-hot variant would be the fairer linear model and might change the ranking.
- Encoders are fit on the full extract before the split. With a fixed set of levels this leaks which categories exist, not the target, and it is still sloppier than fitting them on the training rows only.
- There is no probability calibration and no hyperparameter search. A tuned forest could still overtake the linear model; this repo does not claim otherwise.
- The extract is a static snapshot. Nothing here retrains when a new month of billing arrives.

Next, in that order: one-hot encoding for the linear model, a small search on tree depth and learning rate, and a calibration check before the threshold is treated as a probability.
"""


def _tests() -> str:
    return """## Tests

```bash
make test
```

Coverage is reported in CI (`pytest --cov`). The tests cover cleaning and encoding, stratified folds, the cost threshold and its tie-break, training and the joblib round-trip, the API (health, a missing field, the tuned threshold), figure writing, and two checks on this file: the curl payload validates against the API schema and against the levels in the CSV, and the README text matches `render_readme(docs/metrics.json)`. Model tests train on a synthetic table so the suite stays short; CI also reruns the pipeline on the real extract.
"""


def _layout() -> str:
    return """## Repository layout

```
churnlab/
├── data/raw/Telco-Customer-Churn.csv
├── docs/metrics.json              # regenerated by make train
├── docs/img/                      # figures embedded above
├── src/churnlab/
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── notebook/eda.ipynb
```
"""


def _license() -> str:
    return """## License

MIT. See [LICENSE](LICENSE).
"""


def render_readme(metrics: dict) -> str:
    """Return the full README for ``metrics``, ending in a single newline."""
    parts = [
        _header(metrics),
        _how_to(metrics),
        _architecture(),
        _results(metrics),
        _serve(metrics),
        _decisions(metrics),
        _limitations(metrics),
        _tests(),
        _layout(),
        _license(),
    ]
    body = "\n\n".join(part.strip() for part in parts)
    return body.strip() + "\n"
