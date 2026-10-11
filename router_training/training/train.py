"""Step 4: train the quality-aware router.

    uv run python -m router_training.training.train --collection hosted-2026-10-11

Pipeline
1. Task classifier: TF-IDF (word 1-2 grams) -> multinomial logistic
   regression, trained on the train split. Training rows for the quality models use
   *out-of-fold* task probabilities (GroupKFold), so the quality models learn from the
   same kind of imperfect task predictions they will see at serving time.
2. Per-model quality model: for each candidate LLM, P(acceptable | request features [+
   task probabilities]). Two estimator families are compared (L2 logistic regression,
   histogram gradient boosting); each is calibrated with Platt scaling via grouped
   cross-validation on the train split. The family with the lower validation log-loss
   is kept. Validation ECE becomes that model's safety margin at serving time.
3. Variants: ``full`` (features + task probs); ablations ``no_task`` (request features
   only) and ``task_only`` (task probabilities only, i.e. per-category priors).
4. Output-length table (median output tokens per model x task) and latency priors
   (least squares: latency ~ ttft + in_tokens + out_tokens) from the train split.
5. Test split is used only for the final metrics written to metadata.json.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.evaluation.datasets import Dataset
from app.evaluation.stats import ece
from app.routing.artifact import ARTIFACT_FORMAT_VERSION, save_bundle
from app.routing.features import FEATURE_NAMES, TASK_CATEGORIES
from router_training.collect import COLLECTIONS, DATASETS
from router_training.features.table import TrainingTable, build_table

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SEED = 7


def task_classifier() -> Pipeline:
    """Word uni/bi-gram TF-IDF + logistic regression. Character n-grams were dropped after
    profiling: they cost ~2-3 ms per request at serving time for no accuracy gain here."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000
                ),
            ),
            ("clf", LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced")),
        ]
    )


def _probs_matrix(clf: Pipeline, texts: list[str]) -> np.ndarray:
    raw = clf.predict_proba(texts)
    classes = list(clf.classes_)
    out = np.zeros((len(texts), len(TASK_CATEGORIES)))
    for j, t in enumerate(TASK_CATEGORIES):
        if t in classes:
            out[:, j] = raw[:, classes.index(t)]
    return out


def oof_task_probs(
    texts: list[str], labels: list[str], groups: list[str], folds: int = 5
) -> np.ndarray:
    out = np.zeros((len(texts), len(TASK_CATEGORIES)))
    for tr, te in GroupKFold(n_splits=folds).split(texts, labels, groups):
        clf = task_classifier().fit([texts[i] for i in tr], [labels[i] for i in tr])
        out[te] = _probs_matrix(clf, [texts[i] for i in te])
    return out


def candidate_estimators() -> dict[str, Any]:
    """Small, regularised grid; with ~500 training prompts per model, capacity control
    matters more than model family."""
    return {
        **{
            f"logreg_C{c}": Pipeline(
                [("scale", StandardScaler()), ("lr", LogisticRegression(C=c, max_iter=5000))]
            )
            for c in (0.05, 0.5)
        },
        "hgb": HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    y = y.astype(int)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "base_rate": float(y.mean()) if len(y) else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        "ece": ece(p, y),
        "mean_pred": float(p.mean()) if len(y) else None,
    }


def fit_quality_model(
    X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[Any, str, dict[str, Any]]:
    folds = list(GroupKFold(n_splits=5).split(X_tr, y_tr, g_tr))
    report: dict[str, Any] = {}
    fitted: dict[str, tuple[float, Any]] = {}
    if len(set(y_tr)) < 2:  # degenerate: model always/never acceptable on train
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="prior").fit(X_tr, y_tr), "prior", report
    for name, est in candidate_estimators().items():
        # ensemble=False: one base model refit on all training rows + one Platt calibrator
        # fitted on out-of-fold predictions (5x cheaper to serve than the fold ensemble).
        cal = CalibratedClassifierCV(clone(est), method="sigmoid", cv=folds, ensemble=False).fit(
            X_tr, y_tr
        )
        m = _metrics(y_val, cal.predict_proba(X_val)[:, 1])
        report[name] = m
        fitted[name] = (m["log_loss"] if m["log_loss"] is not None else float("inf"), cal)
    best = min(fitted, key=lambda k: fitted[k][0])
    # Occam / serving-cost tie-break: gradient boosting must beat the best linear model by
    # more than validation noise (0.005 log-loss) to be chosen, since linear models are
    # simpler, better calibrated out of distribution, and compile to a microsecond fast path.
    linear = min((k for k in fitted if k.startswith("logreg")), key=lambda k: fitted[k][0])
    if best == "hgb" and fitted[linear][0] - fitted["hgb"][0] <= 0.005:
        best = linear
    return fitted[best][1], best, report


def latency_prior(df: pd.DataFrame) -> dict[str, float]:
    ok = df[(df.latency_ms > 0) & (~df.errored)]
    if len(ok) < 10:
        return {"ttft_ms": 600.0, "ms_per_output_token": 25.0, "ms_per_input_token": 0.0}
    A = np.column_stack(
        [np.ones(len(ok)), ok.input_tokens.to_numpy(float), ok.output_tokens.to_numpy(float)]
    )
    coef, *_ = np.linalg.lstsq(A, ok.latency_ms.to_numpy(float), rcond=None)
    a, b, c = (max(float(x), 0.0) for x in coef)
    return {"ttft_ms": a, "ms_per_input_token": b, "ms_per_output_token": c}


def train(collection: str, dataset: str, version: str, models: list[str] | None) -> Path:
    ds = Dataset(DATASETS / dataset)
    table: TrainingTable = build_table(ds, COLLECTIONS / collection, models)
    cases = table.cases.reset_index(drop=True)
    X = table.feature_matrix
    split = cases.split.to_numpy()
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    texts = cases.text.tolist()
    cats = cases.category.tolist()
    groups = cases.group_id.to_numpy()

    # 1. task classifier
    task_clf = task_classifier().fit(
        [t for t, m in zip(texts, tr, strict=True) if m],
        [c for c, m in zip(cats, tr, strict=True) if m],
    )
    tp_all = _probs_matrix(task_clf, texts)
    tp_oof_train = oof_task_probs(
        [texts[i] for i in np.where(tr)[0]], [cats[i] for i in np.where(tr)[0]], list(groups[tr])
    )
    tp_design = tp_all.copy()
    tp_design[tr] = tp_oof_train  # train rows use out-of-fold predictions
    task_pred = np.array(TASK_CATEGORIES)[tp_all.argmax(1)]
    task_metrics = {
        s: float(accuracy_score(np.array(cats)[mask], task_pred[mask]))
        for s, mask in (("val", va), ("test", te))
    }
    task_metrics["train_oof"] = float(
        accuracy_score(np.array(cats)[tr], np.array(TASK_CATEGORIES)[tp_oof_train.argmax(1)])
    )

    designs = {"full": np.hstack([X, tp_design]), "no_task": X, "task_only": tp_design}
    outcomes = table.outcomes
    quality_models: dict[str, dict[str, Any]] = {v: {} for v in designs}
    margins: dict[str, dict[str, float]] = {v: {} for v in designs}
    per_model_report: dict[str, Any] = {}
    for m in table.models:
        acc = (
            outcomes[outcomes.model_id == m]
            .set_index("case_id")
            .loc[cases.case_id, "acceptable"]
            .to_numpy()
            .astype(int)
        )
        per_model_report[m] = {}
        for variant, D in designs.items():
            est, family, cand_report = fit_quality_model(D[tr], acc[tr], groups[tr], D[va], acc[va])
            p_val = est.predict_proba(D[va])[:, 1]
            p_test = est.predict_proba(D[te])[:, 1]
            val_m, test_m = _metrics(acc[va], p_val), _metrics(acc[te], p_test)
            quality_models[variant][m] = est
            margins[variant][m] = float(min(max(val_m["ece"] or 0.0, 0.0), 0.15))
            per_model_report[m][variant] = {
                "selected_estimator": family,
                "val_log_loss": val_m["log_loss"],
                "candidates_val": cand_report,
                "val": val_m,
                "test": test_m,
                "train_base_rate": float(acc[tr].mean()),
                "safety_margin": margins[variant][m],
            }

    # 3b. production variant: per model, the feature set with the lowest *validation*
    # log-loss (feature set treated as a hyperparameter; the test split is not consulted).
    selected_sets: dict[str, str] = {}
    quality_models["selected"], margins["selected"] = {}, {}
    for m in table.models:
        best = min(designs, key=lambda v: per_model_report[m][v]["val_log_loss"] or float("inf"))
        selected_sets[m] = best
        quality_models["selected"][m] = quality_models[best][m]
        margins["selected"][m] = margins[best][m]
        per_model_report[m]["selected"] = {**per_model_report[m][best], "feature_set": best}

    # 4. output-token table & latency priors (train split only)
    otr = outcomes[outcomes.split == "train"]
    out_tokens = {
        m: {
            t: float(otr[(otr.model_id == m) & (otr.category == t)].output_tokens.median() or 1.0)
            for t in TASK_CATEGORIES
        }
        for m in table.models
    }
    latency = {m: latency_prior(otr[otr.model_id == m]) for m in table.models}

    bundle = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "router_version": version,
        "feature_names": list(FEATURE_NAMES),
        "models": table.models,
        "task_classifier": task_clf,
        "quality_models": quality_models,
        "margins": margins,
        "selected_feature_sets": selected_sets,
        "default_variant": "selected",
        "output_tokens": out_tokens,
        "latency_priors": latency,
        "max_train_input_tokens": float(cases.loc[tr, "input_tokens_est"].max() * 1.5),
        "ood_task_confidence": 0.4,
    }
    metadata = {
        "router_version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset": ds.name,
        "dataset_version": ds.version,
        "dataset_sha256": ds.manifest.sha256,
        "collection": collection,
        "models": table.models,
        "n_cases": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "feature_names": list(FEATURE_NAMES),
        "task_classifier": {"type": "tfidf(word1-2)+logreg", "accuracy": task_metrics},
        "quality_models": per_model_report,
        "selected_feature_sets": selected_sets,
        "output_tokens_median": out_tokens,
        "latency_priors": latency,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": SEED,
    }
    out_dir = ARTIFACTS / version
    save_bundle(bundle, out_dir, metadata)
    print(
        json.dumps(
            {
                "task_accuracy": task_metrics,
                "models": {
                    m: {
                        v: {
                            "est": r["selected_estimator"],
                            "val_auc": r["val"]["auc"],
                            "test_auc": r["test"]["auc"],
                            "test_brier": r["test"]["brier"],
                            "val_ece": r["val"]["ece"],
                        }
                        for v, r in rep.items()
                    }
                    for m, rep in per_model_report.items()
                },
            },
            indent=2,
            default=str,
        )
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--version", help="artifact version (default: router-<collection>)")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    version = args.version or f"router-{args.collection}"
    path = train(args.collection, args.dataset, version, args.models)
    print(f"artifact written to {path}")


if __name__ == "__main__":
    main()
