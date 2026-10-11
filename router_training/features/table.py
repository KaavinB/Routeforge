"""Builds the (case x model) training table from a dataset, collection outputs and labels.

Features are computed with ``app.routing.features.extract_features`` — the exact function
the gateway calls at serving time — from the same messages/params a client would send.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.evaluation.datasets import Dataset, EvalCase
from app.routing.features import FEATURE_NAMES, extract_features, routing_text


@dataclass
class TrainingTable:
    cases: pd.DataFrame  # one row per case: id, split, category, group_id, text + features
    outcomes: pd.DataFrame  # one row per (case, model): acceptable, score, tokens, latency, cost
    feature_matrix: np.ndarray  # aligned with ``cases`` rows
    models: list[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_table(
    dataset: Dataset, collection_dir: Path, models: list[str] | None = None
) -> TrainingTable:
    outputs = {
        (r["case_id"], r["model_id"]): r
        for r in load_jsonl(collection_dir / "outputs.jsonl")
        if not r.get("error")
    }
    labels = {(r["case_id"], r["model_id"]): r for r in load_jsonl(collection_dir / "labels.jsonl")}
    all_models = sorted({m for _, m in labels})
    models = [m for m in (models or all_models) if m in all_models]

    case_rows: list[dict[str, Any]] = []
    feats: list[np.ndarray] = []
    for c in dataset.cases:
        if not all((c.id, m) in labels for m in models):
            continue  # require a label for every model so policies are comparable
        f = extract_features(c.messages, _client_params(c))
        case_rows.append(
            {
                "case_id": c.id,
                "split": c.split,
                "category": c.category,
                "group_id": c.group_id,
                "text": routing_text(c.messages),
                **f.as_dict(),
            }
        )
        feats.append(f.vector())

    out_rows: list[dict[str, Any]] = []
    for row in case_rows:
        for m in models:
            lab = labels[(row["case_id"], m)]
            o = outputs.get((row["case_id"], m), {})
            out_rows.append(
                {
                    "case_id": row["case_id"],
                    "model_id": m,
                    "split": row["split"],
                    "category": row["category"],
                    "acceptable": bool(lab["acceptable"]),
                    "score": float(lab["score"]),
                    "input_tokens": int(o.get("input_tokens", 0)),
                    "output_tokens": int(o.get("output_tokens", 0)),
                    "latency_ms": float(o.get("latency_s", 0.0)) * 1000,
                    "cost_usd": float(o.get("cost_usd", 0.0)),
                    "errored": bool(lab.get("error")),
                }
            )
    return TrainingTable(
        cases=pd.DataFrame(case_rows),
        outcomes=pd.DataFrame(out_rows),
        feature_matrix=np.vstack(feats) if feats else np.zeros((0, len(FEATURE_NAMES))),
        models=models,
    )


def _client_params(case: EvalCase) -> dict[str, Any]:
    return case.request_params()
