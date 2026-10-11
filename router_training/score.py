"""Step 3: score every collected output and produce router-training labels.

    uv run python -m router_training.score --collection hosted-2026-10-11

Writes ``labels.jsonl`` next to the collection's outputs: one row per (case, model) with
the score, the binary ``acceptable`` label and scorer details. Summaries are graded by
the LLM judge (paid; cached; budget-capped). Generated code runs in the Docker sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from app.evaluation.budget import SpendLedger
from app.evaluation.datasets import Dataset
from app.evaluation.judge import SummaryJudge
from app.evaluation.scoring import Scorer
from router_training.collect import COLLECTIONS, DATASETS, LEDGER

JUDGE_CACHE = Path(__file__).resolve().parent / "data" / "judge_cache.jsonl"


def load_outputs(collection: str) -> list[dict[str, object]]:
    path = COLLECTIONS / collection / "outputs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # keep the last successful record per (case, model)
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for r in rows:
        key = (str(r["case_id"]), str(r["model_id"]))
        if key not in latest or not r.get("error"):
            latest[key] = r
    return list(latest.values())


async def score_collection(collection: str, dataset: str, judge_model: str, no_judge: bool) -> Path:
    ds = Dataset(DATASETS / dataset)
    cases = {c.id: c for c in ds.cases}
    outputs = load_outputs(collection)
    judge = (
        None
        if no_judge
        else SummaryJudge(judge_model, cache_path=JUDGE_CACHE, ledger=SpendLedger(LEDGER))
    )
    scorer = Scorer(judge=judge, sandbox_backend="docker", judge_concurrency=6)
    items = [
        (cases[str(o["case_id"])], None if o.get("error") else str(o["content"])) for o in outputs
    ]
    results = await scorer.score_many(items)
    out = COLLECTIONS / collection / "labels.jsonl"
    summary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with out.open("w") as fh:
        for o, r in zip(outputs, results, strict=True):
            row = {
                "case_id": o["case_id"],
                "model_id": o["model_id"],
                "split": o["split"],
                "category": o["category"],
                "score": r.score,
                "acceptable": r.acceptable,
                "error": o.get("error"),
                "details": r.details,
            }
            fh.write(json.dumps(row, default=str) + "\n")
            s = summary[f"{o['model_id']}|{o['category']}"]
            s["n"] += 1
            s["acceptable"] += float(r.acceptable)
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for key, s in summary.items():
        model, cat = key.split("|")
        table[model][cat] = round(s["acceptable"] / s["n"], 3)
    print(json.dumps(table, indent=2, sort_keys=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--judge-model", default="openai/gpt-4.1")
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="ROUGE fallback for summaries (not for headline metrics)",
    )
    args = ap.parse_args()
    path = asyncio.run(
        score_collection(args.collection, args.dataset, args.judge_model, args.no_judge)
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
