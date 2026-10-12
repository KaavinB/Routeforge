"""How much should we trust the LLM judge?

    uv run python -m router_training.judge_agreement --collection hosted-2026-10-11

Inter-judge agreement: re-grade a stratified sample of summaries with a second judge
   from a different vendor that is not a routing candidate (Claude Fable 5.1) using the
   same rubric, and compare with the primary judge (GPT-4.1): raw agreement and Cohen's
   kappa on the binary ``acceptable`` label, Spearman rank correlation on each 1-5 score.

Sampling: every summary the primary judge rejected (they are rare) plus a seeded random
sample of accepted ones, stratified by model. Agreement on such a sample over-represents
hard cases relative to the natural distribution; both raw and per-class numbers are given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from app.evaluation.budget import SpendLedger
from app.evaluation.datasets import Dataset
from app.evaluation.judge import SummaryJudge, acceptable_from_scores
from benchmarks.common import experiment_metadata, write_summary
from router_training.collect import COLLECTIONS, DATASETS, LEDGER
from router_training.score import JUDGE_CACHE

SECOND_JUDGE = "anthropic/claude-fable-5-1"
SECOND_JUDGE_PARAMS: dict[str, Any] = {"max_tokens": 4096, "output_config": {"effort": "low"}}


def load_sample(collection: str, per_model: int, seed: int) -> list[dict[str, Any]]:
    d = COLLECTIONS / collection
    outputs = {
        (r["case_id"], r["model_id"]): r
        for r in map(json.loads, (d / "outputs.jsonl").read_text().splitlines())
    }
    rows = [
        r
        for r in map(json.loads, (d / "labels.jsonl").read_text().splitlines())
        if r["category"] == "summarization"
        and (r.get("details") or {}).get("method") == "llm_judge"
    ]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        r["content"] = outputs[(r["case_id"], r["model_id"])]["content"]
        by_model[r["model_id"]].append(r)
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for _model, items in sorted(by_model.items()):
        rejected = [r for r in items if not r["acceptable"]]
        accepted = [r for r in items if r["acceptable"]]
        rng.shuffle(accepted)
        sample += rejected + accepted[: max(per_model - len(rejected), 0)]
    return sample


def agreement(a: list[bool], b: list[bool]) -> dict[str, Any]:
    a_arr, b_arr = np.array(a), np.array(b)
    return {
        "n": len(a),
        "raw_agreement": float((a_arr == b_arr).mean()),
        "cohens_kappa": float(cohen_kappa_score(a_arr, b_arr))
        if len(set(a) | set(b)) > 1
        else None,
        "primary_accept_rate": float(a_arr.mean()),
        "other_accept_rate": float(b_arr.mean()),
        "agreement_on_primary_rejects": float((~b_arr[~a_arr]).mean()) if (~a_arr).any() else None,
        "agreement_on_primary_accepts": float(b_arr[a_arr].mean()) if a_arr.any() else None,
    }


async def main_async(args: argparse.Namespace) -> None:
    ds = Dataset(DATASETS / args.dataset)
    cases = {c.id: c for c in ds.cases}
    sample = load_sample(args.collection, args.per_model, args.seed)
    ledger = SpendLedger(LEDGER)
    second = SummaryJudge(
        SECOND_JUDGE, cache_path=JUDGE_CACHE, ledger=ledger, request_overrides=SECOND_JUDGE_PARAMS
    )
    sem = asyncio.Semaphore(4)

    async def grade(r: dict[str, Any]) -> dict[str, Any]:
        ref = cases[r["case_id"]].reference
        async with sem:
            return await second.judge(ref["article"], ref["highlights"], r["content"])

    verdicts = await asyncio.gather(*(grade(r) for r in sample))
    primary = [r["details"]["judge"] for r in sample]
    a = [acceptable_from_scores(p) for p in primary]
    b = [acceptable_from_scores(v) for v in verdicts]
    corr = {}
    for crit in ("faithfulness", "coverage", "conciseness"):
        x = [p[crit] for p in primary]
        y = [v[crit] for v in verdicts]
        rho = spearmanr(x, y).statistic if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")
        corr[crit] = {
            "spearman": float(rho),
            "mean_abs_diff": float(np.mean(np.abs(np.array(x) - np.array(y)))),
            "primary_mean": float(np.mean(x)),
            "other_mean": float(np.mean(y)),
        }

    result = agreement(a, b)
    summary = {
        "metadata": experiment_metadata(
            "H_judge_agreement",
            mode="hosted",
            collection=args.collection,
            primary_judge="openai/gpt-4.1",
            second_judge=SECOND_JUDGE,
            second_judge_params=SECOND_JUDGE_PARAMS,
            rubric="summary-rubric-v1",
            sample_size=len(sample),
            sampling="all primary-rejected + seeded random accepted, stratified by model",
            acceptability_rule="faithfulness >= 4 and coverage >= 3",
        ),
        "status": "complete",
        "headline": [
            {
                "label": "Inter-judge agreement (acceptable label)",
                "value": f"{100 * result['raw_agreement']:.1f}%",
                "detail": f"n={result['n']}",
            },
            {
                "label": "Cohen's kappa",
                "value": "n/a"
                if result["cohens_kappa"] is None
                else f"{result['cohens_kappa']:.2f}",
            },
            {
                "label": "Spearman (faithfulness / coverage)",
                "value": f"{corr['faithfulness']['spearman']:.2f} / {corr['coverage']['spearman']:.2f}",
            },
        ],
        "inter_judge": result,
        "score_correlation": corr,
        "limitations": [
            "Agreement between two LLM judges is not accuracy; both can share biases.",
            "The sample over-represents primary-judge rejections; kappa is sensitive to class balance.",
        ],
    }
    write_summary("H_judge_agreement", summary)
    print(json.dumps({"inter_judge": result, "score_correlation": corr}, indent=2))
    print(f"ledger spent ${ledger.spent:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="hosted-2026-10-11")
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--per-model", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
