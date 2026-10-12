"""Benchmark A (live): routing policies end-to-end through the running gateway.

Every held-out test prompt is sent through ``POST /v1/chat/completions`` for each policy
(cache bypassed), answers are scored with the same scorers/judge as training, and
latency is client-observed end-to-end (gateway + provider + network). This validates the
offline replay with fresh generations (Claude 5.5 models do not accept temperature=0, so
outputs can differ run to run).

Paid in hosted mode. Every call's estimated cost is appended to the spend ledger and the
run refuses to start if the pre-run estimate would exceed the remaining budget.

    uv run python -m benchmarks.evaluation.live_policies --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.evaluation.budget import SpendLedger
from app.evaluation.datasets import Dataset
from app.evaluation.judge import SummaryJudge
from app.evaluation.runner import CaseOutcome, HttpClient, TargetOptions, run_eval, summarize
from app.evaluation.scoring import Scorer
from app.evaluation.stats import (
    cost_reduction_pct,
    paired_bootstrap,
    quality_diff_pp,
    quality_retention_pct,
)
from benchmarks.common import RESULTS, experiment_metadata, write_csv, write_summary
from benchmarks.gateway_client import Gateway, env_value
from router_training.collect import DATASETS, LEDGER
from router_training.score import JUDGE_CACHE

POLICIES = [
    "routeforge-premium",
    "routeforge-economy",
    "routeforge-random",
    "routeforge-heuristic",
    "routeforge-learned",
]


def cols(rows: list[CaseOutcome]) -> dict[str, list[float]]:
    rows = sorted(rows, key=lambda r: r.case_id)
    return {
        "acceptable": [float(r.acceptable) for r in rows],
        "cost": [r.cost_usd for r in rows],
        "latency_ms": [r.latency_ms or 0.0 for r in rows],
    }


async def main_async(args: argparse.Namespace) -> None:
    gw = Gateway(args.base_url)
    ready = await gw.ready()
    await gw.close()
    if ready.get("simulated"):
        raise SystemExit("gateway is in mock mode; live policy comparison needs real models")
    ds = Dataset(DATASETS / args.dataset)
    cases = ds.select("test")
    if args.limit:
        cases = cases[: args.limit]

    ledger = SpendLedger(LEDGER)
    offline = (
        json.loads((RESULTS / args.offline / "summary.json").read_text()) if args.offline else None
    )
    if offline:
        est = sum(
            b["cost_per_request_usd"] * len(cases)
            for k, b in offline["policies"].items()
            if k.split(" ")[0] in ("random", "heuristic")
            or k.startswith("learned (")
            or "(premium)" in k
            or "(economy)" in k
        )
        est = est * 1.5 + 0.004 * len(cases)  # + judge calls on summaries (pessimistic)
        print(
            f"pre-run estimate ~${est:.2f}; ledger spent ${ledger.spent:.2f} of ${ledger.cap_usd:.2f}"
        )
        ledger.check(est)

    client = HttpClient(args.base_url, env_value("ROUTEFORGE_BOOTSTRAP_CLIENT_KEY"))
    judge = SummaryJudge(args.judge_model, cache_path=JUDGE_CACHE, ledger=ledger)
    scorer = Scorer(judge=judge, sandbox_backend="docker", judge_concurrency=6)

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} requests", flush=True)

    outcomes = await run_eval(
        cases,
        POLICIES,
        client,
        scorer,
        opts=TargetOptions(quality_threshold=args.threshold),
        concurrency=args.concurrency,
        progress=progress,
    )
    await client.aclose()
    gateway_cost = sum(o.cost_usd for o in outcomes)
    ledger.record(
        activity="live_policy_benchmark",
        model="mixed",
        cost_usd=gateway_cost,
        input_tokens=sum(o.input_tokens for o in outcomes),
        output_tokens=sum(o.output_tokens for o in outcomes),
    )

    summ = summarize(outcomes)
    by_target: dict[str, list[CaseOutcome]] = {
        t: [o for o in outcomes if o.target == t] for t in POLICIES
    }
    policies = {}
    for t in POLICIES:
        b = summ[t]["overall"]
        policies[t] = {
            **b,
            "cost_per_1k_requests_usd": 1000 * (b["cost_per_request_usd"] or 0),
            "model_distribution": {k: v / b["n"] for k, v in b["model_distribution"].items()},
        }
    base = cols(by_target["routeforge-premium"])
    comparisons = {}
    for t in POLICIES[1:]:
        c = cols(by_target[t])
        comparisons[f"{t} vs routeforge-premium"] = {
            "cost_reduction_pct": paired_bootstrap(base, c, cost_reduction_pct, resamples=4000),
            "quality_retention_pct": paired_bootstrap(
                base, c, quality_retention_pct, resamples=4000
            ),
            "quality_diff_pp": paired_bootstrap(base, c, quality_diff_pp, resamples=4000),
        }
    agreement = None
    if offline:
        key_map = {
            "routeforge-premium": next(k for k in offline["policies"] if "(premium)" in k),
            "routeforge-economy": next(k for k in offline["policies"] if "(economy)" in k),
            "routeforge-random": "random",
            "routeforge-heuristic": "heuristic",
            "routeforge-learned": next(
                k for k in offline["policies"] if k.startswith("learned (τ")
            ),
        }
        agreement = {
            t: {
                "offline_acceptable_rate": offline["policies"][k]["acceptable_rate"],
                "live_acceptable_rate": policies[t]["acceptable_rate"],
                "offline_cost_per_request": offline["policies"][k]["cost_per_request_usd"],
                "live_cost_per_request": policies[t]["cost_per_request_usd"],
            }
            for t, k in key_map.items()
        }
    lc = comparisons["routeforge-learned vs routeforge-premium"]
    summary = {
        "metadata": experiment_metadata(
            "A_routing_live",
            mode=ready.get("components", {}).get("models"),
            base_url=args.base_url,
            dataset=ds.name,
            dataset_version=ds.version,
            split="test",
            n_cases=len(cases),
            policies=POLICIES,
            quality_threshold=args.threshold,
            judge_model=args.judge_model,
            router_artifact=ready.get("components", {}).get("router_artifact"),
            gateway_estimated_cost_usd=gateway_cost,
            cache="bypassed",
            methodology="Live end-to-end: each policy answers every test prompt through the gateway; scored with the same scorers and LLM judge; latency is client-observed.",
        ),
        "status": "complete",
        "headline": [
            {
                "label": "Cost reduction vs premium (learned)",
                "value": f"{lc['cost_reduction_pct']['estimate']:.1f}%",
                "detail": f"95% CI [{lc['cost_reduction_pct']['ci_low']:.1f}, {lc['cost_reduction_pct']['ci_high']:.1f}]",
            },
            {
                "label": "Quality retention vs premium",
                "value": f"{lc['quality_retention_pct']['estimate']:.1f}%",
                "detail": f"Δ {lc['quality_diff_pp']['estimate']:+.1f} pp [{lc['quality_diff_pp']['ci_low']:+.1f}, {lc['quality_diff_pp']['ci_high']:+.1f}]",
            },
            {
                "label": "Learned acceptable rate",
                "value": f"{100 * policies['routeforge-learned']['acceptable_rate']:.1f}%",
                "detail": f"premium {100 * policies['routeforge-premium']['acceptable_rate']:.1f}%",
            },
            {
                "label": "Learned p50 latency (end-to-end)",
                "value": f"{policies['routeforge-learned']['latency_p50_ms']:.0f} ms",
                "detail": f"premium {policies['routeforge-premium']['latency_p50_ms']:.0f} ms",
            },
        ],
        "policies": policies,
        "by_category": {t: summ[t]["by_category"] for t in POLICIES},
        "comparisons": comparisons,
        "offline_live_agreement": agreement,
        "limitations": [
            "161 held-out prompts from one benchmark; single run (provider sampling is not fully deterministic).",
            "Judge-scored summaries; NQ-open answer keys are noisy (see docs/EVALUATION.md).",
            "Costs are list-price estimates from provider-reported tokens.",
        ],
    }
    write_summary(args.name, summary)
    write_csv(args.name, "per_case.csv", [o.as_dict() for o in outcomes])
    for t, b in policies.items():
        print(
            f"{t:22s} acc {100 * b['acceptable_rate']:.1f}%  $/1k {b['cost_per_1k_requests_usd']:.4f}  p50 {b['latency_p50_ms']:.0f} ms  errors {b['errors']}"
        )
    for k, c in comparisons.items():
        print(
            f"{k}: cost {c['cost_reduction_pct']['estimate']:+.1f}% [{c['cost_reduction_pct']['ci_low']:.1f},{c['cost_reduction_pct']['ci_high']:.1f}] "
            f"quality {c['quality_diff_pp']['estimate']:+.1f}pp [{c['quality_diff_pp']['ci_low']:+.1f},{c['quality_diff_pp']['ci_high']:+.1f}]"
        )
    print(f"gateway estimated cost ${gateway_cost:.4f}; ledger total ${ledger.spent:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--judge-model", default="openai/gpt-4.1")
    ap.add_argument("--offline", default="A_routing_offline_latest")
    ap.add_argument("--name", default="A_routing_live_latest", help="results directory name")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
