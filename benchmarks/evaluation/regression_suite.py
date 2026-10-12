"""Validate the CI quality gate with seeded regressions (deterministic, zero cost).

The gateway runs in-process in *replay* mode: every model answer is the real recorded
output from the hosted collection, and summary verdicts come from the judge cache, so
runs are deterministic and make no API calls. A reference configuration (learned router,
τ=0.8) is evaluated on the held-out test split, then each candidate configuration is
evaluated and compared with ``app.evaluation.regression.evaluate_gates``.

Candidates are labelled by construction:
  * seeded regressions — configurations known to be worse (economy-only, random,
    rule-based heuristic, learned router with a permissive threshold, ...)
  * neutral changes   — an identical re-run and near-equivalent ablations
and we report detection rate (regressions whose quality gate failed) and false-alarm
rate (neutral changes whose quality gate failed), alongside each candidate's true
quality delta on the test split. With 161 prompts the gate only fails when the 95% CI of
the paired quality difference lies entirely below -2 pp, so small regressions are
expected to go undetected — that is the statistical power of this suite, reported as is.

    uv run python -m benchmarks.evaluation.regression_suite          # full suite
    uv run python -m benchmarks.evaluation.regression_suite --ci     # CI subset, exit code
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from typing import Any

from app.core.config import REPO_ROOT, Settings
from app.evaluation.datasets import Dataset
from app.evaluation.judge import SummaryJudge
from app.evaluation.regression import GateConfig, evaluate_gates
from app.evaluation.runner import InProcessClient, TargetOptions, run_eval
from app.evaluation.scoring import Scorer
from app.state import build_state
from benchmarks.common import experiment_metadata, write_summary
from router_training.collect import DATASETS
from router_training.score import JUDGE_CACHE

POOLS = {
    # pool -> (collection id, replay registry file)
    "latest": ("latest-2026-10-11", "replay-latest.yaml"),
    "hosted": ("hosted-2026-10-11", "replay-hosted.yaml"),
}
REFERENCE = ("routeforge-learned", 0.8)

CANDIDATES: list[dict[str, Any]] = [
    {"name": "identical re-run", "target": "routeforge-learned", "tau": 0.8, "seeded": "neutral"},
    {"name": "learned τ=0.85", "target": "routeforge-learned", "tau": 0.85, "seeded": "neutral"},
    {
        "name": "learned-full ablation",
        "target": "routeforge-learned-full",
        "tau": 0.8,
        "seeded": "neutral",
    },
    {
        "name": "learned-no-cost (cost regression only)",
        "target": "routeforge-learned-no-cost",
        "tau": 0.8,
        "seeded": "neutral",
    },
    {
        "name": "economy model only",
        "target": "routeforge-economy",
        "tau": 0.8,
        "seeded": "expected_regression",
    },
    {
        "name": "learned τ=0.3 (permissive threshold)",
        "target": "routeforge-learned",
        "tau": 0.3,
        "seeded": "expected_regression",
    },
    {
        "name": "learned τ=0.5",
        "target": "routeforge-learned",
        "tau": 0.5,
        "seeded": "expected_regression",
    },
    {
        "name": "random routing",
        "target": "routeforge-random",
        "tau": 0.8,
        "seeded": "expected_regression",
    },
    {
        "name": "rule-based heuristic",
        "target": "routeforge-heuristic",
        "tau": 0.8,
        "seeded": "expected_regression",
    },
    {
        "name": "mid-tier model only",
        "target": "tier:2",
        "tau": 0.8,
        "seeded": "expected_regression",
    },
]
CI_CANDIDATES = {"identical re-run"}


async def evaluate(
    client: InProcessClient, scorer: Scorer, cases: list[Any], target: str, tau: float
) -> list[dict[str, Any]]:
    outs = await run_eval(
        cases, [target], client, scorer, opts=TargetOptions(quality_threshold=tau), concurrency=16
    )
    errors = [o for o in outs if o.error]
    if errors:
        raise RuntimeError(f"{len(errors)} replay errors for {target}: {errors[0].error}")
    return [o.as_dict() for o in outs]


async def main_async(args: argparse.Namespace) -> int:
    collection, registry_file = POOLS[args.pool]
    collections_dir = REPO_ROOT / "router_training" / "data" / "collections"
    settings = Settings(
        _env_file=None,
        env="test",
        log_level="WARNING",
        database_url="",
        redis_url="",
        model_config_path=REPO_ROOT / "config" / "models" / registry_file,
        replay_recordings_path=collections_dir / collection / "outputs.jsonl",
        router_artifact_path=REPO_ROOT / "router_training" / "artifacts" / f"router-{collection}",
        default_strategy="learned",
        cache_enabled=False,
        otel_enabled=False,
        langfuse_enabled=False,
        request_log_enabled=False,
    )
    state = build_state(settings)
    client = InProcessClient(state.gateway)
    scorer = Scorer(
        judge=SummaryJudge(cache_path=JUDGE_CACHE, cache_only=True), sandbox_backend=args.sandbox
    )
    ds = Dataset(DATASETS / "routeforge-bench-v1")
    cases = ds.select("test")

    reference = await evaluate(client, scorer, cases, *REFERENCE)
    ref_acc = sum(r["acceptable"] for r in reference) / len(reference)
    rows = []
    for cand in CANDIDATES:
        if args.ci and cand["name"] not in CI_CANDIDATES:
            continue
        if cand["target"].startswith("tier:"):
            tier = int(cand["target"].split(":")[1])
            cand = {
                **cand,
                "target": next(m.id for m in state.registry.candidates() if m.tier == tier),
            }
            cand["name"] = f"{cand['name']} ({cand['target']})"
        res = await evaluate(client, scorer, cases, cand["target"], cand["tau"])
        report = evaluate_gates(reference, res, GateConfig())
        gates = {g["name"]: g for g in report["gates"]}
        acc = sum(r["acceptable"] for r in res) / len(res)
        q = gates["quality_drop"]["detail"]
        rows.append(
            {
                "candidate": cand["name"],
                "target": cand["target"],
                "threshold": cand["tau"],
                "seeded": cand["seeded"],
                "acceptable_rate": acc,
                "true_quality_delta_pp": 100 * (acc - ref_acc),
                "quality_delta_ci_low_pp": q.get("ci_low"),
                "quality_delta_ci_high_pp": q.get("ci_high"),
                "quality_gate": gates["quality_drop"]["status"],
                "cost_gate": gates["cost_per_acceptable_increase"]["status"],
                "latency_gate": gates["latency_increase"]["status"],
                "json_gate": gates.get("structured_output_validity", {}).get("status"),
                "gate_passed": report["passed"],
                "detected": gates["quality_drop"]["status"] == "fail",
                "flagged": gates["quality_drop"]["status"] in ("fail", "warn"),
                "fallback_summary_scoring": sum(
                    1
                    for r in res
                    if (r.get("score_details") or {}).get("method") == "rouge_fallback_cache_miss"
                ),
            }
        )
        r = rows[-1]
        print(
            f"{r['candidate']:40s} {r['seeded']:10s} Δ={r['true_quality_delta_pp']:+5.1f}pp CI[{r['quality_delta_ci_low_pp']:+.1f},{r['quality_delta_ci_high_pp']:+.1f}] "
            f"quality_gate={r['quality_gate']:4s} cost_gate={r['cost_gate']}",
            flush=True,
        )

    # Part 2: power analysis with synthetic regressions of known size. Each trial flips
    # exactly k% of the reference's acceptable answers to unacceptable (a regression of
    # exactly k pp on this test set) and runs the quality gate.
    power = power_curve(reference, trials=args.trials, seed=7)
    power_churn = power_curve(reference, trials=args.trials, seed=8, churn=0.04)
    for row in power + power_churn:
        print(
            f"churn {row['churn']:.2f} drop {row['drop_pp']:4.1f} pp: FAIL {100 * row['fail_rate']:5.1f}%  FAIL-or-WARN {100 * row['flag_rate']:5.1f}%",
            flush=True,
        )
    neutral = [r for r in rows if r["seeded"] == "neutral"]
    if args.ci:
        drop10 = next(r for r in power if r["drop_pp"] >= 10)
        ok = (
            all(not r["detected"] for r in neutral)
            and drop10["fail_rate"] >= 0.95
            and power[0]["fail_rate"] <= 0.05
        )
        print("CI regression gate self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    at = {r["drop_pp"]: r for r in power}
    atc = {r["drop_pp"]: r for r in power_churn}
    summary = {
        "metadata": experiment_metadata(
            "G_regression_gate_validation",
            mode=f"replay of {collection} (recorded outputs; judge cache only)",
            pool=args.pool,
            dataset=ds.name,
            dataset_version=ds.version,
            split="test",
            n_cases=len(cases),
            reference=f"{REFERENCE[0]} τ={REFERENCE[1]}",
            reference_acceptable_rate=ref_acc,
            gate_config=GateConfig().__dict__,
            power_trials_per_size=args.trials,
            router_artifact=state.engine.artifact.version if state.engine.artifact else None,
        ),
        "status": "complete",
        "headline": [
            {
                "label": "Synthetic 5 pp regressions blocked",
                "value": f"{100 * at[5.0]['fail_rate']:.0f}%",
                "detail": f"flagged (FAIL or WARN) {100 * at[5.0]['flag_rate']:.0f}%; {args.trials} trials",
            },
            {
                "label": "Synthetic 10 pp regressions blocked",
                "value": f"{100 * at[10.0]['fail_rate']:.0f}%",
            },
            {
                "label": "False FAIL rate with no regression",
                "value": f"{100 * at[0.0]['fail_rate']:.1f}%",
                "detail": f"false FAIL-or-WARN {100 * at[0.0]['flag_rate']:.1f}%",
            },
            {
                "label": "With 4% answer churn: 5 pp / 10 pp regressions blocked",
                "value": f"{100 * atc[5.0]['fail_rate']:.0f}% / {100 * atc[10.0]['fail_rate']:.0f}%",
                "detail": f"false FAIL at 0 pp {100 * atc[0.0]['fail_rate']:.1f}%; "
                f"5 pp flagged {100 * atc[5.0]['flag_rate']:.0f}%",
            },
            {
                "label": "Neutral configuration changes blocked",
                "value": f"{sum(r['detected'] for r in neutral)}/{len(neutral)}",
            },
        ],
        "power_curve": power,
        "power_curve_with_churn": power_churn,
        "scenarios": rows,
        "limitations": [
            "Synthetic regressions flip outcomes uniformly at random; real regressions may concentrate in categories.",
            "Configuration scenarios are replayed on recorded outputs; prompt or model changes would need fresh generations.",
            "With n=161 the gate only blocks when the 95% CI of the paired drop lies below -2 pp: it trades sensitivity for few false alarms.",
        ],
    }
    write_summary(f"G_regression_gate_validation_{args.pool}", summary)
    return 0


def power_curve(
    reference: list[dict[str, Any]], *, trials: int, seed: int, churn: float = 0.0
) -> list[dict[str, Any]]:
    """Gate detection rate vs. regression size. ``churn`` additionally swaps that fraction
    of cases in *both* directions (good->bad and bad->good) without changing the net
    effect, mimicking a real change where some answers improve while others regress."""
    rng = random.Random(seed)
    n = len(reference)
    acc_idx = [i for i, r in enumerate(reference) if r["acceptable"]]
    bad_idx = [i for i, r in enumerate(reference) if not r["acceptable"]]
    m = min(round(churn * n), len(bad_idx))
    out = []
    for drop_pp in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        k = round(drop_pp / 100 * n)
        fails = flags = 0
        for _ in range(trials):
            to_bad = set(rng.sample(acc_idx, k + m))
            to_good = set(rng.sample(bad_idx, m))
            cand = [
                {**r, "acceptable": False}
                if i in to_bad
                else ({**r, "acceptable": True} if i in to_good else r)
                for i, r in enumerate(reference)
            ]
            status = evaluate_gates(reference, cand, GateConfig(resamples=1000))["gates"][0][
                "status"
            ]
            fails += status == "fail"
            flags += status in ("fail", "warn")
        out.append(
            {
                "drop_pp": drop_pp,
                "flipped_to_bad": k + m,
                "flipped_to_good": m,
                "churn": churn,
                "trials": trials,
                "fail_rate": fails / trials,
                "flag_rate": flags / trials,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--pool", default="latest", choices=sorted(POOLS))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--sandbox", default="docker", choices=["docker", "subprocess"])
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
