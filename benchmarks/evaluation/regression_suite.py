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

COLLECTION = (
    REPO_ROOT / "router_training" / "data" / "collections" / "hosted-2026-10-11" / "outputs.jsonl"
)
ARTIFACT = REPO_ROOT / "router_training" / "artifacts" / "router-hosted-2026-10-11"
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
        "seeded": "regression",
    },
    {
        "name": "learned τ=0.3 (permissive threshold)",
        "target": "routeforge-learned",
        "tau": 0.3,
        "seeded": "regression",
    },
    {"name": "learned τ=0.5", "target": "routeforge-learned", "tau": 0.5, "seeded": "regression"},
    {"name": "random routing", "target": "routeforge-random", "tau": 0.8, "seeded": "regression"},
    {
        "name": "rule-based heuristic",
        "target": "routeforge-heuristic",
        "tau": 0.8,
        "seeded": "regression",
    },
    {
        "name": "mid-tier model only (gpt-4.1-mini)",
        "target": "gpt-4.1-mini",
        "tau": 0.8,
        "seeded": "regression",
    },
]
CI_CANDIDATES = {"identical re-run", "economy model only"}


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
    settings = Settings(
        _env_file=None,
        env="test",
        log_level="WARNING",
        database_url="",
        redis_url="",
        model_config_path=REPO_ROOT / "config" / "models" / "replay-hosted.yaml",
        replay_recordings_path=COLLECTION,
        router_artifact_path=ARTIFACT,
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

    regs = [r for r in rows if r["seeded"] == "regression"]
    neutral = [r for r in rows if r["seeded"] == "neutral"]
    detected = sum(r["detected"] for r in regs)
    flagged = sum(r["flagged"] for r in regs)
    false_alarms = sum(r["detected"] for r in neutral)
    false_flags = sum(r["flagged"] for r in neutral)
    if args.ci:
        ok = all(not r["detected"] for r in neutral) and all(r["detected"] for r in regs)
        print("CI regression gate self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    summary = {
        "metadata": experiment_metadata(
            "G_regression_gate_validation",
            mode="replay (recorded hosted outputs; judge cache only)",
            dataset=ds.name,
            dataset_version=ds.version,
            split="test",
            n_cases=len(cases),
            reference=f"{REFERENCE[0]} τ={REFERENCE[1]}",
            reference_acceptable_rate=ref_acc,
            gate_config=GateConfig().__dict__,
            router_artifact=state.engine.artifact.version if state.engine.artifact else None,
        ),
        "status": "complete",
        "headline": [
            {
                "label": "Seeded regressions blocked (gate FAIL)",
                "value": f"{detected}/{len(regs)}",
                "detail": "95% CI of quality drop entirely below -2 pp",
            },
            {
                "label": "Seeded regressions flagged (FAIL or WARN)",
                "value": f"{flagged}/{len(regs)}",
                "detail": "WARN = point estimate below -2 pp, CI inconclusive",
            },
            {
                "label": "Quality false alarms on neutral changes",
                "value": f"{false_alarms}/{len(neutral)} blocked, {false_flags}/{len(neutral)} flagged",
            },
            {
                "label": "Smallest detected drop",
                "value": f"{min((r['true_quality_delta_pp'] for r in regs if r['detected']), default=float('nan')):+.1f} pp",
            },
            {
                "label": "Largest undetected drop",
                "value": f"{min((r['true_quality_delta_pp'] for r in regs if not r['detected']), default=float('nan')):+.1f} pp",
                "detail": "below the gate's statistical power at n=161",
            },
        ],
        "totals": {
            "regressions": len(regs),
            "blocked": detected,
            "flagged": flagged,
            "neutral": len(neutral),
            "false_blocks": false_alarms,
            "false_flags": false_flags,
        },
        "scenarios": rows,
        "limitations": [
            "Seeded regressions are routing/configuration changes replayed on recorded outputs; prompt or model changes would need fresh generations.",
            "With 161 test prompts, drops of a few points cannot be distinguished from noise at 95% confidence; the gate is deliberately conservative.",
        ],
    }
    write_summary("G_regression_gate_validation", summary)
    print(
        f"blocked {detected}/{len(regs)}, flagged {flagged}/{len(regs)} seeded regressions; false alarms {false_alarms}/{len(neutral)} (flags {false_flags})"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--sandbox", default="docker", choices=["docker", "subprocess"])
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
