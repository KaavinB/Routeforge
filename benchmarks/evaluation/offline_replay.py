"""Benchmarks A (offline) + F: routing-policy comparison and quality/cost Pareto frontier
by counterfactual replay.

Every held-out test prompt was answered by *every* candidate model during collection
(router_training/collect.py) and scored (router_training/score.py). A routing policy
therefore only has to choose a model per prompt; its quality, cost and latency are the
recorded outcomes of that model on that prompt. This isolates the routing decision,
makes all policies paired (same prompts, same outputs), and lets us sweep thresholds
without new API spend. The live end-to-end run (benchmarks/evaluation/live_policies.py)
validates these numbers through the real gateway.

    uv run python -m benchmarks.evaluation.offline_replay --collection hosted-2026-10-11 \
        --registry config/models/hosted.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.core.registry import ModelRegistry
from app.evaluation.datasets import Dataset, EvalCase
from app.evaluation.stats import (
    brier,
    cost_reduction_pct,
    ece,
    paired_bootstrap,
    quality_diff_pp,
    quality_retention_pct,
    wilson_interval,
)
from app.routing.artifact import RouterArtifact
from app.routing.engine import RoutingEngine
from app.routing.features import extract_features, routing_text
from benchmarks.common import (
    NEUTRAL,
    SERIES,
    TEXT_2,
    experiment_metadata,
    setup_matplotlib,
    write_csv,
    write_summary,
)
from router_training.collect import COLLECTIONS, DATASETS
from router_training.training.train import ARTIFACTS

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.99]
DEFAULT_TAU = 0.8


def load_outcomes(collection: str) -> dict[tuple[str, str], dict[str, Any]]:
    d = COLLECTIONS / collection
    outs = {
        (r["case_id"], r["model_id"]): r
        for r in map(json.loads, (d / "outputs.jsonl").read_text().splitlines())
        if not r.get("error")
    }
    res: dict[tuple[str, str], dict[str, Any]] = {}
    for r in map(json.loads, (d / "labels.jsonl").read_text().splitlines()):
        o = outs.get((r["case_id"], r["model_id"]), {})
        res[(r["case_id"], r["model_id"])] = {
            "acceptable": bool(r["acceptable"]),
            "score": float(r["score"]),
            "cost": float(o.get("cost_usd", 0.0)),
            "latency_ms": float(o.get("latency_s", 0.0)) * 1000,
        }
    return res


def run_policy(
    name: str,
    choose: Any,
    cases: list[EvalCase],
    outcomes: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        model, extra = choose(c)
        decision_us = (time.perf_counter() - t0) * 1e6
        o = outcomes[(c.id, model)]
        rows.append(
            {
                "policy": name,
                "case_id": c.id,
                "category": c.category,
                "model": model,
                "acceptable": o["acceptable"],
                "cost": o["cost"],
                "latency_ms": o["latency_ms"],
                "decision_us": decision_us,
                **extra,
            }
        )
    return rows


def engine_chooser(
    engine: RoutingEngine, strategy: str, tau: float, budget: float | None = None
) -> Any:
    def choose(c: EvalCase) -> tuple[str, dict[str, Any]]:
        d = engine.decide(
            strategy,
            c.messages,
            c.request_params(),
            quality_threshold=tau,
            latency_budget_ms=budget,
        )
        sel = next((x for x in d.candidates if x.model_id == d.selected_model), None)
        return d.selected_model, {
            "reason": d.reason,
            "claimed_ok": bool(sel and sel.meets_quality),
            "p_selected": sel.p_quality if sel else None,
            "pred_latency_ms": sel.expected_latency_ms if sel else None,
        }

    return choose


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    acc = sum(r["acceptable"] for r in rows)
    cost = sum(r["cost"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    lo, hi = wilson_interval(acc, n)
    claimed = [r for r in rows if r.get("claimed_ok")]
    dec = [r["decision_us"] for r in rows]
    return {
        "n": n,
        "acceptable_rate": acc / n,
        "acceptable_rate_ci95": [lo, hi],
        "total_cost_usd": cost,
        "cost_per_request_usd": cost / n,
        "cost_per_1k_requests_usd": 1000 * cost / n,
        "cost_per_acceptable_usd": cost / acc if acc else None,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "model_distribution": {
            k: v / n for k, v in sorted(Counter(r["model"] for r in rows).items())
        },
        "quality_violation_rate": (sum(not r["acceptable"] for r in claimed) / len(claimed))
        if claimed
        else None,
        "n_claimed_ok": len(claimed),
        "decision_us_p50": float(np.percentile(dec, 50)),
        "decision_us_p95": float(np.percentile(dec, 95)),
        "by_category": {
            cat: {
                "n": len(v),
                "acceptable_rate": sum(r["acceptable"] for r in v) / len(v),
                "cost_per_request_usd": sum(r["cost"] for r in v) / len(v),
                "models": dict(Counter(r["model"] for r in v)),
            }
            for cat, v in sorted(_group(rows, "category").items())
        },
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def cols(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "acceptable": [float(r["acceptable"]) for r in rows],
        "cost": [r["cost"] for r in rows],
        "latency_ms": [r["latency_ms"] for r in rows],
    }


def compare(base: list[dict[str, Any]], cand: list[dict[str, Any]]) -> dict[str, Any]:
    a, b = cols(base), cols(cand)
    return {
        "cost_reduction_pct": paired_bootstrap(a, b, cost_reduction_pct, resamples=4000),
        "quality_retention_pct": paired_bootstrap(a, b, quality_retention_pct, resamples=4000),
        "quality_diff_pp": paired_bootstrap(a, b, quality_diff_pp, resamples=4000),
    }


def pareto_flags(points: list[dict[str, Any]]) -> None:
    for p in points:
        p["dominated_by"] = [
            q["label"]
            for q in points
            if q is not p
            and not q.get("reference_only")
            and q["acceptable_rate"] >= p["acceptable_rate"]
            and q["cost_per_request_usd"] <= p["cost_per_request_usd"]
            and (
                q["acceptable_rate"] > p["acceptable_rate"]
                or q["cost_per_request_usd"] < p["cost_per_request_usd"]
            )
        ]
        p["dominated"] = bool(p["dominated_by"])


def calibration(
    artifact: RouterArtifact,
    cases: list[EvalCase],
    outcomes: dict[tuple[str, str], dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    probs: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        f = extract_features(c.messages, c.request_params())
        tp = artifact.task_probs(routing_text(c.messages))
        for m, p in artifact.quality_probs(f, tp, variant).items():
            probs[m].append(p)
            labels[m].append(int(outcomes[(c.id, m)]["acceptable"]))
    for m in probs:
        p, y = np.array(probs[m]), np.array(labels[m])
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        out[m] = {
            "brier": brier(p, y),
            "ece": ece(p, y),
            "base_rate": float(y.mean()),
            "mean_pred": float(p.mean()),
            "bins": [
                {
                    "lo": float(bins[i]),
                    "hi": float(bins[i + 1]),
                    "n": int((idx == i).sum()),
                    "mean_pred": float(p[idx == i].mean()) if (idx == i).any() else None,
                    "frac_pos": float(y[idx == i].mean()) if (idx == i).any() else None,
                }
                for i in range(10)
            ],
        }
    return out


def charts(name: str, summary: dict[str, Any], models: list[str]) -> list[str]:
    plt = setup_matplotlib()
    out_dir = Path(__file__).resolve().parents[1] / "results" / name / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    model_color = {m: SERIES[i] for i, m in enumerate(models)}

    # 1. Pareto frontier
    fig, ax = plt.subplots(figsize=(8, 5.2))
    pts = summary["pareto"]["points"]
    sweep = sorted(
        [p for p in pts if p["kind"] == "learned_sweep"], key=lambda p: p["cost_per_request_usd"]
    )
    ax.plot(
        [1000 * p["cost_per_request_usd"] for p in sweep],
        [100 * p["acceptable_rate"] for p in sweep],
        "-o",
        color=SERIES[0],
        markersize=5,
        label="learned router (threshold sweep)",
        zorder=3,
    )
    for p in sweep:
        if p["threshold"] in (0.5, 0.8, 0.9, 0.95, 0.99):
            ax.annotate(
                f"τ={p['threshold']}",
                (1000 * p["cost_per_request_usd"], 100 * p["acceptable_rate"]),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=8,
                color=TEXT_2,
            )
    for p in pts:
        if p["kind"] in ("fixed", "heuristic", "random", "oracle"):
            style = {
                "fixed": {"color": NEUTRAL, "marker": "s"},
                "heuristic": {"color": SERIES[1], "marker": "D"},
                "random": {"color": SERIES[2], "marker": "^"},
                "oracle": {"color": TEXT_2, "marker": "*", "s": 110},
            }[p["kind"]]
            ax.scatter(
                1000 * p["cost_per_request_usd"],
                100 * p["acceptable_rate"],
                s=style.pop("s", 55),
                edgecolor="white",
                linewidth=1.5,
                zorder=4,
                **style,
            )
            offset = {"random": (-12, -16), "fixed": (7, -12)}.get(p["kind"], (7, 4))
            ax.annotate(
                p["label"],
                (1000 * p["cost_per_request_usd"], 100 * p["acceptable_rate"]),
                textcoords="offset points",
                xytext=offset,
                ha="right" if p["kind"] == "random" else "left",
                fontsize=8,
                color=TEXT_2,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost per 1,000 requests (USD, log scale)")
    ax.set_ylabel("Acceptable answers (%)")
    ax.set_title(f"Quality vs. cost on held-out prompts ({summary['metadata']['collection']})")
    ax.legend(loc="lower right")
    f = out_dir / "pareto.png"
    fig.savefig(f)
    plt.close(fig)
    files.append(f.name)

    # 2. Acceptable rate with CI per policy
    pol = summary["policies"]
    names = list(pol)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.2))
    rates = [100 * pol[n]["acceptable_rate"] for n in names]
    err = [
        [100 * (pol[n]["acceptable_rate"] - pol[n]["acceptable_rate_ci95"][0]) for n in names],
        [100 * (pol[n]["acceptable_rate_ci95"][1] - pol[n]["acceptable_rate"]) for n in names],
    ]
    y = np.arange(len(names))
    ax.barh(
        y,
        rates,
        height=0.6,
        color=SERIES[0],
        xerr=err,
        error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 2},
    )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Acceptable answers (%) with 95% Wilson CI")
    ax.set_xlim(min(rates) - 15, 100)
    ax.set_title("Quality by routing policy")
    for i, r in enumerate(rates):
        ax.text(r + 0.5, i, f"{r:.1f}", va="center", fontsize=8, color=TEXT_2)
    f = out_dir / "policy_quality.png"
    fig.savefig(f)
    plt.close(fig)
    files.append(f.name)

    # 3. Model mix per policy (stacked)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.4))
    left = np.zeros(len(names))
    for m in models:
        share = np.array([100 * pol[n]["model_distribution"].get(m, 0.0) for n in names])
        ax.barh(
            y,
            share,
            left=left,
            height=0.6,
            color=model_color[m],
            label=m,
            edgecolor="white",
            linewidth=2,
        )
        left += share
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Share of requests routed to each model (%)")
    ax.set_title("Model mix by routing policy")
    ax.legend(ncol=min(4, len(models)), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", visible=False)
    f = out_dir / "model_mix.png"
    fig.savefig(f)
    plt.close(fig)
    files.append(f.name)

    # 4. Reliability diagrams (small multiples)
    cal = summary["calibration"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models, strict=False):
        bins = [b for b in cal[m]["bins"] if b["n"] > 0]
        ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, linestyle="--")
        ax.plot(
            [b["mean_pred"] for b in bins],
            [b["frac_pos"] for b in bins],
            "-o",
            color=model_color[m],
            markersize=5,
        )
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("Predicted P(acceptable)")
        ax.text(
            0.04,
            0.9,
            f"ECE {cal[m]['ece']:.3f}\nBrier {cal[m]['brier']:.3f}",
            fontsize=8,
            color=TEXT_2,
            transform=ax.transAxes,
            va="top",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("Observed acceptable rate")
    fig.suptitle(
        "Router calibration on the held-out test split",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    f = out_dir / "calibration.png"
    fig.savefig(f)
    plt.close(fig)
    files.append(f.name)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--latency-budget-ms", type=float, default=None, help="for the latency-ablation experiment"
    )
    args = ap.parse_args()

    registry = ModelRegistry.from_yaml(args.registry)
    artifact_dir = args.artifact or ARTIFACTS / f"router-{args.collection}"
    artifact = RouterArtifact.load(artifact_dir)
    ds = Dataset(DATASETS / args.dataset)
    outcomes = load_outcomes(args.collection)
    models = [m.id for m in registry.candidates()]
    cases = [c for c in ds.select(args.split) if all((c.id, m) in outcomes for m in models)]
    engine = RoutingEngine(registry, artifact=artifact, seed=1234)
    name_suffix = registry.mode

    policies: dict[str, list[dict[str, Any]]] = {}
    for m in models:
        tag = (
            " (premium)"
            if m == registry.premium_model
            else (" (economy)" if m == registry.economy_model else "")
        )
        policies[f"fixed: {m}{tag}"] = run_policy(m, lambda c, m=m: (m, {}), cases, outcomes)
    policies["random"] = run_policy(
        "random", engine_chooser(engine, "random", DEFAULT_TAU), cases, outcomes
    )
    policies["heuristic"] = run_policy(
        "heuristic", engine_chooser(engine, "heuristic", DEFAULT_TAU), cases, outcomes
    )
    for strat in (
        "learned",
        "learned-full",
        "learned-no-task",
        "learned-task-only",
        "learned-no-cost",
    ):
        if strat in engine.strategies:
            policies[f"{strat} (τ={DEFAULT_TAU})"] = run_policy(
                strat, engine_chooser(engine, strat, DEFAULT_TAU), cases, outcomes
            )

    def oracle(c: EvalCase) -> tuple[str, dict[str, Any]]:
        ok = [m for m in models if outcomes[(c.id, m)]["acceptable"]]
        pool = ok or models
        return min(pool, key=lambda m: outcomes[(c.id, m)]["cost"]), {}

    policies["oracle (upper bound)"] = run_policy("oracle", oracle, cases, outcomes)
    summary_pol = {k: block(v) for k, v in policies.items()}

    # random: expected value over 50 seeds (a single seed is one draw)
    rand_runs = []
    for s in range(50):
        eng = RoutingEngine(registry, artifact=artifact, seed=s)
        rand_runs.append(
            block(run_policy("random", engine_chooser(eng, "random", DEFAULT_TAU), cases, outcomes))
        )
    summary_pol["random"]["over_50_seeds"] = {
        "acceptable_rate_mean": float(np.mean([r["acceptable_rate"] for r in rand_runs])),
        "acceptable_rate_sd": float(np.std([r["acceptable_rate"] for r in rand_runs])),
        "cost_per_request_mean": float(np.mean([r["cost_per_request_usd"] for r in rand_runs])),
    }

    premium_key = next(k for k in policies if "(premium)" in k)
    best_single = max(
        (k for k in policies if k.startswith("fixed")),
        key=lambda k: (summary_pol[k]["acceptable_rate"], -summary_pol[k]["cost_per_request_usd"]),
    )
    learned_key = f"learned (τ={DEFAULT_TAU})"
    comparisons = {
        f"{learned_key} vs {premium_key}": compare(policies[premium_key], policies[learned_key]),
        f"heuristic vs {premium_key}": compare(policies[premium_key], policies["heuristic"]),
    }
    for k in policies:
        if k.startswith("fixed") and k != premium_key:
            comparisons[f"{learned_key} vs {k}"] = compare(policies[k], policies[learned_key])

    # Pareto sweep (learned) + reference points
    points: list[dict[str, Any]] = []
    sweep_rows: dict[float, list[dict[str, Any]]] = {}
    for tau in THRESHOLDS:
        rows = run_policy("learned", engine_chooser(engine, "learned", tau), cases, outcomes)
        sweep_rows[tau] = rows
        b = block(rows)
        points.append(
            {
                "label": f"learned τ={tau}",
                "kind": "learned_sweep",
                "threshold": tau,
                **{
                    k: b[k]
                    for k in (
                        "acceptable_rate",
                        "cost_per_request_usd",
                        "latency_p50_ms",
                        "quality_violation_rate",
                        "model_distribution",
                    )
                },
            }
        )
    for k, b in summary_pol.items():
        kind = "fixed" if k.startswith("fixed") else k.split(" ")[0]
        if kind in ("fixed", "heuristic", "random", "oracle"):
            points.append(
                {
                    "label": k.replace("fixed: ", ""),
                    "kind": kind,
                    "reference_only": kind == "oracle",
                    "acceptable_rate": b["acceptable_rate"],
                    "cost_per_request_usd": b["cost_per_request_usd"],
                    "latency_p50_ms": b["latency_p50_ms"],
                }
            )
    pareto_flags(points)
    # Does learned routing beat the best single model at matched quality?
    best = summary_pol[best_single]
    matched = [
        p
        for p in points
        if p["kind"] == "learned_sweep" and p["acceptable_rate"] >= best["acceptable_rate"]
    ]
    cheapest_matched = min(matched, key=lambda p: p["cost_per_request_usd"]) if matched else None

    # Latency-budget ablation
    latency_exp = None
    if args.latency_budget_ms:
        lb = args.latency_budget_ms
        with_b = run_policy(
            "learned", engine_chooser(engine, "learned", DEFAULT_TAU, lb), cases, outcomes
        )
        without = run_policy(
            "learned-no-latency",
            engine_chooser(engine, "learned-no-latency", DEFAULT_TAU, lb),
            cases,
            outcomes,
        )
        latency_exp = {
            "budget_ms": lb,
            "with_latency_model": {
                **block(with_b),
                "over_budget_rate": float(np.mean([r["latency_ms"] > lb for r in with_b])),
            },
            "without_latency_model": {
                **block(without),
                "over_budget_rate": float(np.mean([r["latency_ms"] > lb for r in without])),
            },
        }

    meta = experiment_metadata(
        f"A_F_offline_replay_{name_suffix}",
        collection=args.collection,
        mode=registry.mode,
        dataset=ds.name,
        dataset_version=ds.version,
        dataset_sha256=ds.manifest.sha256,
        split=args.split,
        n_cases=len(cases),
        registry_version=registry.version,
        pricing=registry.data.pricing.model_dump(),
        router_artifact=artifact.version,
        router_selected_feature_sets=artifact.selected_feature_sets,
        default_threshold=DEFAULT_TAU,
        methodology="Counterfactual replay: each policy picks a model per prompt; outcome = that model's recorded, "
        "scored output for that prompt. Latency = recorded provider wall-clock latency of the chosen model "
        "(excludes gateway overhead, measured separately). Costs are token usage x list price (estimates).",
    )
    summary = {
        "metadata": meta,
        "status": "complete",
        "policies": summary_pol,
        "comparisons": comparisons,
        "best_single_model": best_single,
        "learned_vs_best_single_at_matched_quality": {
            "best_single": best_single,
            "best_single_acceptable_rate": best["acceptable_rate"],
            "best_single_cost_per_request": best["cost_per_request_usd"],
            "cheapest_learned_point_matching_quality": cheapest_matched,
            "cost_reduction_pct": (
                100 * (1 - cheapest_matched["cost_per_request_usd"] / best["cost_per_request_usd"])
            )
            if cheapest_matched
            else None,
        },
        "pareto": {"points": points, "thresholds": THRESHOLDS},
        "latency_budget_experiment": latency_exp,
        "calibration": calibration(artifact, cases, outcomes, artifact.default_variant),
        "limitations": [
            "Single dataset version (840 prompts, 161 held-out test prompts); category priors dominate the routing signal.",
            "Task categories are trivially identifiable from their fixed instructions; real traffic is more heterogeneous.",
            "Summaries graded by an LLM judge (gpt-4.1); judge is not ground truth (see docs/EVALUATION.md).",
            "Costs are list-price estimates, not billing data. Latencies reflect one region/time window.",
        ],
    }
    out_name = f"A_routing_offline_{name_suffix}"
    summary["charts"] = charts(out_name, summary, models)
    write_summary(out_name, summary)
    write_csv(out_name, "per_case.csv", [r for rows in policies.values() for r in rows])
    write_csv(
        out_name,
        "pareto_points.csv",
        [{k: v for k, v in p.items() if k != "model_distribution"} for p in points],
    )

    print(
        f"\n{'policy':42s} {'acc%':>6s} {'ci95':>13s} {'$/1k req':>9s} {'$/acc':>9s} {'p50ms':>7s} {'viol':>6s}"
    )
    for k, b in summary_pol.items():
        v = b["quality_violation_rate"]
        print(
            f"{k:42s} {100 * b['acceptable_rate']:6.1f} [{100 * b['acceptable_rate_ci95'][0]:5.1f},{100 * b['acceptable_rate_ci95'][1]:5.1f}] "
            f"{b['cost_per_1k_requests_usd']:9.4f} {b['cost_per_acceptable_usd'] or 0:9.6f} {b['latency_p50_ms']:7.0f} {'' if v is None else f'{100 * v:5.1f}%'}"
        )
    for k, c in comparisons.items():
        print(
            f"{k}: cost reduction {c['cost_reduction_pct']['estimate']:+.1f}% [{c['cost_reduction_pct']['ci_low']:.1f},{c['cost_reduction_pct']['ci_high']:.1f}], "
            f"quality {c['quality_diff_pp']['estimate']:+.1f}pp [{c['quality_diff_pp']['ci_low']:+.1f},{c['quality_diff_pp']['ci_high']:+.1f}]"
        )
    print(
        "matched-quality vs best single:",
        json.dumps(summary["learned_vs_best_single_at_matched_quality"], default=str)[:400],
    )
    print("dominated points:", [p["label"] for p in points if p["dominated"]])


if __name__ == "__main__":
    main()
