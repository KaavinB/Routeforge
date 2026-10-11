"""Regression gates: compare a candidate evaluation run against a reference run on the
same cases and decide pass/fail with uncertainty in mind.

A gate fails only when the *confidence interval* excludes the tolerated region, so small
noisy differences on a few hundred prompts do not block (or wrongly promote) a change:

* quality:      fail if CI upper bound of (cand - ref) acceptable rate < -max_quality_drop_pp
* latency:      warn if p50 latency rises by more than max_latency_increase_pct (CI-aware)
* cost:         warn if cost per acceptable answer rises by more than max_cost_increase_pct
* structured:   fail if JSON schema-invalid rate exceeds max_invalid_json_rate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.evaluation.stats import paired_bootstrap, quality_diff_pp


@dataclass(frozen=True)
class GateConfig:
    max_quality_drop_pp: float = 2.0
    max_latency_increase_pct: float = 25.0
    max_cost_increase_pct: float = 20.0
    max_invalid_json_rate: float = 0.05
    resamples: int = 4000


@dataclass
class GateResult:
    name: str
    status: str  # pass | fail | warn | skip
    detail: dict[str, Any] = field(default_factory=dict)


def _align(
    ref: list[dict[str, Any]], cand: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rmap = {r["case_id"]: r for r in ref}
    cmap = {c["case_id"]: c for c in cand}
    common = sorted(set(rmap) & set(cmap))
    return [rmap[k] for k in common], [cmap[k] for k in common]


def _cols(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "acceptable": [float(r["acceptable"]) for r in rows],
        "cost": [float(r.get("cost_usd") or 0.0) for r in rows],
        "latency_ms": [float(r.get("latency_ms") or 0.0) for r in rows],
    }


def _cost_per_acceptable_change(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    ca = a["cost"].sum() / max(a["acceptable"].sum(), 1)
    cb = b["cost"].sum() / max(b["acceptable"].sum(), 1)
    return float(100 * (cb - ca) / ca) if ca > 0 else math.nan


def _latency_change(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    pa = np.percentile(a["latency_ms"], 50)
    return float(100 * (np.percentile(b["latency_ms"], 50) - pa) / pa) if pa > 0 else math.nan


def evaluate_gates(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    config: GateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GateConfig()
    ref, cand = _align(reference, candidate)
    gates: list[GateResult] = []
    if len(ref) < 20:
        return {
            "passed": False,
            "n_aligned": len(ref),
            "gates": [
                GateResult("alignment", "fail", {"reason": "fewer than 20 aligned cases"}).__dict__
            ],
        }
    a, b = _cols(ref), _cols(cand)

    q = paired_bootstrap(a, b, quality_diff_pp, resamples=cfg.resamples)
    q_status = "fail" if q["ci_high"] < -cfg.max_quality_drop_pp else "pass"
    if q_status == "pass" and q["estimate"] < -cfg.max_quality_drop_pp:
        q_status = "warn"  # point estimate past threshold but CI inconclusive
    gates.append(
        GateResult("quality_drop", q_status, {**q, "threshold_pp": -cfg.max_quality_drop_pp})
    )

    if any(x > 0 for x in a["latency_ms"]) and any(x > 0 for x in b["latency_ms"]):
        lat = paired_bootstrap(a, b, _latency_change, resamples=cfg.resamples)
        lat_status = "warn" if lat["ci_low"] > cfg.max_latency_increase_pct else "pass"
        gates.append(
            GateResult(
                "latency_increase",
                lat_status,
                {**lat, "threshold_pct": cfg.max_latency_increase_pct},
            )
        )
    else:
        gates.append(GateResult("latency_increase", "skip", {"reason": "no latency data"}))

    if sum(a["cost"]) > 0:
        cost = paired_bootstrap(a, b, _cost_per_acceptable_change, resamples=cfg.resamples)
        cost_status = "warn" if cost["ci_low"] > cfg.max_cost_increase_pct else "pass"
        gates.append(
            GateResult(
                "cost_per_acceptable_increase",
                cost_status,
                {**cost, "threshold_pct": cfg.max_cost_increase_pct},
            )
        )
    else:
        gates.append(GateResult("cost_per_acceptable_increase", "skip", {"reason": "no cost data"}))

    json_rows = [c for c in cand if c.get("category") == "json"]
    if json_rows:
        invalid = sum(
            1 for c in json_rows if not (c.get("score_details") or {}).get("schema_valid", False)
        )
        rate = invalid / len(json_rows)
        gates.append(
            GateResult(
                "structured_output_validity",
                "fail" if rate > cfg.max_invalid_json_rate else "pass",
                {"invalid_rate": rate, "n": len(json_rows), "threshold": cfg.max_invalid_json_rate},
            )
        )
    passed = all(g.status != "fail" for g in gates)
    return {"passed": passed, "n_aligned": len(ref), "gates": [g.__dict__ for g in gates]}
