"""Routing strategies.

Baselines (fixed premium, fixed economy, seeded random), a documented rule-based
heuristic, and the learned quality-aware router with its ablations.
"""

from __future__ import annotations

import random

from app.core.registry import ModelSpec
from app.routing.artifact import RouterArtifact
from app.routing.cost import CostModel
from app.routing.features import LONG_CONTEXT_TOKENS
from app.routing.latency import LatencyModel
from app.routing.types import (
    CandidateScore,
    NoEligibleModelError,
    RoutingContext,
    RoutingDecision,
    tier_fallbacks,
)


def _estimates(
    ctx: RoutingContext, cost_model: CostModel, latency_model: LatencyModel
) -> dict[str, CandidateScore]:
    """Cost/latency estimates for every candidate (shared by all strategies so the
    dashboard can show the same comparison regardless of the policy in use)."""
    max_tokens = ctx.params.get("max_tokens")
    scores: dict[str, CandidateScore] = {}
    for spec in ctx.candidates:
        cost, out = cost_model.expected_cost(
            spec, ctx.features.input_tokens_est, ctx.task_probs, max_tokens
        )
        scores[spec.id] = CandidateScore(
            model_id=spec.id,
            eligible=spec.id not in ctx.ineligible,
            ineligible_reason=ctx.ineligible.get(spec.id),
            expected_output_tokens=out,
            expected_cost_usd=cost,
            expected_latency_ms=latency_model.predict_ms(spec, ctx.features.input_tokens_est, out),
        )
    return scores


def _require_eligible(ctx: RoutingContext) -> list[ModelSpec]:
    eligible = ctx.eligible
    if not eligible:
        raise NoEligibleModelError(
            "no eligible model: " + ", ".join(f"{k}={v}" for k, v in ctx.ineligible.items())
        )
    return eligible


class _EstimatingStrategy:
    def __init__(self, cost_model: CostModel, latency_model: LatencyModel) -> None:
        self.cost_model = cost_model
        self.latency_model = latency_model


class FixedModelStrategy(_EstimatingStrategy):
    """Always route to one configured model; fall back by tier proximity if it is unavailable."""

    def __init__(
        self, name: str, model_id: str, cost_model: CostModel, latency_model: LatencyModel
    ) -> None:
        super().__init__(cost_model, latency_model)
        self.name = name
        self.model_id = model_id

    def decide(self, ctx: RoutingContext) -> RoutingDecision:
        eligible = _require_eligible(ctx)
        scores = _estimates(ctx, self.cost_model, self.latency_model)
        target = next((m for m in eligible if m.id == self.model_id), None)
        reason = "fixed_target"
        if target is None:
            # Target unavailable (e.g. circuit open): degrade to the closest-tier model.
            pinned = next(m for m in ctx.candidates if m.id == self.model_id)
            ordered = tier_fallbacks(pinned, eligible)
            target = next(m for m in eligible if m.id == ordered[0])
            reason = f"fixed_target_unavailable:{ctx.ineligible.get(self.model_id, 'unknown')}"
        return RoutingDecision(
            strategy=self.name,
            selected_model=target.id,
            fallback_chain=tier_fallbacks(target, eligible),
            reason=reason,
            candidates=list(scores.values()),
            predicted_task=ctx.predicted_task,
        )


class RandomStrategy(_EstimatingStrategy):
    """Uniformly random eligible model. Seeded per request fingerprint so the same request
    always maps to the same model for a given seed, independent of arrival order."""

    name = "random"

    def __init__(self, seed: int, cost_model: CostModel, latency_model: LatencyModel) -> None:
        super().__init__(cost_model, latency_model)
        self.seed = seed

    def decide(self, ctx: RoutingContext) -> RoutingDecision:
        eligible = _require_eligible(ctx)
        rng = random.Random(f"{self.seed}:{ctx.fingerprint}")
        target = rng.choice(eligible)
        rest = [m.id for m in eligible if m.id != target.id]
        rng.shuffle(rest)
        return RoutingDecision(
            strategy=self.name,
            selected_model=target.id,
            fallback_chain=rest,
            reason="uniform_random",
            candidates=list(_estimates(ctx, self.cost_model, self.latency_model).values()),
            predicted_task=ctx.predicted_task,
        )


# Heuristic rules: predicted task -> capability level (0 = economy ... 3 = premium).
# Levels are mapped onto however many tiers the registry provides.
HEURISTIC_TASK_LEVEL: dict[str, int] = {
    "classification": 0,
    "factual_qa": 1,
    "extraction": 1,
    "summarization": 2,
    "json": 2,
    "reasoning": 3,
    "code": 3,
}


class HeuristicStrategy(_EstimatingStrategy):
    """Hand-written rules over request features (documented in docs/ARCHITECTURE.md):

    1. Predict the task with keyword/structure scores (``rule_based_task_probs``).
    2. Map the task to a capability level via ``HEURISTIC_TASK_LEVEL``.
    3. Escalate one level for long context (>= 2000 estimated tokens) or large
       ``max_tokens`` (> 1000); escalate to at least level 2 when JSON output is requested.
    4. Pick the cheapest eligible model at or above that level's tier.
    """

    name = "heuristic"

    def decide(self, ctx: RoutingContext) -> RoutingDecision:
        eligible = _require_eligible(ctx)
        task = ctx.predicted_task
        level = HEURISTIC_TASK_LEVEL.get(task, 2)
        reasons = [f"task={task}"]
        if ctx.features.input_tokens_est >= LONG_CONTEXT_TOKENS:
            level += 1
            reasons.append("long_context")
        if (ctx.params.get("max_tokens") or 0) > 1000:
            level += 1
            reasons.append("large_max_tokens")
        if ctx.features.json_requested and level < 2:
            level = 2
            reasons.append("json_requested")
        tiers = sorted({m.tier for m in ctx.candidates})
        level = min(level, 3)
        # Map level 0..3 proportionally onto available tiers.
        target_tier = tiers[round(level / 3 * (len(tiers) - 1))]
        at_or_above = [m for m in eligible if m.tier >= target_tier]
        pool = at_or_above or sorted(eligible, key=lambda m: -m.tier)[:1]
        target = min(pool, key=lambda m: (m.tier, m.input_cost_per_mtok + m.output_cost_per_mtok))
        return RoutingDecision(
            strategy=self.name,
            selected_model=target.id,
            fallback_chain=tier_fallbacks(target, eligible),
            reason=";".join(reasons) + f";level={level}",
            candidates=list(_estimates(ctx, self.cost_model, self.latency_model).values()),
            predicted_task=task,
        )


class LearnedStrategy(_EstimatingStrategy):
    """Quality-aware cost minimisation.

        selected = argmin_m expected_cost(m, q)
                   s.t. P_hat(acceptable | q, m) - margin_m >= threshold
                        predicted_latency(m, q) <= latency_budget   (if configured)

    * P_hat comes from per-model classifiers calibrated on a held-out validation split.
    * margin_m is the model's validation calibration error (ECE), plus an extra
      uncertainty margin when the request looks out-of-distribution (input longer than
      anything seen in training, or a low-confidence task prediction).
    * If no model satisfies the constraints: choose the highest-P_hat model within the
      latency budget; if none fits the budget, the highest-P_hat model overall
      (quality is preferred over latency when both cannot be met).

    Ablations: ``variant="no_task"`` drops task-classifier features; ``use_cost=False``
    picks uniformly (seeded) among models satisfying the constraints instead of the
    cheapest; ``use_latency=False`` ignores the latency budget.
    """

    def __init__(
        self,
        name: str,
        artifact: RouterArtifact,
        cost_model: CostModel,
        latency_model: LatencyModel,
        *,
        variant: str = "full",
        use_cost: bool = True,
        use_latency: bool = True,
        uncertainty_margin: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(cost_model, latency_model)
        if variant not in artifact.variants:
            raise ValueError(f"artifact has no variant {variant!r}")
        self.name = name
        self.artifact = artifact
        self.variant = variant
        self.use_cost = use_cost
        self.use_latency = use_latency
        self.uncertainty_margin = uncertainty_margin
        self.seed = seed

    def decide(self, ctx: RoutingContext) -> RoutingDecision:
        eligible = _require_eligible(ctx)
        scores = _estimates(ctx, self.cost_model, self.latency_model)
        probs = self.artifact.quality_probs(ctx.features, ctx.task_probs, self.variant)

        uncertain = (
            ctx.features.input_tokens_est > self.artifact.max_train_input_tokens
            or max(ctx.task_probs.values()) < self.artifact.ood_task_confidence
        )
        extra = self.uncertainty_margin if uncertain else 0.0
        budget = ctx.latency_budget_ms if self.use_latency else None

        for spec in ctx.candidates:
            s = scores[spec.id]
            p = probs.get(spec.id)
            if p is None:  # model not covered by the artifact: cannot vouch for quality
                s.p_quality = None
                s.meets_quality = False
                continue
            s.p_quality = p
            s.p_quality_adjusted = p - self.artifact.margin(self.variant, spec.id) - extra
            s.meets_quality = s.p_quality_adjusted >= ctx.quality_threshold
            s.meets_latency = True if budget is None else (s.expected_latency_ms or 0.0) <= budget

        elig = [scores[m.id] for m in eligible]
        feasible = [s for s in elig if s.meets_quality and s.meets_latency]
        if feasible:
            if self.use_cost:
                chosen = min(feasible, key=lambda s: (s.expected_cost_usd or 0.0, s.model_id))
                reason = "cheapest_meeting_constraints"
            else:
                rng = random.Random(f"{self.seed}:{ctx.fingerprint}")
                chosen = rng.choice(sorted(feasible, key=lambda s: s.model_id))
                reason = "random_meeting_constraints"
        else:
            in_budget = [s for s in elig if s.meets_latency and s.p_quality is not None]
            pool = in_budget or [s for s in elig if s.p_quality is not None] or elig
            chosen = max(pool, key=lambda s: (s.p_quality or 0.0, -(s.expected_cost_usd or 0.0)))
            reason = (
                "no_model_met_quality:best_within_latency"
                if in_budget
                else "no_model_met_constraints:best_quality"
            )

        # Fallbacks: other feasible models by cost, then the rest by predicted quality.
        rest = [s for s in elig if s.model_id != chosen.model_id]
        feas_rest = sorted(
            (s for s in rest if s.meets_quality and s.meets_latency),
            key=lambda s: s.expected_cost_usd or 0.0,
        )
        infeas_rest = sorted(
            (s for s in rest if not (s.meets_quality and s.meets_latency)),
            key=lambda s: -(s.p_quality or 0.0),
        )
        return RoutingDecision(
            strategy=self.name,
            selected_model=chosen.model_id,
            fallback_chain=[s.model_id for s in feas_rest + infeas_rest],
            reason=reason,
            candidates=list(scores.values()),
            confidence=chosen.p_quality,
            predicted_task=ctx.predicted_task,
            uncertain=uncertain,
        )
