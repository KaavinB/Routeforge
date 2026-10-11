from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.registry import ModelSpec
from app.routing.features import RequestFeatures


@dataclass
class CandidateScore:
    model_id: str
    eligible: bool
    ineligible_reason: str | None = None
    p_quality: float | None = None  # calibrated P(acceptable | request, model)
    p_quality_adjusted: float | None = None  # after safety margin
    expected_output_tokens: float | None = None
    expected_cost_usd: float | None = None
    expected_latency_ms: float | None = None
    meets_quality: bool | None = None
    meets_latency: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class RoutingContext:
    messages: list[dict[str, str]]
    params: dict[str, Any]
    features: RequestFeatures
    task_probs: dict[str, float]
    quality_threshold: float
    latency_budget_ms: float | None
    candidates: list[ModelSpec]  # all routable models, ordered by tier
    ineligible: dict[str, str]  # model_id -> reason (circuit open, context too small, ...)
    fingerprint: str  # stable hash of the request, used for reproducible randomness

    @property
    def eligible(self) -> list[ModelSpec]:
        return [m for m in self.candidates if m.id not in self.ineligible]

    @property
    def predicted_task(self) -> str:
        return max(self.task_probs, key=lambda k: self.task_probs[k])


@dataclass
class RoutingDecision:
    strategy: str
    selected_model: str
    fallback_chain: list[str]
    reason: str
    candidates: list[CandidateScore] = field(default_factory=list)
    confidence: float | None = None
    predicted_task: str | None = None
    quality_threshold: float | None = None
    latency_budget_ms: float | None = None
    uncertain: bool = False
    decision_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "selected_model": self.selected_model,
            "fallback_chain": self.fallback_chain,
            "reason": self.reason,
            "confidence": self.confidence,
            "predicted_task": self.predicted_task,
            "quality_threshold": self.quality_threshold,
            "latency_budget_ms": self.latency_budget_ms,
            "uncertain": self.uncertain,
            "decision_ms": round(self.decision_ms, 4),
            "candidates": [c.as_dict() for c in self.candidates],
        }


class RoutingStrategy(Protocol):
    name: str

    def decide(self, ctx: RoutingContext) -> RoutingDecision: ...


class NoEligibleModelError(Exception):
    pass


def tier_fallbacks(selected: ModelSpec, eligible: list[ModelSpec]) -> list[str]:
    """Order alternative models by tier distance, preferring stronger models on ties."""
    others = [m for m in eligible if m.id != selected.id]
    return [
        m.id
        for m in sorted(
            others,
            key=lambda m: abs(m.tier - selected.tier) + (0.0 if m.tier > selected.tier else 0.5),
        )
    ]
