"""Routing engine: builds the routing context (features, task distribution, eligibility)
and dispatches to the requested strategy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.registry import ModelRegistry, ModelSpec
from app.providers.base import messages_fingerprint
from app.routing.artifact import ArtifactError, RouterArtifact
from app.routing.cost import CostModel
from app.routing.features import extract_features, routing_text, rule_based_task_probs
from app.routing.latency import LatencyModel
from app.routing.strategies import (
    FixedModelStrategy,
    HeuristicStrategy,
    LearnedStrategy,
    RandomStrategy,
)
from app.routing.types import CandidateScore, RoutingContext, RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)

# Virtual model names clients may put in the "model" field.
VIRTUAL_MODEL_PREFIX = "routeforge-"
AUTO_ALIASES = {"auto", "routeforge-auto", "routeforge"}


class UnknownStrategyError(KeyError):
    pass


class RoutingEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        artifact: RouterArtifact | None = None,
        default_strategy: str = "heuristic",
        seed: int = 1234,
        uncertainty_margin: float = 0.1,
        is_available: Callable[[str], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.artifact = artifact
        self.is_available = is_available or (lambda _model_id: True)
        self.cost_model = CostModel(artifact.output_tokens if artifact else None)
        self.latency_model = LatencyModel(artifact.latency_priors if artifact else None)
        cm, lm = self.cost_model, self.latency_model

        self.strategies: dict[str, RoutingStrategy] = {
            "premium": FixedModelStrategy("premium", registry.premium_model, cm, lm),
            "economy": FixedModelStrategy("economy", registry.economy_model, cm, lm),
            "random": RandomStrategy(seed, cm, lm),
            "heuristic": HeuristicStrategy(cm, lm),
        }
        if artifact is not None:
            missing = {m.id for m in registry.candidates()} - set(artifact.models)
            if missing:
                logger.warning(
                    "router artifact does not cover models %s; they will never be selected "
                    "by learned strategies",
                    sorted(missing),
                )
            common: dict[str, Any] = {"uncertainty_margin": uncertainty_margin, "seed": seed}
            default = artifact.default_variant
            self.strategies["learned"] = LearnedStrategy(
                "learned", artifact, cm, lm, variant=default, **common
            )
            self.strategies["learned-no-cost"] = LearnedStrategy(
                "learned-no-cost", artifact, cm, lm, variant=default, use_cost=False, **common
            )
            self.strategies["learned-no-latency"] = LearnedStrategy(
                "learned-no-latency", artifact, cm, lm, variant=default, use_latency=False, **common
            )
            for variant in ("full", "no_task", "task_only"):
                if variant in artifact.variants:
                    name = "learned-" + variant.replace("_", "-")
                    self.strategies[name] = LearnedStrategy(
                        name, artifact, cm, lm, variant=variant, **common
                    )
        if default_strategy not in self.strategies:
            logger.warning(
                "default strategy %r unavailable (no router artifact?); using heuristic",
                default_strategy,
            )
            default_strategy = "heuristic"
        self.default_strategy = default_strategy

    @classmethod
    def load_artifact(cls, path: Path | None) -> RouterArtifact | None:
        if path is None:
            return None
        try:
            artifact = RouterArtifact.load(path)
        except (ArtifactError, OSError) as exc:
            logger.error("could not load router artifact from %s: %s", path, exc)
            return None
        logger.info("loaded router artifact %s from %s", artifact.version, path)
        return artifact

    def resolve(self, model_field: str, strategy_override: str | None) -> tuple[str | None, str]:
        """Map the request's ``model`` field to (explicit_model_id | None, strategy_name)."""
        if model_field in AUTO_ALIASES:
            name = strategy_override or self.default_strategy
        elif model_field.startswith(VIRTUAL_MODEL_PREFIX):
            name = model_field[len(VIRTUAL_MODEL_PREFIX) :]
        else:
            return model_field, "explicit"
        if name not in self.strategies:
            raise UnknownStrategyError(name)
        return None, name

    def virtual_models(self) -> list[str]:
        return ["routeforge-auto"] + [VIRTUAL_MODEL_PREFIX + s for s in self.strategies]

    def build_context(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        *,
        quality_threshold: float,
        latency_budget_ms: float | None,
        candidates: list[ModelSpec] | None = None,
        learned_task_probs: bool = True,
    ) -> RoutingContext:
        """``learned_task_probs``: use the artifact's task classifier (learned strategies);
        otherwise the hand-written rule-based detector (heuristic and baselines)."""
        features = extract_features(messages, params)
        if learned_task_probs and self.artifact is not None:
            task_probs = self.artifact.task_probs(routing_text(messages))
        else:
            task_probs = rule_based_task_probs(features)
        pool = candidates if candidates is not None else self.registry.candidates()
        ineligible: dict[str, str] = {}
        needed = features.input_tokens_est + float(params.get("max_tokens") or 0)
        for spec in pool:
            if not self.is_available(spec.id):
                ineligible[spec.id] = "circuit_open"
            elif needed > spec.context_window:
                ineligible[spec.id] = "context_window"
        return RoutingContext(
            messages=messages,
            params=params,
            features=features,
            task_probs=task_probs,
            quality_threshold=quality_threshold,
            latency_budget_ms=latency_budget_ms,
            candidates=pool,
            ineligible=ineligible,
            fingerprint=messages_fingerprint(messages),
        )

    def decide(
        self,
        strategy_name: str,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        *,
        quality_threshold: float,
        latency_budget_ms: float | None = None,
    ) -> RoutingDecision:
        started = time.perf_counter()
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            raise UnknownStrategyError(strategy_name)
        ctx = self.build_context(
            messages,
            params,
            quality_threshold=quality_threshold,
            latency_budget_ms=latency_budget_ms,
        )
        decision = strategy.decide(ctx)
        decision.quality_threshold = quality_threshold
        decision.latency_budget_ms = latency_budget_ms
        decision.decision_ms = (time.perf_counter() - started) * 1000
        return decision

    def explicit(
        self, model_id: str, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> RoutingDecision:
        """Decision for a client-pinned model (benchmarking / direct access). No fallback
        to other models: an explicit model request must not silently change models."""
        started = time.perf_counter()
        features = extract_features(messages, params)
        spec = self.registry.require(model_id)
        cost, out = self.cost_model.expected_cost(
            spec,
            features.input_tokens_est,
            rule_based_task_probs(features),
            params.get("max_tokens"),
        )
        return RoutingDecision(
            strategy="explicit",
            selected_model=model_id,
            fallback_chain=[],
            reason="client_pinned_model",
            candidates=[
                CandidateScore(
                    model_id=model_id,
                    eligible=True,
                    expected_cost_usd=cost,
                    expected_output_tokens=out,
                )
            ],
            decision_ms=(time.perf_counter() - started) * 1000,
        )
