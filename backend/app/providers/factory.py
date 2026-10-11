"""Resolves a model spec to the provider instance that serves it."""

from __future__ import annotations

from app.core.registry import ModelRegistry, ModelSpec
from app.providers.base import Provider
from app.providers.faults import FaultInjectingProvider, FaultInjector
from app.providers.litellm_provider import LiteLLMProvider
from app.providers.mock import MockProvider
from app.providers.replay import ReplayProvider


class ProviderPool:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        mock_latency_scale: float = 1.0,
        replay: ReplayProvider | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.registry = registry
        self.fault_injector = fault_injector
        base: dict[str, Provider] = {
            "litellm": LiteLLMProvider(),
            "mock": MockProvider(latency_scale=mock_latency_scale),
        }
        if replay is not None:
            base["replay"] = replay
        if fault_injector is not None:
            base = {k: FaultInjectingProvider(v, fault_injector) for k, v in base.items()}
        self._providers = base

    def for_model(self, spec: ModelSpec) -> Provider:
        try:
            return self._providers[spec.provider]
        except KeyError:
            raise RuntimeError(
                f"no provider configured for kind {spec.provider!r} (model {spec.id})"
            ) from None
