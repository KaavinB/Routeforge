"""Fault injection for reliability testing (Benchmark D and failure-injection tests).

Rules are attached per model id and evaluated before the real provider is called, so
injected failures look to the reliability layer exactly like real upstream failures.
Disabled unless ``ROUTEFORGE_FAULT_INJECTION_ENABLED=true``; never enable in production.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.errors import ErrorKind, ProviderError
from app.providers.base import Provider, ProviderRequest, ProviderResponse, StreamChunk


class FaultMode(StrEnum):
    TIMEOUT = "timeout"  # hang until the request timeout, then fail
    HTTP_429 = "http_429"
    HTTP_500 = "http_500"
    INVALID_RESPONSE = "invalid_response"
    OUTAGE = "outage"  # connection refused, every call
    SLOW = "slow"  # add latency, then succeed
    STREAM_ABORT = "stream_abort"  # streams fail after the first chunk


class FaultRule(BaseModel):
    mode: FaultMode
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    added_latency_ms: float = Field(default=0.0, ge=0.0)
    retry_after_s: float | None = None


class FaultInjector:
    def __init__(self, seed: int = 0) -> None:
        self._rules: dict[str, FaultRule] = {}
        self._rng = random.Random(seed)
        self.injected: dict[str, int] = {}

    def set_rule(self, model_id: str, rule: FaultRule) -> None:
        self._rules[model_id] = rule

    def clear(self, model_id: str | None = None) -> None:
        if model_id is None:
            self._rules.clear()
        else:
            self._rules.pop(model_id, None)

    def rules(self) -> dict[str, FaultRule]:
        return dict(self._rules)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def draw(self, model_id: str) -> FaultRule | None:
        rule = self._rules.get(model_id)
        if rule is None or self._rng.random() >= rule.probability:
            return None
        self.injected[model_id] = self.injected.get(model_id, 0) + 1
        return rule


async def _raise_for(rule: FaultRule, request: ProviderRequest) -> None:
    mid = request.spec.id
    if rule.mode == FaultMode.TIMEOUT:
        await asyncio.sleep(request.timeout_s)
        raise ProviderError(ErrorKind.TIMEOUT, "injected timeout", model_id=mid)
    if rule.mode == FaultMode.HTTP_429:
        raise ProviderError(
            ErrorKind.RATE_LIMITED,
            "injected 429",
            status_code=429,
            retry_after_s=rule.retry_after_s,
            model_id=mid,
        )
    if rule.mode == FaultMode.HTTP_500:
        raise ProviderError(ErrorKind.SERVER_ERROR, "injected 500", status_code=500, model_id=mid)
    if rule.mode == FaultMode.INVALID_RESPONSE:
        raise ProviderError(ErrorKind.INVALID_RESPONSE, "injected malformed body", model_id=mid)
    if rule.mode == FaultMode.OUTAGE:
        raise ProviderError(ErrorKind.CONNECTION, "injected outage", model_id=mid)


class FaultInjectingProvider:
    """Wraps a real provider and applies the injector's rule for the target model."""

    def __init__(self, inner: Provider, injector: FaultInjector) -> None:
        self.inner = inner
        self.injector = injector
        self.name = inner.name

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        rule = self.injector.draw(request.spec.id)
        if rule is not None:
            if rule.added_latency_ms:
                await asyncio.sleep(rule.added_latency_ms / 1000)
            if rule.mode not in (FaultMode.SLOW, FaultMode.STREAM_ABORT):
                await _raise_for(rule, request)
        return await self.inner.complete(request)

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamChunk]:
        rule = self.injector.draw(request.spec.id)
        if rule is not None:
            if rule.added_latency_ms:
                await asyncio.sleep(rule.added_latency_ms / 1000)
            if rule.mode not in (FaultMode.SLOW, FaultMode.STREAM_ABORT):
                await _raise_for(rule, request)
        emitted = 0
        async for chunk in self.inner.stream(request):
            yield chunk
            emitted += 1
            if rule is not None and rule.mode == FaultMode.STREAM_ABORT and emitted >= 1:
                raise ProviderError(
                    ErrorKind.CONNECTION, "injected mid-stream abort", model_id=request.spec.id
                )
