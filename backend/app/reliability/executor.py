"""Resilient execution across a chain of candidate models.

For each model in ``[selected] + fallbacks[:max_fallbacks]``:

1. Skip it if its circuit breaker rejects the call.
2. Call it with a timeout bounded by the request's overall deadline.
3. On a retryable error, retry the same model with exponential backoff + full jitter,
   up to ``max_attempts_per_model``, and only if the global retry budget allows.
4. On a non-retryable but fallbackable error (or exhausted retries), move to the next
   model. Client errors (HTTP 400-class) stop immediately: another model will not fix a
   malformed request, and trying would only multiply cost.

Worst case upstream calls per request = (1 + max_fallbacks) * max_attempts_per_model
(6 with defaults), further limited by the retry budget and the request deadline.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import BREAKER_KINDS, ErrorKind, InvalidRequestError, ProviderError
from app.core.registry import ModelRegistry, ModelSpec
from app.providers.base import ProviderRequest, ProviderResponse, StreamChunk
from app.providers.factory import ProviderPool
from app.reliability.circuit_breaker import BreakerRegistry
from app.reliability.concurrency import ConcurrencyLimiter, ConcurrencyLimitExceededError
from app.reliability.retry import BackoffPolicy, RetryBudget
from app.telemetry import metrics

logger = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    model_id: str
    attempt: int
    outcome: str  # "success" or an ErrorKind value
    latency_s: float


@dataclass
class ExecutionResult:
    model_id: str
    response: ProviderResponse
    attempts: list[AttemptRecord]
    dropped_params: list[str]
    fallback_from: str | None = None

    @property
    def fallback_used(self) -> bool:
        return self.fallback_from is not None


@dataclass
class StreamHandle:
    model_id: str
    first: StreamChunk
    rest: AsyncIterator[StreamChunk]
    attempts: list[AttemptRecord]
    dropped_params: list[str]
    started_at: float
    first_chunk_at: float
    fallback_from: str | None = None


class AllAttemptsFailedError(Exception):
    def __init__(self, last_error: ProviderError, attempts: list[AttemptRecord]) -> None:
        super().__init__(str(last_error))
        self.last_error = last_error
        self.attempts = attempts


@dataclass
class _ChainState:
    attempts: list[AttemptRecord] = field(default_factory=list)
    last_error: ProviderError | None = None
    first_model: str | None = None


def adapt_params(
    spec: ModelSpec, params: dict[str, Any], *, cap: int, policy: str = "drop"
) -> tuple[dict[str, Any], list[str]]:
    """Return (params accepted by this model, names of dropped params). ``max_tokens`` is
    always sent, capped, so a single request can never generate unbounded output. Shared
    by the gateway and the router-training collector so training data reflects exactly
    what production sends upstream."""
    allowed = set(spec.supported_params)
    adapted: dict[str, Any] = {}
    dropped: list[str] = []
    for name, value in params.items():
        if name == "max_tokens" or name in allowed:
            adapted[name] = value
        else:
            dropped.append(name)
    if dropped and policy == "reject":
        raise InvalidRequestError(
            f"model {spec.id} does not support parameter(s): {', '.join(sorted(dropped))}",
            code="unsupported_parameter",
            param=sorted(dropped)[0],
        )
    requested = adapted.get("max_tokens") or cap
    adapted["max_tokens"] = int(min(requested, spec.max_output_tokens, cap))
    return adapted, sorted(dropped)


class ResilientExecutor:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: ProviderPool,
        breakers: BreakerRegistry,
        concurrency: ConcurrencyLimiter,
        *,
        retry_budget: RetryBudget,
        backoff: BackoffPolicy,
        max_attempts_per_model: int = 2,
        max_fallbacks: int = 2,
        max_output_tokens_cap: int = 4096,
        unsupported_param_policy: str = "drop",
        seed: int = 0,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.breakers = breakers
        self.concurrency = concurrency
        self.retry_budget = retry_budget
        self.backoff = backoff
        self.max_attempts_per_model = max_attempts_per_model
        self.max_fallbacks = max_fallbacks
        self.max_output_tokens_cap = max_output_tokens_cap
        self.unsupported_param_policy = unsupported_param_policy
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers
    def adapt_params(
        self, spec: ModelSpec, params: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        return adapt_params(
            spec, params, cap=self.max_output_tokens_cap, policy=self.unsupported_param_policy
        )

    def chain(self, selected: str, fallbacks: list[str]) -> list[str]:
        return [selected, *fallbacks[: self.max_fallbacks]]

    def _record_failure(self, spec: ModelSpec, err: ProviderError, latency: float) -> None:
        metrics.PROVIDER_ERRORS.labels(spec.family, spec.id, err.kind.value).inc()
        metrics.PROVIDER_LATENCY.labels(spec.family, spec.id, "error").observe(latency)
        breaker = self.breakers.get(spec.id)
        if err.kind in BREAKER_KINDS:
            breaker.record_failure()
        else:
            breaker.release_probe()

    async def _sleep_before_retry(self, attempt: int, err: ProviderError, deadline: float) -> bool:
        if not self.retry_budget.try_acquire():
            metrics.RETRY_BUDGET_EXHAUSTED.inc()
            return False
        delay = self.backoff.delay(attempt, self._rng, err.retry_after_s)
        if time.monotonic() + delay >= deadline - 0.5:
            return False
        if err.retry_after_s is not None and err.retry_after_s > self.backoff.max_delay_s:
            return False  # provider asks us to back off longer than we tolerate: fall back
        metrics.RETRIES.labels(err.model_id or "unknown", err.kind.value).inc()
        await asyncio.sleep(delay)
        return True

    # ---------------------------------------------------------------- complete
    async def complete(
        self,
        chain: list[str],
        messages: list[dict[str, str]],
        params: dict[str, Any],
        deadline: float,
    ) -> ExecutionResult:
        state = _ChainState()
        for idx, model_id in enumerate(chain):
            spec = self.registry.require(model_id)
            adapted, dropped = self.adapt_params(spec, params)
            breaker = self.breakers.get(model_id)
            for attempt in range(1, self.max_attempts_per_model + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0.5:
                    break
                if not breaker.try_acquire():
                    state.attempts.append(
                        AttemptRecord(model_id, attempt, ErrorKind.CIRCUIT_OPEN, 0.0)
                    )
                    state.last_error = ProviderError(
                        ErrorKind.CIRCUIT_OPEN, "circuit open", model_id=model_id
                    )
                    break
                req = ProviderRequest(
                    spec, messages, adapted, timeout_s=min(spec.timeout_s, remaining)
                )
                started = time.perf_counter()
                try:
                    async with self.concurrency.model_slot(model_id, spec.max_concurrency):
                        response = await asyncio.wait_for(
                            self.providers.for_model(spec).complete(req),
                            timeout=req.timeout_s + 1.0,
                        )
                except ConcurrencyLimitExceededError:
                    breaker.release_probe()
                    metrics.RATE_LIMITED.labels("model_concurrency").inc()
                    state.last_error = ProviderError(
                        ErrorKind.RATE_LIMITED, "model concurrency saturated", model_id=model_id
                    )
                    state.attempts.append(AttemptRecord(model_id, attempt, "saturated", 0.0))
                    break
                except (ProviderError, TimeoutError) as exc:
                    latency = time.perf_counter() - started
                    err = (
                        exc
                        if isinstance(exc, ProviderError)
                        else ProviderError(ErrorKind.TIMEOUT, "gateway timeout", model_id=model_id)
                    )
                    err.model_id = model_id
                    self._record_failure(spec, err, latency)
                    state.attempts.append(AttemptRecord(model_id, attempt, err.kind.value, latency))
                    state.last_error = err
                    if not err.fallbackable:
                        raise AllAttemptsFailedError(err, state.attempts) from None
                    if not err.retryable or attempt == self.max_attempts_per_model:
                        break
                    if not await self._sleep_before_retry(attempt, err, deadline):
                        break
                    continue
                latency = time.perf_counter() - started
                breaker.record_success()
                self.retry_budget.record_success()
                metrics.PROVIDER_LATENCY.labels(spec.family, spec.id, "success").observe(latency)
                state.attempts.append(AttemptRecord(model_id, attempt, "success", latency))
                fallback_from = chain[0] if idx > 0 else None
                if fallback_from:
                    metrics.FALLBACKS.labels(fallback_from, model_id, "success").inc()
                return ExecutionResult(model_id, response, state.attempts, dropped, fallback_from)
            if idx + 1 < len(chain):
                metrics.FALLBACKS.labels(model_id, chain[idx + 1], "attempted").inc()
        assert state.last_error is not None or not chain
        raise AllAttemptsFailedError(
            state.last_error or ProviderError(ErrorKind.UNKNOWN, "empty model chain"),
            state.attempts,
        )

    # ------------------------------------------------------------------ stream
    async def open_stream(
        self,
        chain: list[str],
        messages: list[dict[str, str]],
        params: dict[str, Any],
        deadline: float,
    ) -> StreamHandle:
        """Retries/falls back until a model produces its first chunk. Failures after the
        first chunk cannot be transparently recovered (the client already received
        output) and are surfaced to the caller of ``rest``."""
        state = _ChainState()
        for idx, model_id in enumerate(chain):
            spec = self.registry.require(model_id)
            adapted, dropped = self.adapt_params(spec, params)
            breaker = self.breakers.get(model_id)
            for attempt in range(1, self.max_attempts_per_model + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0.5:
                    break
                if not breaker.try_acquire():
                    state.attempts.append(
                        AttemptRecord(model_id, attempt, ErrorKind.CIRCUIT_OPEN, 0.0)
                    )
                    state.last_error = ProviderError(
                        ErrorKind.CIRCUIT_OPEN, "circuit open", model_id=model_id
                    )
                    break
                req = ProviderRequest(
                    spec, messages, adapted, timeout_s=min(spec.timeout_s, remaining)
                )
                started = time.perf_counter()
                agen = self.providers.for_model(spec).stream(req)
                try:
                    first = await asyncio.wait_for(anext(agen), timeout=req.timeout_s + 1.0)
                except (ProviderError, TimeoutError, StopAsyncIteration) as exc:
                    latency = time.perf_counter() - started
                    if isinstance(exc, ProviderError):
                        err = exc
                    elif isinstance(exc, StopAsyncIteration):
                        err = ProviderError(
                            ErrorKind.INVALID_RESPONSE, "empty stream", model_id=model_id
                        )
                    else:
                        err = ProviderError(
                            ErrorKind.TIMEOUT, "first-token timeout", model_id=model_id
                        )
                    err.model_id = model_id
                    await _aclose(agen)
                    self._record_failure(spec, err, latency)
                    state.attempts.append(AttemptRecord(model_id, attempt, err.kind.value, latency))
                    state.last_error = err
                    if not err.fallbackable:
                        raise AllAttemptsFailedError(err, state.attempts) from None
                    if not err.retryable or attempt == self.max_attempts_per_model:
                        break
                    if not await self._sleep_before_retry(attempt, err, deadline):
                        break
                    continue
                first_at = time.perf_counter()
                state.attempts.append(
                    AttemptRecord(model_id, attempt, "success", first_at - started)
                )
                fallback_from = chain[0] if idx > 0 else None
                if fallback_from:
                    metrics.FALLBACKS.labels(fallback_from, model_id, "success").inc()
                return StreamHandle(
                    model_id=model_id,
                    first=first,
                    rest=self._guard_stream(spec, agen),
                    attempts=state.attempts,
                    dropped_params=dropped,
                    started_at=started,
                    first_chunk_at=first_at,
                    fallback_from=fallback_from,
                )
            if idx + 1 < len(chain):
                metrics.FALLBACKS.labels(model_id, chain[idx + 1], "attempted").inc()
        raise AllAttemptsFailedError(
            state.last_error or ProviderError(ErrorKind.UNKNOWN, "empty model chain"),
            state.attempts,
        )

    async def _guard_stream(
        self, spec: ModelSpec, agen: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """Applies an idle timeout between chunks and updates breaker state at the end."""
        breaker = self.breakers.get(spec.id)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(agen), timeout=spec.timeout_s)
                except StopAsyncIteration:
                    break
                yield chunk
        except (ProviderError, TimeoutError) as exc:
            err = (
                exc
                if isinstance(exc, ProviderError)
                else ProviderError(ErrorKind.TIMEOUT, "stream idle timeout", model_id=spec.id)
            )
            self._record_failure(spec, err, 0.0)
            metrics.STREAM_ABORTS.labels(spec.id).inc()
            raise err from None
        finally:
            await _aclose(agen)
        breaker.record_success()
        self.retry_budget.record_success()


async def _aclose(agen: AsyncIterator[StreamChunk]) -> None:
    aclose = getattr(agen, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            logger.debug("error closing upstream stream", exc_info=True)
