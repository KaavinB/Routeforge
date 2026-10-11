"""Gateway orchestration: the request path

    authenticate -> rate/concurrency limits -> analyse & route -> cache lookup
      -> resilient execution (retries, fallbacks, breakers) -> cache write -> telemetry

Both the HTTP API and the in-process evaluation runner call this service, so evaluations
exercise exactly the production code path.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from opentelemetry.trace import Status, StatusCode

from app.caching.cache import CachedCompletion, CachePolicy, ResponseCache, cache_key
from app.core.config import Settings
from app.core.errors import (
    GatewayError,
    InvalidRequestError,
    NotFoundError,
    RateLimitExceededError,
    UpstreamUnavailableError,
    gateway_error_from_provider,
)
from app.core.registry import ModelRegistry
from app.providers.base import messages_fingerprint
from app.reliability.concurrency import ConcurrencyLimiter, ConcurrencyLimitExceededError
from app.reliability.executor import AllAttemptsFailedError, AttemptRecord, ResilientExecutor
from app.reliability.rate_limit import RateLimiter
from app.routing.engine import RoutingEngine, UnknownStrategyError
from app.routing.tokens import estimate_tokens
from app.routing.types import NoEligibleModelError, RoutingDecision
from app.schemas.openai import (
    AssistantMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    DeltaMessage,
    RouteForgeMeta,
    Usage,
)
from app.services.auth import Principal
from app.telemetry import metrics
from app.telemetry.langfuse_sink import LangfuseSink
from app.telemetry.request_log import RequestLogWriter
from app.telemetry.tracing import current_trace_id, tracer

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    principal: Principal
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cache_bypass: bool = False
    no_store: bool = False
    skip_rate_limit: bool = False  # internal callers (evaluation runner)
    source: str = "api"


@dataclass
class GatewayResult:
    response: ChatCompletionResponse
    headers: dict[str, str]
    decision: RoutingDecision
    latency_ms: float
    cache_status: str


@dataclass
class _Outcome:
    """Everything recorded about one request, for metrics, logs and Langfuse."""

    ctx: RequestContext
    req: ChatCompletionRequest
    strategy: str
    started: float
    decision: RoutingDecision | None = None
    final_model: str | None = None
    cache_status: str = "n/a"
    status: str = "success"
    http_status: int = 200
    error_type: str | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    fallback_used: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_saved_usd: float = 0.0
    provider_ms: float | None = None
    ttft_ms: float | None = None
    queue_ms: float | None = None
    simulated: bool = False
    content: str | None = None
    trace_id: str | None = None
    messages_sha: str | None = None


def _convert_messages(req: ChatCompletionRequest) -> list[dict[str, str]]:
    # "developer" is OpenAI's newer name for system instructions; providers that do not
    # know it (Anthropic, Ollama) receive it as "system".
    return [
        {"role": "system" if m.role == "developer" else m.role, "content": m.text()}
        for m in req.messages
    ]


class GatewayService:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: ModelRegistry,
        engine: RoutingEngine,
        executor: ResilientExecutor,
        cache: ResponseCache,
        rate_limiter: RateLimiter,
        concurrency: ConcurrencyLimiter,
        log_writer: RequestLogWriter,
        langfuse: LangfuseSink,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.engine = engine
        self.executor = executor
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.concurrency = concurrency
        self.log_writer = log_writer
        self.langfuse = langfuse

    # ----------------------------------------------------------------- shared
    async def _admit(self, ctx: RequestContext) -> dict[str, str]:
        if ctx.skip_rate_limit:
            return {}
        p = ctx.principal
        result = await self.rate_limiter.check(p.key_id, p.rate_limit_rpm)
        if result.degraded:
            metrics.DEGRADED.labels("rate_limiter").inc()
        headers = {"x-ratelimit-remaining-requests": str(int(result.remaining))}
        if not result.allowed:
            metrics.RATE_LIMITED.labels("api_key_rpm").inc()
            raise RateLimitExceededError(
                "Rate limit exceeded for this API key.",
                code="rate_limit_exceeded",
                headers={**headers, "retry-after": str(max(1, round(result.retry_after_s)))},
            )
        return headers

    def _validate(self, req: ChatCompletionRequest) -> None:
        if len(req.messages) > self.settings.max_messages:
            raise InvalidRequestError(
                f"Too many messages (max {self.settings.max_messages}).", param="messages"
            )
        mt = req.effective_max_tokens
        if mt is not None and mt > self.settings.max_output_tokens_cap:
            raise InvalidRequestError(
                f"max_tokens exceeds the gateway cap of {self.settings.max_output_tokens_cap}.",
                param="max_tokens",
            )

    def _route(
        self, req: ChatCompletionRequest, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> RoutingDecision:
        opts = req.routeforge
        try:
            explicit, strategy = self.engine.resolve(req.model, opts.strategy if opts else None)
        except UnknownStrategyError as exc:
            raise InvalidRequestError(
                f"Unknown routing strategy '{exc.args[0]}'. Available: "
                + ", ".join(sorted(self.engine.strategies)),
                param="model",
                code="unknown_strategy",
            ) from None
        with tracer().start_as_current_span("routing.decide") as span:
            if explicit is not None:
                spec = self.registry.get(explicit)
                if spec is None or not spec.enabled:
                    raise NotFoundError(
                        f"The model '{explicit}' does not exist.",
                        code="model_not_found",
                        param="model",
                    )
                decision = self.engine.explicit(explicit, messages, params)
            else:
                threshold = (
                    opts.quality_threshold
                    if opts and opts.quality_threshold is not None
                    else self.settings.default_quality_threshold
                )
                budget = (
                    opts.latency_budget_ms
                    if opts and opts.latency_budget_ms is not None
                    else self.settings.default_latency_budget_ms
                )
                try:
                    decision = self.engine.decide(
                        strategy,
                        messages,
                        params,
                        quality_threshold=threshold,
                        latency_budget_ms=budget,
                    )
                except NoEligibleModelError as exc:
                    raise UpstreamUnavailableError(
                        "No model is currently available for this request.",
                        code="no_eligible_model",
                    ) from exc
            span.set_attribute("routeforge.strategy", decision.strategy)
            span.set_attribute("routeforge.selected_model", decision.selected_model)
            span.set_attribute("routeforge.reason", decision.reason)
        metrics.ROUTING_LATENCY.labels(decision.strategy).observe(decision.decision_ms / 1000)
        if decision.confidence is not None:
            metrics.PREDICTED_QUALITY.labels(decision.strategy).observe(decision.confidence)
        return decision

    def _cache_key_for(
        self,
        ctx: RequestContext,
        model_id: str,
        messages: list[dict[str, str]],
        params: dict[str, Any],
    ) -> str:
        adapted, _ = self.executor.adapt_params(self.registry.require(model_id), params)
        # The *requested* temperature participates too: two requests that differ only in a
        # parameter the model ignores still have different client-visible semantics.
        return cache_key(
            tenant=ctx.principal.tenant,
            model_id=model_id,
            registry_fingerprint=self.registry.fingerprint,
            messages=messages,
            params={"adapted": adapted, "requested": params},
        )

    def _cache_eligibility(
        self, req: ChatCompletionRequest, ctx: RequestContext, params: dict[str, Any]
    ) -> tuple[bool, str]:
        opt_out = ctx.cache_bypass or bool(req.routeforge and req.routeforge.cache is False)
        policy: CachePolicy = self.cache.policy
        return policy.eligible(params, opt_out=opt_out, no_store=ctx.no_store)

    def _headers(self, o: _Outcome, dropped: list[str]) -> dict[str, str]:
        h = {
            "x-request-id": o.ctx.request_id,
            "x-routeforge-strategy": o.strategy,
            "x-routeforge-cache": o.cache_status,
            "x-routeforge-cost-usd": f"{o.cost_usd:.8f}",
            "x-routeforge-fallback": str(o.fallback_used).lower(),
            "x-routeforge-simulated": str(o.simulated).lower(),
        }
        if o.final_model:
            h["x-routeforge-model"] = o.final_model
        if o.trace_id:
            h["x-routeforge-trace-id"] = o.trace_id
        if dropped:
            h["x-routeforge-dropped-params"] = ",".join(dropped)
        return h

    # ------------------------------------------------------------ non-stream
    async def chat(self, req: ChatCompletionRequest, ctx: RequestContext) -> GatewayResult:
        o = _Outcome(ctx=ctx, req=req, strategy="unresolved", started=time.perf_counter())
        rl_headers: dict[str, str] = {}
        metrics.IN_FLIGHT.inc()
        with tracer().start_as_current_span("gateway.chat_completion") as span:
            o.trace_id = current_trace_id()
            try:
                self._validate(req)
                rl_headers = await self._admit(ctx)
                messages = _convert_messages(req)
                o.messages_sha = messages_fingerprint(messages)
                params = req.sampling_params()
                async with self.concurrency.request_slot(
                    ctx.principal.key_id, ctx.principal.max_concurrency
                ) as waited:
                    o.queue_ms = waited * 1000
                    metrics.QUEUE_WAIT.observe(waited)
                    decision = self._route(req, messages, params)
                    o.decision, o.strategy = decision, decision.strategy
                    result = await self._execute(req, ctx, o, decision, messages, params)
                span.set_attribute("routeforge.final_model", o.final_model or "")
                span.set_attribute("routeforge.cache", o.cache_status)
                return GatewayResult(
                    response=result,
                    headers={
                        **self._headers(
                            o, result.routeforge.dropped_params if result.routeforge else []
                        ),
                        **rl_headers,
                    },
                    decision=decision,
                    latency_ms=(time.perf_counter() - o.started) * 1000,
                    cache_status=o.cache_status,
                )
            except ConcurrencyLimitExceededError as exc:
                metrics.RATE_LIMITED.labels(exc.scope).inc()
                err = RateLimitExceededError(
                    "Too many concurrent requests.",
                    code="concurrency_limit_exceeded",
                    headers={"retry-after": "1"},
                )
                self._fail(o, err, span)
                raise err from None
            except GatewayError as err:
                self._fail(o, err, span)
                err.headers = {**err.headers, **self._headers(o, []), **rl_headers}
                raise
            finally:
                metrics.IN_FLIGHT.dec()
                self._finish(o)

    async def _execute(
        self,
        req: ChatCompletionRequest,
        ctx: RequestContext,
        o: _Outcome,
        decision: RoutingDecision,
        messages: list[dict[str, str]],
        params: dict[str, Any],
    ) -> ChatCompletionResponse:
        eligible, reason = self._cache_eligibility(req, ctx, params)
        key = None
        if eligible:
            key = self._cache_key_for(ctx, decision.selected_model, messages, params)
            with tracer().start_as_current_span("cache.lookup"):
                hit = await self.cache.get(key)
            if hit is not None:
                o.cache_status = "hit"
                o.final_model = hit.model_id
                o.input_tokens, o.output_tokens = hit.input_tokens, hit.output_tokens
                o.cache_saved_usd = hit.original_cost_usd
                o.content = hit.content
                o.simulated = self.registry.simulated
                _, dropped = self.executor.adapt_params(self.registry.require(hit.model_id), params)
                return self._build_response(o, hit.content, hit.finish_reason, dropped)
            o.cache_status = "miss"
        else:
            o.cache_status = reason

        chain = self.executor.chain(decision.selected_model, decision.fallback_chain)
        deadline = time.monotonic() + self.settings.request_deadline_s
        with tracer().start_as_current_span("provider.execute") as span:
            try:
                result = await self.executor.complete(chain, messages, params, deadline)
            except AllAttemptsFailedError as exc:
                o.attempts = exc.attempts
                o.final_model = exc.last_error.model_id
                span.set_status(Status(StatusCode.ERROR, exc.last_error.kind.value))
                raise gateway_error_from_provider(exc.last_error) from None
            span.set_attribute("routeforge.attempts", len(result.attempts))
        resp = result.response
        spec = self.registry.require(result.model_id)
        o.final_model = result.model_id
        o.attempts = result.attempts
        o.fallback_used = result.fallback_used
        o.input_tokens, o.output_tokens = resp.input_tokens, resp.output_tokens
        o.cost_usd = spec.cost_usd(resp.input_tokens, resp.output_tokens)
        o.provider_ms = resp.latency_s * 1000
        o.simulated = resp.simulated or self.registry.simulated
        o.content = resp.content
        self.engine.latency_model.observe(
            spec, resp.input_tokens, resp.output_tokens, resp.latency_s * 1000
        )
        if eligible and resp.finish_reason in ("stop", "length", None):
            final_key = (
                key
                if result.model_id == decision.selected_model and key
                else self._cache_key_for(ctx, result.model_id, messages, params)
            )
            await self.cache.set(
                final_key,
                CachedCompletion(
                    content=resp.content,
                    finish_reason=resp.finish_reason,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    model_id=result.model_id,
                    original_cost_usd=o.cost_usd,
                    created_at=time.time(),
                ),
            )
        return self._build_response(o, resp.content, resp.finish_reason, result.dropped_params)

    def _build_response(
        self, o: _Outcome, content: str, finish_reason: str | None, dropped: list[str]
    ) -> ChatCompletionResponse:
        assert o.final_model is not None
        return ChatCompletionResponse(
            id=f"chatcmpl-{o.ctx.request_id}",
            model=o.final_model,
            choices=[
                Choice(message=AssistantMessage(content=content), finish_reason=finish_reason)
            ],
            usage=Usage(
                prompt_tokens=o.input_tokens,
                completion_tokens=o.output_tokens,
                total_tokens=o.input_tokens + o.output_tokens,
            ),
            system_fingerprint=f"rf-{self.registry.fingerprint}",
            routeforge=RouteForgeMeta(
                request_id=o.ctx.request_id,
                strategy=o.strategy,
                selected_model=o.decision.selected_model if o.decision else o.final_model,
                cache=o.cache_status,
                cost_usd=round(o.cost_usd, 10),
                fallback_used=o.fallback_used,
                simulated=o.simulated,
                dropped_params=dropped,
            ),
        )

    # ---------------------------------------------------------------- stream
    async def chat_stream(
        self, req: ChatCompletionRequest, ctx: RequestContext
    ) -> tuple[dict[str, str], AsyncIterator[bytes]]:
        """Performs admission, routing and the first upstream attempt eagerly so that
        errors before the first token are returned as normal HTTP errors; returns the
        SSE body iterator."""
        o = _Outcome(ctx=ctx, req=req, strategy="unresolved", started=time.perf_counter())
        metrics.IN_FLIGHT.inc()
        slot = self.concurrency.request_slot(ctx.principal.key_id, ctx.principal.max_concurrency)
        slot_held = False
        span = tracer().start_span("gateway.chat_completion.stream")
        o.trace_id = format(span.get_span_context().trace_id, "032x")
        try:
            self._validate(req)
            rl_headers = await self._admit(ctx)
            messages = _convert_messages(req)
            o.messages_sha = messages_fingerprint(messages)
            params = req.sampling_params()
            waited = await slot.__aenter__()
            slot_held = True
            o.queue_ms = waited * 1000
            decision = self._route(req, messages, params)
            o.decision, o.strategy = decision, decision.strategy
            include_usage = bool(req.stream_options and req.stream_options.include_usage)

            eligible, reason = self._cache_eligibility(req, ctx, params)
            o.cache_status = reason
            key = None
            if eligible:
                key = self._cache_key_for(ctx, decision.selected_model, messages, params)
                hit = await self.cache.get(key)
                if hit is not None:
                    o.cache_status = "hit"
                    o.final_model = hit.model_id
                    o.input_tokens, o.output_tokens = hit.input_tokens, hit.output_tokens
                    o.cache_saved_usd = hit.original_cost_usd
                    o.simulated = self.registry.simulated
                    o.ttft_ms = (time.perf_counter() - o.started) * 1000
                    headers = {**self._headers(o, []), **rl_headers}
                    body = self._replay_stream(o, hit, include_usage, slot, span)
                    slot_held = False  # ownership passes to the generator
                    return headers, body
                o.cache_status = "miss"

            chain = self.executor.chain(decision.selected_model, decision.fallback_chain)
            deadline = time.monotonic() + self.settings.request_deadline_s
            try:
                handle = await self.executor.open_stream(chain, messages, params, deadline)
            except AllAttemptsFailedError as exc:
                o.attempts = exc.attempts
                o.final_model = exc.last_error.model_id
                raise gateway_error_from_provider(exc.last_error) from None
            o.final_model = handle.model_id
            o.attempts = handle.attempts
            o.fallback_used = handle.fallback_from is not None
            o.ttft_ms = (handle.first_chunk_at - o.started) * 1000
            o.simulated = self.registry.simulated
            metrics.TTFT.labels(handle.model_id).observe(o.ttft_ms / 1000)
            headers = {**self._headers(o, handle.dropped_params), **rl_headers}
            body = self._live_stream(
                o, handle, include_usage, key if eligible else None, messages, params, slot, span
            )
            slot_held = False
            return headers, body
        except ConcurrencyLimitExceededError as exc:
            metrics.RATE_LIMITED.labels(exc.scope).inc()
            err = RateLimitExceededError(
                "Too many concurrent requests.",
                code="concurrency_limit_exceeded",
                headers={"retry-after": "1"},
            )
            self._fail(o, err, span)
            span.end()
            metrics.IN_FLIGHT.dec()
            self._finish(o)
            raise err from None
        except GatewayError as err:
            self._fail(o, err, span)
            err.headers = {**err.headers, **self._headers(o, [])}
            span.end()
            metrics.IN_FLIGHT.dec()
            self._finish(o)
            raise
        finally:
            if slot_held:
                await slot.__aexit__(None, None, None)

    def _chunk(
        self,
        o: _Outcome,
        delta: DeltaMessage,
        finish: str | None = None,
        usage: Usage | None = None,
    ) -> bytes:
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{o.ctx.request_id}",
            created=int(time.time()),
            model=o.final_model or "",
            choices=[]
            if usage is not None and finish is None and delta.content is None and delta.role is None
            else [ChunkChoice(delta=delta, finish_reason=finish)],
            usage=usage,
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()

    async def _replay_stream(
        self, o: _Outcome, hit: CachedCompletion, include_usage: bool, slot: Any, span: Any
    ) -> AsyncIterator[bytes]:
        try:
            yield self._chunk(o, DeltaMessage(role="assistant", content=""))
            for i in range(0, len(hit.content), 64):
                yield self._chunk(o, DeltaMessage(content=hit.content[i : i + 64]))
            yield self._chunk(o, DeltaMessage(), finish=hit.finish_reason or "stop")
            if include_usage:
                yield self._chunk(o, DeltaMessage(), usage=self._usage(o))
            yield b"data: [DONE]\n\n"
        finally:
            await slot.__aexit__(None, None, None)
            span.end()
            metrics.IN_FLIGHT.dec()
            self._finish(o)

    def _usage(self, o: _Outcome) -> Usage:
        return Usage(
            prompt_tokens=o.input_tokens,
            completion_tokens=o.output_tokens,
            total_tokens=o.input_tokens + o.output_tokens,
        )

    async def _live_stream(
        self,
        o: _Outcome,
        handle: Any,
        include_usage: bool,
        key: str | None,
        messages: list[dict[str, str]],
        params: dict[str, Any],
        slot: Any,
        span: Any,
    ) -> AsyncIterator[bytes]:
        parts: list[str] = []
        finish: str | None = None
        completed = False
        spec = self.registry.require(handle.model_id)
        try:
            yield self._chunk(o, DeltaMessage(role="assistant", content=""))
            chunks = [handle.first]

            async def _all() -> AsyncIterator[Any]:
                for c in chunks:
                    yield c
                async for c in handle.rest:
                    yield c

            async for chunk in _all():
                if chunk.delta:
                    parts.append(chunk.delta)
                    yield self._chunk(o, DeltaMessage(content=chunk.delta))
                if chunk.finish_reason:
                    finish = chunk.finish_reason
                if chunk.input_tokens is not None:
                    o.input_tokens = chunk.input_tokens
                if chunk.output_tokens is not None:
                    o.output_tokens = chunk.output_tokens
            content = "".join(parts)
            if not o.output_tokens:
                # Provider did not report usage; estimate and keep the request flagged.
                o.output_tokens = estimate_tokens(content)
                o.error_type = "usage_estimated"
            o.cost_usd = spec.cost_usd(o.input_tokens, o.output_tokens)
            o.provider_ms = (time.perf_counter() - handle.started_at) * 1000
            o.content = content
            yield self._chunk(o, DeltaMessage(), finish=finish or "stop")
            if include_usage:
                yield self._chunk(o, DeltaMessage(), usage=self._usage(o))
            yield b"data: [DONE]\n\n"
            completed = True
            if key is not None:
                final_key = (
                    key
                    if o.decision and handle.model_id == o.decision.selected_model
                    else self._cache_key_for(o.ctx, handle.model_id, messages, params)
                )
                await self.cache.set(
                    final_key,
                    CachedCompletion(
                        content=content,
                        finish_reason=finish,
                        input_tokens=o.input_tokens,
                        output_tokens=o.output_tokens,
                        model_id=handle.model_id,
                        original_cost_usd=o.cost_usd,
                        created_at=time.time(),
                    ),
                )
        except GatewayError:
            raise
        except Exception as exc:  # upstream failure after first token: cannot fall back
            o.status = "error"
            o.http_status = 200  # headers were already sent
            o.error_type = "stream_aborted"
            o.output_tokens = o.output_tokens or estimate_tokens("".join(parts))
            o.cost_usd = spec.cost_usd(o.input_tokens, o.output_tokens)
            logger.warning("stream aborted after first token: %s", type(exc).__name__)
            err = {
                "error": {
                    "message": "Upstream stream terminated unexpectedly.",
                    "type": "upstream_error",
                    "code": "stream_aborted",
                }
            }
            yield f"data: {json.dumps(err)}\n\n".encode()
        finally:
            if not completed and o.status == "success":
                o.status = "error"
                o.error_type = o.error_type or "client_disconnected"
            await slot.__aexit__(None, None, None)
            if o.status == "error":
                span.set_status(Status(StatusCode.ERROR, o.error_type or "error"))
            span.end()
            metrics.IN_FLIGHT.dec()
            self._finish(o)

    # ------------------------------------------------------------- telemetry
    def _fail(self, o: _Outcome, err: GatewayError, span: Any) -> None:
        o.status = "error"
        o.http_status = err.status_code
        o.error_type = err.code or err.error_type
        span.set_status(Status(StatusCode.ERROR, o.error_type or "error"))

    def _finish(self, o: _Outcome) -> None:
        latency_ms = (time.perf_counter() - o.started) * 1000
        model_label = o.final_model or "none"
        metrics.REQUESTS.labels(o.strategy, model_label, o.status, o.cache_status).inc()
        metrics.REQUEST_LATENCY.labels(o.strategy, model_label, str(o.req.stream).lower()).observe(
            latency_ms / 1000
        )
        if o.final_model and o.cache_status != "hit":
            metrics.TOKENS.labels(o.final_model, "input").inc(o.input_tokens)
            metrics.TOKENS.labels(o.final_model, "output").inc(o.output_tokens)
            metrics.COST.labels(o.final_model, o.strategy).inc(o.cost_usd)
        if o.cache_saved_usd and o.final_model:
            metrics.CACHE_SAVED_COST.labels(o.final_model).inc(o.cache_saved_usd)

        d = o.decision
        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "request_id": o.ctx.request_id,
            "trace_id": o.trace_id,
            "created_at": datetime.now(UTC),
            "api_key_id": o.ctx.principal.key_id,
            "tenant": o.ctx.principal.tenant,
            "requested_model": o.req.model,
            "strategy": o.strategy,
            "selected_model": d.selected_model if d else None,
            "final_model": o.final_model,
            "routing_reason": (d.reason[:256] if d else None),
            "predicted_task": d.predicted_task if d else None,
            "predicted_quality": d.confidence if d else None,
            "quality_threshold": d.quality_threshold if d else None,
            "routing_detail": d.summary() if d else None,
            "cache_status": o.cache_status,
            "status": o.status,
            "error_type": o.error_type,
            "http_status": o.http_status,
            "attempts": len(o.attempts),
            "attempt_detail": [a.__dict__ for a in o.attempts],
            "fallback_used": o.fallback_used,
            "stream": o.req.stream,
            "simulated": o.simulated,
            "latency_ms": latency_ms,
            "routing_ms": d.decision_ms if d else None,
            "provider_ms": o.provider_ms,
            "ttft_ms": o.ttft_ms,
            "queue_ms": o.queue_ms,
            "input_tokens": o.input_tokens,
            "output_tokens": o.output_tokens,
            "cost_usd": o.cost_usd,
            "cache_saved_usd": o.cache_saved_usd,
            "messages_sha256": o.messages_sha,
            "prompt_text": (
                json.dumps(_convert_messages(o.req))[:20000] if self.settings.log_prompts else None
            ),
            "tags": (o.req.routeforge.tags if o.req.routeforge else None),
        }
        self.log_writer.submit(record)
        if self.langfuse.enabled and o.final_model:
            self.langfuse.record_generation(
                trace_id=o.trace_id,
                name=f"routeforge.{o.strategy}",
                model=o.final_model,
                input_messages=_convert_messages(o.req) if self.settings.log_prompts else None,
                output=o.content if self.settings.log_prompts else None,
                input_tokens=o.input_tokens,
                output_tokens=o.output_tokens,
                cost_usd=o.cost_usd,
                completion_start_time=None,
                metadata={
                    "request_id": o.ctx.request_id,
                    "strategy": o.strategy,
                    "cache": o.cache_status,
                    "fallback_used": o.fallback_used,
                    "predicted_task": d.predicted_task if d else None,
                    "predicted_quality": d.confidence if d else None,
                    "routing_reason": d.reason if d else None,
                    "latency_ms": round(latency_ms, 2),
                    "simulated": o.simulated,
                },
                error=o.error_type if o.status == "error" else None,
            )
