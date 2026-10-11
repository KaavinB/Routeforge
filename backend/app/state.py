"""Application wiring: constructs every component from settings (composition root)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.asyncio import Redis

from app.caching.cache import CachePolicy, ResponseCache
from app.core.config import Settings
from app.core.registry import ModelRegistry
from app.models.db import Database
from app.providers.factory import ProviderPool
from app.providers.faults import FaultInjector
from app.providers.replay import ReplayProvider
from app.reliability.circuit_breaker import (
    STATE_VALUE,
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
)
from app.reliability.concurrency import ConcurrencyLimiter
from app.reliability.executor import ResilientExecutor
from app.reliability.rate_limit import RateLimiter
from app.reliability.retry import BackoffPolicy, RetryBudget
from app.routing.engine import RoutingEngine
from app.services.auth import AuthService
from app.services.gateway import GatewayService
from app.telemetry import metrics
from app.telemetry.langfuse_sink import LangfuseSink
from app.telemetry.request_log import RequestLogWriter

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    registry: ModelRegistry
    db: Database | None
    redis: Redis | None
    auth: AuthService
    engine: RoutingEngine
    breakers: BreakerRegistry
    fault_injector: FaultInjector | None
    executor: ResilientExecutor
    cache: ResponseCache
    log_writer: RequestLogWriter
    langfuse: LangfuseSink
    gateway: GatewayService
    retry_budget: RetryBudget

    async def close(self) -> None:
        await self.log_writer.stop()
        self.langfuse.shutdown()
        if self.redis is not None:
            await self.redis.aclose()
        if self.db is not None:
            await self.db.dispose()


def _on_breaker_change(model_id: str, state: BreakerState) -> None:
    metrics.BREAKER_STATE.labels(model_id).set(STATE_VALUE[state])
    logger.warning("circuit breaker %s -> %s", model_id, state.value)


def build_state(
    settings: Settings,
    *,
    db: Database | None = None,
    redis: Redis | None = None,
    registry: ModelRegistry | None = None,
) -> AppState:
    registry = registry or ModelRegistry.from_yaml(settings.model_config_path)
    if db is None and settings.database_url:
        db = Database(settings.database_url)
    if redis is None and settings.redis_url:
        redis = Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.redis_socket_timeout_s,
            socket_connect_timeout=settings.redis_socket_timeout_s,
            decode_responses=False,
        )

    def make_breaker(model_id: str) -> CircuitBreaker:
        metrics.BREAKER_STATE.labels(model_id).set(0)
        return CircuitBreaker(
            model_id,
            failure_threshold=settings.breaker_failure_threshold,
            failure_rate=settings.breaker_failure_rate,
            window=settings.breaker_window,
            min_calls=settings.breaker_min_calls,
            reset_timeout_s=settings.breaker_reset_timeout_s,
            on_state_change=_on_breaker_change,
        )

    breakers = BreakerRegistry(make_breaker)
    for spec in registry.all():
        breakers.get(spec.id)

    artifact = RoutingEngine.load_artifact(settings.router_artifact_path)
    engine = RoutingEngine(
        registry,
        artifact=artifact,
        default_strategy=settings.default_strategy,
        seed=settings.random_seed,
        uncertainty_margin=settings.router_uncertainty_margin,
        is_available=breakers.is_available,
    )

    replay = None
    if settings.replay_recordings_path is not None:
        replay = ReplayProvider.from_jsonl(settings.replay_recordings_path)
        logger.info("replay provider loaded %d recordings", len(replay))
    injector = (
        FaultInjector(seed=settings.random_seed) if settings.fault_injection_enabled else None
    )
    if injector is not None:
        logger.warning("FAULT INJECTION ENABLED - not for production use")
    providers = ProviderPool(
        registry,
        mock_latency_scale=settings.mock_latency_scale,
        replay=replay,
        fault_injector=injector,
    )
    concurrency = ConcurrencyLimiter(settings.global_max_concurrency, settings.concurrency_wait_s)
    budget = RetryBudget(
        ratio=settings.retry_budget_ratio, min_per_s=settings.retry_budget_min_per_s
    )
    executor = ResilientExecutor(
        registry,
        providers,
        breakers,
        concurrency,
        retry_budget=budget,
        backoff=BackoffPolicy(settings.retry_base_delay_s, settings.retry_max_delay_s),
        max_attempts_per_model=settings.retry_max_attempts_per_model,
        max_fallbacks=settings.max_fallbacks,
        max_output_tokens_cap=settings.max_output_tokens_cap,
        unsupported_param_policy=settings.unsupported_param_policy,
        seed=settings.random_seed,
    )
    cache = ResponseCache(
        redis,
        CachePolicy(
            enabled=settings.cache_enabled,
            mode=settings.cache_mode,
            ttl_s=settings.cache_ttl_s,
            max_value_bytes=settings.cache_max_value_bytes,
        ),
    )
    bootstrap: dict[str, str] = {}
    if settings.bootstrap_admin_key:
        bootstrap["admin"] = settings.bootstrap_admin_key.get_secret_value()
    if settings.bootstrap_client_key:
        bootstrap["client"] = settings.bootstrap_client_key.get_secret_value()
    auth = AuthService(
        db,
        pepper=settings.api_key_pepper.get_secret_value(),
        default_rpm=settings.default_rate_limit_rpm,
        default_concurrency=settings.default_key_max_concurrency,
        bootstrap=bootstrap,
    )
    log_writer = RequestLogWriter(
        db if settings.request_log_enabled else None,
        queue_size=settings.request_log_queue_size,
        batch_size=settings.request_log_batch_size,
    )
    langfuse = LangfuseSink(
        enabled=settings.langfuse_enabled, environment=settings.env.value, release=settings.version
    )
    gateway = GatewayService(
        settings=settings,
        registry=registry,
        engine=engine,
        executor=executor,
        cache=cache,
        rate_limiter=RateLimiter(redis, fail_open=settings.rate_limit_fail_open),
        concurrency=concurrency,
        log_writer=log_writer,
        langfuse=langfuse,
    )
    return AppState(
        settings=settings,
        registry=registry,
        db=db,
        redis=redis,
        auth=auth,
        engine=engine,
        breakers=breakers,
        fault_injector=injector,
        executor=executor,
        cache=cache,
        log_writer=log_writer,
        langfuse=langfuse,
        gateway=gateway,
        retry_budget=budget,
    )
