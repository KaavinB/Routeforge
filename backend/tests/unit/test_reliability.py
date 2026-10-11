from __future__ import annotations

import asyncio
import random

import fakeredis
import pytest

from app.caching.cache import CachedCompletion, CachePolicy, ResponseCache, cache_key
from app.core.config import CacheMode
from app.core.errors import ErrorKind, ProviderError
from app.core.registry import ModelRegistry
from app.reliability.circuit_breaker import BreakerState, CircuitBreaker
from app.reliability.concurrency import ConcurrencyLimiter, ConcurrencyLimitExceededError
from app.reliability.executor import adapt_params
from app.reliability.rate_limit import RateLimiter
from app.reliability.retry import BackoffPolicy, RetryBudget
from tests.conftest import MOCK_REGISTRY


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------- retry
def test_backoff_full_jitter_bounds() -> None:
    policy = BackoffPolicy(base_delay_s=0.5, max_delay_s=4.0)
    rng = random.Random(0)
    for attempt in range(1, 8):
        cap = min(4.0, 0.5 * 2 ** (attempt - 1))
        for _ in range(50):
            assert 0 <= policy.delay(attempt, rng) <= cap
    assert policy.delay(1, rng, retry_after_s=2.0) >= 2.0


def test_retry_budget_caps_amplification() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_per_s=0.0, max_tokens=10, clock=clock)
    granted = sum(budget.try_acquire() for _ in range(100))
    assert granted == 5  # starts half full, no refill without successes
    for _ in range(100):
        budget.record_success()
    assert sum(budget.try_acquire() for _ in range(100)) == 10  # 100 * 0.1, capped at max


def test_error_classification() -> None:
    assert ProviderError(ErrorKind.TIMEOUT, "x").retryable
    assert ProviderError(ErrorKind.RATE_LIMITED, "x").retryable
    assert not ProviderError(ErrorKind.BAD_REQUEST, "x").retryable
    assert not ProviderError(ErrorKind.BAD_REQUEST, "x").fallbackable
    assert ProviderError(ErrorKind.CONTEXT_LENGTH, "x").fallbackable
    assert not ProviderError(ErrorKind.CONTEXT_LENGTH, "x").retryable


def test_litellm_error_mapping() -> None:
    import httpx
    import litellm

    from app.providers.litellm_provider import classify_litellm_error

    req = httpx.Request("POST", "https://x")
    rl = litellm.exceptions.RateLimitError(
        "slow down",
        llm_provider="openai",
        model="m",
        response=httpx.Response(429, request=req, headers={"retry-after": "3"}),
    )
    err = classify_litellm_error(rl, "m")
    assert err.kind == ErrorKind.RATE_LIMITED and err.retry_after_s == 3.0
    assert (
        classify_litellm_error(
            litellm.exceptions.Timeout("t", model="m", llm_provider="openai"), "m"
        ).kind
        == ErrorKind.TIMEOUT
    )
    bad = litellm.exceptions.BadRequestError("bad", model="m", llm_provider="openai")
    assert classify_litellm_error(bad, "m").kind == ErrorKind.BAD_REQUEST
    ctx = litellm.exceptions.ContextWindowExceededError(
        "too long", model="m", llm_provider="openai"
    )
    assert classify_litellm_error(ctx, "m").kind == ErrorKind.CONTEXT_LENGTH


# ---------------------------------------------------------- circuit breaker
def test_breaker_opens_on_consecutive_failures_and_recovers() -> None:
    clock = FakeClock()
    b = CircuitBreaker("m", failure_threshold=3, reset_timeout_s=10, clock=clock)
    for _ in range(3):
        b.record_failure()
    assert b.state == BreakerState.OPEN and not b.is_available()
    clock.t = 10.1
    assert b.state == BreakerState.HALF_OPEN
    assert b.try_acquire() and not b.try_acquire()  # single probe
    b.record_success()
    assert b.state == BreakerState.CLOSED


def test_breaker_half_open_failure_reopens() -> None:
    clock = FakeClock()
    b = CircuitBreaker("m", failure_threshold=1, reset_timeout_s=5, clock=clock)
    b.record_failure()
    clock.t = 6
    assert b.try_acquire()
    b.record_failure()
    assert b.state == BreakerState.OPEN
    assert b.opened_count == 2


def test_breaker_failure_rate_trip() -> None:
    b = CircuitBreaker("m", failure_threshold=100, failure_rate=0.5, window=10, min_calls=10)
    for i in range(10):
        (b.record_failure if i % 2 else b.record_success)()
    assert b.state == BreakerState.OPEN


# ------------------------------------------------------------- rate limiting
async def test_redis_token_bucket() -> None:
    limiter = RateLimiter(fakeredis.aioredis.FakeRedis())
    results = [await limiter.check("k", rpm=3) for _ in range(5)]
    assert [r.allowed for r in results] == [True, True, True, False, False]
    assert results[-1].retry_after_s > 0
    other = await limiter.check("other-key", rpm=3)
    assert other.allowed  # isolation between keys


async def test_rate_limiter_fails_open_locally_when_redis_down() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    limiter = RateLimiter(redis, fail_open=True)

    async def boom(*_a: object, **_k: object) -> None:
        raise ConnectionError("down")

    limiter._script = boom  # type: ignore[assignment]
    r = await limiter.check("k", rpm=2)
    assert r.allowed and r.degraded
    await limiter.check("k", rpm=2)
    assert not (await limiter.check("k", rpm=2)).allowed  # local bucket still enforces
    strict = RateLimiter(redis, fail_open=False)
    strict._script = boom  # type: ignore[assignment]
    assert not (await strict.check("k", rpm=100)).allowed


# --------------------------------------------------------------- concurrency
async def test_concurrency_limit_rejects_after_wait() -> None:
    limiter = ConcurrencyLimiter(global_limit=10, wait_s=0.05)
    async with limiter.request_slot("k", 1):
        with pytest.raises(ConcurrencyLimitExceededError):
            async with limiter.request_slot("k", 1):
                pass
        async with limiter.request_slot("other", 1):  # other keys unaffected
            assert limiter.in_flight == 2
    assert limiter.in_flight == 0


# ------------------------------------------------------------------ params
def test_adapt_params_drops_unsupported_and_caps_tokens() -> None:
    registry = ModelRegistry.from_yaml(MOCK_REGISTRY)
    spec = registry.require("mock-small").model_copy(update={"supported_params": ["max_tokens"]})
    adapted, dropped = adapt_params(spec, {"temperature": 0, "max_tokens": 99_999}, cap=1000)
    assert adapted == {"max_tokens": 1000}
    assert dropped == ["temperature"]
    adapted, _ = adapt_params(spec, {}, cap=512)
    assert adapted["max_tokens"] == 512  # max_tokens always sent
    with pytest.raises(Exception, match="does not support"):
        adapt_params(spec, {"temperature": 0}, cap=10, policy="reject")


# -------------------------------------------------------------------- cache
def test_cache_key_covers_all_inputs() -> None:
    base = dict(
        tenant="t",
        model_id="m",
        registry_fingerprint="f",
        messages=[{"role": "user", "content": "hi"}],
        params={"temperature": 0, "max_tokens": 10},
    )
    k = cache_key(**base)  # type: ignore[arg-type]
    assert k.startswith("rf:cache:v1:") and "hi" not in k
    for field, value in [
        ("tenant", "t2"),
        ("model_id", "m2"),
        ("registry_fingerprint", "f2"),
        ("messages", [{"role": "user", "content": "hi!"}]),
        ("params", {"temperature": 0, "max_tokens": 11}),
    ]:
        assert cache_key(**{**base, field: value}) != k  # type: ignore[arg-type]
    reordered = {**base, "params": {"max_tokens": 10, "temperature": 0}}
    assert cache_key(**reordered) == k  # type: ignore[arg-type]


def test_cache_policy() -> None:
    p = CachePolicy(enabled=True, mode=CacheMode.DETERMINISTIC, ttl_s=60, max_value_bytes=1000)
    assert p.eligible({"temperature": 0}, opt_out=False, no_store=False) == (True, "eligible")
    assert p.eligible({"temperature": 0.5}, opt_out=False, no_store=False)[1] == "nondeterministic"
    assert p.eligible({}, opt_out=False, no_store=False)[1] == "nondeterministic"
    assert p.eligible({"temperature": 0}, opt_out=True, no_store=False)[1] == "bypass"
    assert p.eligible({"temperature": 0}, opt_out=False, no_store=True)[1] == "no_store"
    p_all = CachePolicy(enabled=True, mode=CacheMode.ALL, ttl_s=60, max_value_bytes=1000)
    assert p_all.eligible({"temperature": 1}, opt_out=False, no_store=False)[0]
    off = CachePolicy(enabled=False, mode=CacheMode.ALL, ttl_s=60, max_value_bytes=1000)
    assert off.eligible({}, opt_out=False, no_store=False)[1] == "disabled"


async def test_response_cache_roundtrip_ttl_and_size_limit() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    cache = ResponseCache(
        redis, CachePolicy(True, CacheMode.DETERMINISTIC, ttl_s=30, max_value_bytes=500)
    )
    v = CachedCompletion("hello", "stop", 3, 1, "m", 0.001, 0.0)
    assert await cache.set("rf:cache:v1:a", v)
    assert (await cache.get("rf:cache:v1:a")) == v
    assert 0 < await redis.ttl("rf:cache:v1:a") <= 30
    big = CachedCompletion("x" * 1000, "stop", 3, 1, "m", 0.0, 0.0)
    assert not await cache.set("rf:cache:v1:b", big)
    assert await cache.invalidate_all() == 1
    assert await cache.get("rf:cache:v1:a") is None


async def test_response_cache_backs_off_after_redis_error() -> None:
    redis = fakeredis.aioredis.FakeRedis()
    cache = ResponseCache(redis, CachePolicy(True, CacheMode.ALL, 30, 1000), error_backoff_s=60)
    calls = 0

    async def broken(*_a: object, **_k: object) -> None:
        nonlocal calls
        calls += 1
        raise ConnectionError("down")

    redis.get = broken  # type: ignore[method-assign]
    assert await cache.get("k") is None
    assert await cache.get("k") is None
    assert calls == 1  # second lookup skipped during backoff instead of paying the timeout again
    await asyncio.sleep(0)
