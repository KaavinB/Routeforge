"""Exact-match response cache backed by Redis.

Key construction (``cache_key``): SHA-256 over a canonical JSON document containing
  * the tenant (cache entries are never shared across tenants),
  * the *resolved* model id and the registry fingerprint (so outputs are never reused
    across models, and any change to model/provider configuration invalidates entries),
  * the full message list,
  * every generation parameter after per-model adaptation (temperature, top_p,
    max_tokens, stop, seed, penalties, response_format).
Prompts never appear in keys or logs; only the digest does.

Eligibility (``CachePolicy.eligible``):
  * cache globally enabled, request did not opt out (``routeforge.cache=false`` or
    ``X-RouteForge-Cache: bypass``), request not marked ``no-store``;
  * in the default ``deterministic`` mode, only requests with ``temperature == 0``.
    A hit then returns the first answer sampled for that exact request. In ``all`` mode
    any request is cacheable (a hit returns one valid sample, not a fresh one).

Streaming: hits are replayed as a stream; misses are written only after the upstream
stream completes cleanly (an aborted stream is never cached).

Redis failures degrade to "miss" without failing the request; after an error the cache
backs off for a few seconds so an outage does not add a socket timeout to every request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import CacheMode
from app.telemetry import metrics

logger = logging.getLogger(__name__)

CACHE_PREFIX = "rf:cache:v1:"


def cache_key(
    *,
    tenant: str,
    model_id: str,
    registry_fingerprint: str,
    messages: list[dict[str, str]],
    params: dict[str, Any],
) -> str:
    doc = {
        "tenant": tenant,
        "model": model_id,
        "registry": registry_fingerprint,
        "messages": messages,
        "params": params,
    }
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return CACHE_PREFIX + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class CachedCompletion:
    content: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    model_id: str
    original_cost_usd: float
    created_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> CachedCompletion:
        return cls(**json.loads(raw))


@dataclass(frozen=True)
class CachePolicy:
    enabled: bool
    mode: CacheMode
    ttl_s: int
    max_value_bytes: int

    def eligible(
        self, params: dict[str, Any], *, opt_out: bool, no_store: bool
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "disabled"
        if opt_out:
            return False, "bypass"
        if no_store:
            return False, "no_store"
        if self.mode == CacheMode.DETERMINISTIC and params.get("temperature") != 0:
            return False, "nondeterministic"
        return True, "eligible"


class ResponseCache:
    def __init__(
        self, redis: Redis | None, policy: CachePolicy, error_backoff_s: float = 5.0
    ) -> None:
        self.redis = redis
        self.policy = policy
        self.error_backoff_s = error_backoff_s
        self._unavailable_until = 0.0

    def _available(self) -> bool:
        return self.redis is not None and time.monotonic() >= self._unavailable_until

    def _mark_error(self, exc: Exception) -> None:
        self._unavailable_until = time.monotonic() + self.error_backoff_s
        metrics.DEGRADED.labels("cache").inc()
        logger.warning("response cache unavailable: %s", type(exc).__name__)

    async def get(self, key: str) -> CachedCompletion | None:
        if not self._available():
            metrics.CACHE_LOOKUPS.labels("error").inc()
            return None
        assert self.redis is not None
        started = time.perf_counter()
        try:
            raw = await self.redis.get(key)
        except (RedisError, OSError, TimeoutError) as exc:
            self._mark_error(exc)
            metrics.CACHE_LOOKUPS.labels("error").inc()
            return None
        finally:
            metrics.CACHE_LATENCY.observe(time.perf_counter() - started)
        if raw is None:
            metrics.CACHE_LOOKUPS.labels("miss").inc()
            return None
        try:
            value = CachedCompletion.from_json(raw)
        except (ValueError, TypeError):
            metrics.CACHE_LOOKUPS.labels("corrupt").inc()
            return None
        metrics.CACHE_LOOKUPS.labels("hit").inc()
        return value

    async def set(self, key: str, value: CachedCompletion) -> bool:
        if not self._available():
            return False
        assert self.redis is not None
        payload = value.to_json()
        if len(payload.encode()) > self.policy.max_value_bytes:
            return False
        try:
            await self.redis.set(key, payload, ex=self.policy.ttl_s)
            return True
        except (RedisError, OSError, TimeoutError) as exc:
            self._mark_error(exc)
            return False

    async def invalidate_all(self) -> int:
        """Delete every cache entry (admin operation). Uses SCAN, never KEYS."""
        if self.redis is None:
            return 0
        deleted = 0
        async for key in self.redis.scan_iter(match=CACHE_PREFIX + "*", count=500):
            deleted += int(await self.redis.delete(key))
        return deleted
