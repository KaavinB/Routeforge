"""Per-API-key rate limiting (token bucket) with Redis as the shared store.

The bucket update runs as a single Lua script so concurrent gateway replicas cannot race.
If Redis is unreachable the limiter degrades to a per-process in-memory bucket
(``fail_open=True``, default) so a cache outage does not take the API down, or rejects
requests (``fail_open=False``) for deployments that prefer strict enforcement.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# KEYS[1] bucket key; ARGV: capacity, refill_per_ms, now_ms, cost
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_ms = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  ts = now
end
local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * refill_per_ms)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.ceil(capacity / refill_per_ms) + 1000)
return {allowed, tostring(tokens)}
"""  # noqa: S105 - Lua source, not a secret


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: float
    retry_after_s: float
    degraded: bool = False  # True when enforced by the local fallback


class _LocalBucket:
    def __init__(self, capacity: float, refill_per_s: float) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.tokens = capacity
        self.ts = time.monotonic()

    def take(self) -> tuple[bool, float]:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.refill_per_s)
        self.ts = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True, self.tokens
        return False, self.tokens


class RateLimiter:
    def __init__(self, redis: Redis | None, *, fail_open: bool = True) -> None:
        self.redis = redis
        self.fail_open = fail_open
        self._script = redis.register_script(_TOKEN_BUCKET_LUA) if redis is not None else None
        self._local: dict[str, _LocalBucket] = {}
        self._lock = threading.Lock()

    async def check(self, key_id: str, rpm: int) -> RateLimitResult:
        capacity = float(max(rpm, 1))
        refill_per_s = capacity / 60.0
        if self._script is not None:
            try:
                allowed, remaining = await self._script(
                    keys=[f"rf:rl:{key_id}"],
                    args=[capacity, refill_per_s / 1000.0, int(time.time() * 1000), 1],
                )
                remaining_f = float(remaining)
                retry = 0.0 if allowed else math.ceil((1 - remaining_f) / refill_per_s * 10) / 10
                return RateLimitResult(bool(allowed), remaining_f, retry)
            except (RedisError, OSError, TimeoutError) as exc:
                logger.warning("rate limiter redis error, degrading: %s", type(exc).__name__)
                if not self.fail_open:
                    return RateLimitResult(False, 0.0, 1.0, degraded=True)
        with self._lock:
            bucket = self._local.get(key_id)
            if bucket is None or bucket.capacity != capacity:
                bucket = self._local[key_id] = _LocalBucket(capacity, refill_per_s)
            ok, remaining_local = bucket.take()
        retry = 0.0 if ok else (1 - remaining_local) / refill_per_s
        return RateLimitResult(ok, remaining_local, retry, degraded=self._script is not None)
