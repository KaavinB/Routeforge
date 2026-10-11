"""Concurrency controls: a global in-flight cap, per-API-key caps, and per-model caps.

Requests wait at most ``wait_s`` for a slot (bounded queueing) and are then rejected,
so overload turns into fast 429/503 responses instead of unbounded latency growth.
Limits are per process; a multi-replica deployment multiplies them by replica count.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ConcurrencyLimitExceededError(Exception):
    def __init__(self, scope: str) -> None:
        super().__init__(scope)
        self.scope = scope


class ConcurrencyLimiter:
    def __init__(self, global_limit: int, wait_s: float = 2.0) -> None:
        self.wait_s = wait_s
        self._global = asyncio.Semaphore(global_limit)
        self.global_limit = global_limit
        self._per_key: dict[str, tuple[int, asyncio.Semaphore]] = {}
        self._per_model: dict[str, tuple[int, asyncio.Semaphore]] = {}
        self.in_flight = 0

    def _sem(
        self, table: dict[str, tuple[int, asyncio.Semaphore]], key: str, limit: int
    ) -> asyncio.Semaphore:
        entry = table.get(key)
        if entry is None or entry[0] != limit:
            entry = (limit, asyncio.Semaphore(limit))
            table[key] = entry
        return entry[1]

    async def _acquire(self, sem: asyncio.Semaphore, scope: str, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        try:
            await asyncio.wait_for(sem.acquire(), timeout=max(remaining, 0.0))
        except TimeoutError:
            raise ConcurrencyLimitExceededError(scope) from None

    @asynccontextmanager
    async def request_slot(self, key_id: str, key_limit: int) -> AsyncIterator[float]:
        """Acquire key + global slots. Yields the time spent waiting (seconds)."""
        started = time.monotonic()
        deadline = started + self.wait_s
        key_sem = self._sem(self._per_key, key_id, key_limit)
        await self._acquire(key_sem, "api_key", deadline)
        try:
            await self._acquire(self._global, "global", deadline)
        except ConcurrencyLimitExceededError:
            key_sem.release()
            raise
        self.in_flight += 1
        try:
            yield time.monotonic() - started
        finally:
            self.in_flight -= 1
            self._global.release()
            key_sem.release()

    @asynccontextmanager
    async def model_slot(self, model_id: str, limit: int) -> AsyncIterator[None]:
        sem = self._sem(self._per_model, model_id, limit)
        await self._acquire(sem, f"model:{model_id}", time.monotonic() + self.wait_s)
        try:
            yield
        finally:
            sem.release()
