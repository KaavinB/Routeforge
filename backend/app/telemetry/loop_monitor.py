"""Event-loop lag monitor: a heartbeat task that measures how late it wakes up. Sustained
lag means synchronous work is blocking the loop, which inflates every in-flight request
and can trip short socket timeouts (e.g. Redis) spuriously."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import time
from typing import Any

from app.telemetry import metrics


class LoopLagMonitor:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self.max_lag_s = 0.0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="loop-lag-monitor")

    async def _run(self) -> None:
        while True:
            started = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            lag = max(time.perf_counter() - started - self.interval_s, 0.0)
            self.max_lag_s = max(self.max_lag_s, lag)
            metrics.EVENT_LOOP_LAG.observe(lag)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


_gc_started: dict[int, float] = {}


def _gc_callback(phase: str, info: dict[str, Any]) -> None:
    gen = int(info.get("generation", 0))
    if phase == "start":
        _gc_started[gen] = time.perf_counter()
    elif gen in _gc_started:
        metrics.GC_PAUSE.labels(str(gen)).observe(time.perf_counter() - _gc_started.pop(gen))


def freeze_startup_heap() -> int:
    """Move every object allocated during startup (LiteLLM's model tables, sklearn,
    pydantic schemas: ~375k objects) into the permanent generation so cyclic GC never
    re-scans them. Measured: a full collection drops from ~97 ms to ~0 ms, removing
    periodic event-loop stalls under load. Returns the number of frozen objects."""
    if _gc_callback not in gc.callbacks:
        gc.callbacks.append(_gc_callback)
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()
