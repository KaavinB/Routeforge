"""Asynchronous, batched request-log writer.

Requests never wait on the database: records go onto a bounded in-memory queue and a
background task inserts them in batches. If the queue is full or the database is
unavailable, records are dropped and counted (``routeforge_request_log_dropped_total``)
— serving traffic takes priority over analytics completeness, and Prometheus metrics
(which are updated synchronously) remain accurate during a database outage.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError

from app.models.db import Database, RequestLog
from app.telemetry import metrics

logger = logging.getLogger(__name__)


class RequestLogWriter:
    def __init__(
        self, db: Database | None, *, queue_size: int = 10_000, batch_size: int = 200
    ) -> None:
        self.db = db
        self.batch_size = batch_size
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self.written = 0

    def submit(self, record: dict[str, Any]) -> None:
        if self.db is None:
            return
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            metrics.REQUEST_LOG_DROPPED.labels("queue_full").inc()

    def start(self) -> None:
        if self.db is not None and self._task is None:
            self._task = asyncio.create_task(self._run(), name="request-log-writer")

    async def _run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            # Small linger to accumulate a batch under load without delaying idle writes.
            await asyncio.sleep(0.05)
            while len(batch) < self.batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            try:
                await self._write(batch)
            finally:
                for _ in batch:
                    self.queue.task_done()

    async def _write(self, batch: list[dict[str, Any]]) -> None:
        assert self.db is not None
        for attempt in range(2):
            try:
                async with self.db.sessionmaker() as session:
                    await session.execute(insert(RequestLog), batch)
                    await session.commit()
                self.written += len(batch)
                return
            except (SQLAlchemyError, OSError, TimeoutError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                logger.warning("dropping %d request logs: %s", len(batch), type(exc).__name__)
                metrics.REQUEST_LOG_DROPPED.labels("db_error").inc(len(batch))
                metrics.DEGRADED.labels("database").inc()

    async def flush(self, timeout_s: float = 5.0) -> None:
        """Wait until every submitted record has been written (or dropped). Used at
        shutdown and by tests/benchmarks that read analytics right after sending traffic."""
        if self._task is not None and not self._task.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.queue.join(), timeout=timeout_s)
            return
        while not self.queue.empty():
            batch: list[dict[str, Any]] = []
            while len(batch) < self.batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            await self._write(batch)
            for _ in batch:
                self.queue.task_done()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.flush()
