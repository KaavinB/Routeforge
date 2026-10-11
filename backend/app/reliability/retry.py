"""Retry policy: bounded attempts, exponential backoff with full jitter, and a retry
budget that caps retry amplification.

Why a budget: per-request attempt limits alone still let retries multiply load (and
cost) during a provider incident — every request retries. The budget (token bucket,
Finagle/Envoy style) lets retries consume at most ``ratio`` of recent successful
request volume, plus a small constant floor so low-traffic periods can still retry.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BackoffPolicy:
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    multiplier: float = 2.0

    def delay(self, attempt: int, rng: random.Random, retry_after_s: float | None = None) -> float:
        """Delay before retry number ``attempt`` (1-based). Full jitter:
        uniform(0, min(max, base * mult^(attempt-1))). A provider ``Retry-After`` hint
        is honoured as a lower bound."""
        cap = min(self.max_delay_s, self.base_delay_s * self.multiplier ** (attempt - 1))
        delay = rng.uniform(0, cap)
        if retry_after_s is not None:
            delay = max(delay, retry_after_s)
        return delay


class RetryBudget:
    def __init__(
        self,
        ratio: float = 0.2,
        min_per_s: float = 1.0,
        max_tokens: float = 50.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ratio = ratio
        self.min_per_s = min_per_s
        self.max_tokens = max_tokens
        self._clock = clock
        self._tokens = max_tokens / 2
        self._last = self._clock()
        self._lock = threading.Lock()
        self.denied = 0

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.max_tokens, self._tokens + (now - self._last) * self.min_per_s)
        self._last = now

    def record_success(self) -> None:
        with self._lock:
            self._refill()
            self._tokens = min(self.max_tokens, self._tokens + self.ratio)

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0 - 1e-9:  # tolerate float accumulation error
                self._tokens -= 1.0
                return True
            self.denied += 1
            return False

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens
