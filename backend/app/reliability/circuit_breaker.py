"""Per-model circuit breaker (closed -> open -> half-open -> closed).

Opens when either ``failure_threshold`` consecutive failures occur, or the failure rate
over the last ``window`` calls reaches ``failure_rate`` (with at least ``min_calls``
samples). While open, the model is ineligible for routing so traffic shifts to healthy
models without paying a timeout per request. After ``reset_timeout_s`` one probe request
is allowed (half-open); success closes the breaker, failure re-opens it.

State is per process. With several gateway replicas each learns provider health
independently, which is acceptable because detection takes only a handful of calls.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


STATE_VALUE = {BreakerState.CLOSED: 0, BreakerState.HALF_OPEN: 1, BreakerState.OPEN: 2}


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        failure_rate: float = 0.5,
        window: int = 20,
        min_calls: int = 10,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        on_state_change: Callable[[str, BreakerState], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_rate = failure_rate
        self.min_calls = min_calls
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._on_change = on_state_change
        self._results: deque[bool] = deque(maxlen=window)
        self._consecutive_failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._lock = threading.Lock()
        self.opened_count = 0

    def _set(self, state: BreakerState) -> None:
        if state != self._state:
            self._state = state
            if state == BreakerState.OPEN:
                self._opened_at = self._clock()
                self.opened_count += 1
            if self._on_change:
                self._on_change(self.name, state)

    @property
    def state(self) -> BreakerState:
        with self._lock:
            if (
                self._state == BreakerState.OPEN
                and self._clock() - self._opened_at >= self.reset_timeout_s
            ):
                self._set(BreakerState.HALF_OPEN)
                self._probe_in_flight = False
            return self._state

    def is_available(self) -> bool:
        """Routing eligibility: closed, or half-open with no probe currently running."""
        state = self.state
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.HALF_OPEN:
            with self._lock:
                return not self._probe_in_flight
        return False

    def try_acquire(self) -> bool:
        """Called immediately before an upstream call."""
        state = self.state
        if state == BreakerState.CLOSED:
            return True
        if state == BreakerState.HALF_OPEN:
            with self._lock:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
                return True
        return False

    def record_success(self) -> None:
        with self._lock:
            self._results.append(True)
            self._consecutive_failures = 0
            if self._state == BreakerState.HALF_OPEN:
                self._probe_in_flight = False
                self._results.clear()
                self._set(BreakerState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._results.append(False)
            self._consecutive_failures += 1
            if self._state == BreakerState.HALF_OPEN:
                self._probe_in_flight = False
                self._set(BreakerState.OPEN)
                return
            failures = self._results.count(False)
            rate_trip = (
                len(self._results) >= self.min_calls
                and failures / len(self._results) >= self.failure_rate
            )
            if self._consecutive_failures >= self.failure_threshold or rate_trip:
                self._set(BreakerState.OPEN)

    def release_probe(self) -> None:
        """Release a half-open probe slot without recording an outcome (e.g. the call
        failed for a client-side reason that says nothing about provider health)."""
        with self._lock:
            self._probe_in_flight = False

    def reset(self) -> None:
        with self._lock:
            self._results.clear()
            self._consecutive_failures = 0
            self._probe_in_flight = False
            self._set(BreakerState.CLOSED)


class BreakerRegistry:
    def __init__(self, factory: Callable[[str], CircuitBreaker]) -> None:
        self._factory = factory
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str) -> CircuitBreaker:
        with self._lock:
            if model_id not in self._breakers:
                self._breakers[model_id] = self._factory(model_id)
            return self._breakers[model_id]

    def is_available(self, model_id: str) -> bool:
        return self.get(model_id).is_available()

    def snapshot(self) -> dict[str, dict[str, object]]:
        with self._lock:
            items = list(self._breakers.items())
        return {mid: {"state": b.state.value, "opened_count": b.opened_count} for mid, b in items}

    def reset_all(self) -> None:
        with self._lock:
            items = list(self._breakers.values())
        for b in items:
            b.reset()
