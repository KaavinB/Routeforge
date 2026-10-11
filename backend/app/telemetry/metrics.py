"""Prometheus metrics.

Label cardinality is bounded by design: labels are drawn only from the model registry,
the strategy list, and small fixed enums (status, cache result, error kind). Request
IDs, API keys and prompt content are never labels; use trace IDs for per-request
investigation.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    5,
    7.5,
    10,
    15,
    20,
    30,
    60,
)
_FAST_BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)

REQUESTS = Counter(
    "routeforge_requests_total",
    "Chat completion requests by outcome",
    ["strategy", "model", "status", "cache"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "routeforge_request_duration_seconds",
    "End-to-end gateway latency for chat completions",
    ["strategy", "model", "stream"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
ROUTING_LATENCY = Histogram(
    "routeforge_routing_decision_seconds",
    "Time spent deciding which model to use (feature extraction + strategy)",
    ["strategy"],
    buckets=_FAST_BUCKETS,
    registry=REGISTRY,
)
PROVIDER_LATENCY = Histogram(
    "routeforge_provider_duration_seconds",
    "Upstream provider call latency per attempt",
    ["family", "model", "outcome"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
TTFT = Histogram(
    "routeforge_time_to_first_token_seconds",
    "Time to first streamed token as seen by the gateway",
    ["model"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
QUEUE_WAIT = Histogram(
    "routeforge_concurrency_wait_seconds",
    "Time requests waited for a concurrency slot",
    buckets=(*_FAST_BUCKETS, 0.25, 0.5, 1, 2),
    registry=REGISTRY,
)
TOKENS = Counter(
    "routeforge_tokens_total",
    "Tokens processed (provider-reported usage)",
    ["model", "direction"],
    registry=REGISTRY,
)
COST = Counter(
    "routeforge_estimated_cost_usd_total",
    "Estimated spend from token usage x list price (not billing data)",
    ["model", "strategy"],
    registry=REGISTRY,
)
CACHE_SAVED_COST = Counter(
    "routeforge_cache_saved_cost_usd_total",
    "Estimated spend avoided by cache hits",
    ["model"],
    registry=REGISTRY,
)
PROVIDER_ERRORS = Counter(
    "routeforge_provider_errors_total",
    "Upstream errors by kind",
    ["family", "model", "kind"],
    registry=REGISTRY,
)
RETRIES = Counter(
    "routeforge_retries_total",
    "Retries performed against the same model",
    ["model", "kind"],
    registry=REGISTRY,
)
RETRY_BUDGET_EXHAUSTED = Counter(
    "routeforge_retry_budget_exhausted_total",
    "Retries skipped because the retry budget was exhausted",
    registry=REGISTRY,
)
FALLBACKS = Counter(
    "routeforge_fallbacks_total",
    "Requests that moved to another model after a failure",
    ["from_model", "to_model", "outcome"],
    registry=REGISTRY,
)
BREAKER_STATE = Gauge(
    "routeforge_circuit_breaker_state",
    "Circuit breaker state per model (0=closed, 1=half_open, 2=open)",
    ["model"],
    registry=REGISTRY,
)
CACHE_LOOKUPS = Counter(
    "routeforge_cache_lookups_total",
    "Response cache lookups by result",
    ["result"],
    registry=REGISTRY,
)
CACHE_LATENCY = Histogram(
    "routeforge_cache_lookup_seconds",
    "Response cache lookup latency",
    buckets=_FAST_BUCKETS,
    registry=REGISTRY,
)
IN_FLIGHT = Gauge(
    "routeforge_in_flight_requests", "Requests currently being processed", registry=REGISTRY
)
RATE_LIMITED = Counter(
    "routeforge_rate_limited_total",
    "Requests rejected by rate or concurrency limits",
    ["scope"],
    registry=REGISTRY,
)
DEGRADED = Counter(
    "routeforge_degraded_operations_total",
    "Operations served in degraded mode because a dependency was unavailable",
    ["component"],
    registry=REGISTRY,
)
REQUEST_LOG_DROPPED = Counter(
    "routeforge_request_log_dropped_total",
    "Request log records dropped (queue full or database unavailable)",
    ["reason"],
    registry=REGISTRY,
)
PREDICTED_QUALITY = Histogram(
    "routeforge_selected_predicted_quality",
    "Router-predicted P(acceptable) of the selected model",
    ["strategy"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)
STREAM_ABORTS = Counter(
    "routeforge_stream_aborts_total",
    "Streams terminated by an upstream error after the first token",
    ["model"],
    registry=REGISTRY,
)
EVENT_LOOP_LAG = Histogram(
    "routeforge_event_loop_lag_seconds",
    "Scheduling delay of a 100 ms heartbeat task (time the event loop was blocked)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
    registry=REGISTRY,
)
GC_PAUSE = Histogram(
    "routeforge_gc_pause_seconds",
    "Duration of CPython cyclic garbage-collection passes (they block the event loop)",
    ["generation"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    registry=REGISTRY,
)
