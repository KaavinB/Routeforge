"""OpenTelemetry tracing setup.

Spans are always created (so every request gets a W3C trace id that is returned to the
client and stored in the request log). Spans are exported over OTLP/HTTP only when an
endpoint is configured (``ROUTEFORGE_OTEL_EXPORTER_ENDPOINT`` or the standard
``OTEL_EXPORTER_OTLP_ENDPOINT``), e.g. to the Jaeger instance in the observability profile.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None


def setup_tracing(service_name: str, version: str, endpoint: str | None) -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": version})
    )
    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
        )
        logger.info("OTLP trace export enabled -> %s", endpoint)
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def shutdown_tracing() -> None:
    if _provider is not None:
        _provider.shutdown()


def tracer() -> trace.Tracer:
    return trace.get_tracer("routeforge")


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")
