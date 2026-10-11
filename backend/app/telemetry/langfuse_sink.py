"""Langfuse (v4 SDK) integration for LLM-centric traces.

Enabled only when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are present. The
client gets its own TracerProvider so it never hijacks the gateway's global
OpenTelemetry pipeline, and each generation is attached to the gateway's trace id, so a
trace id from a response header finds the same request in Jaeger and Langfuse.

Prompt/completion text is sent only when ``ROUTEFORGE_LOG_PROMPTS=true``; otherwise
Langfuse receives model, usage, cost, latency and routing metadata only.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class LangfuseSink:
    def __init__(self, *, enabled: bool, environment: str, release: str) -> None:
        self._client: Any = None
        if not enabled or not (
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        ):
            return
        try:
            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider

            self._client = Langfuse(
                tracer_provider=TracerProvider(), environment=environment, release=release
            )
            logger.info("Langfuse tracing enabled")
        except Exception as exc:
            logger.warning("Langfuse disabled: %s", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def record_generation(
        self,
        *,
        trace_id: str | None,
        name: str,
        model: str,
        input_messages: list[dict[str, str]] | None,
        output: str | None,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        completion_start_time: datetime | None,
        metadata: dict[str, Any],
        error: str | None = None,
    ) -> None:
        if self._client is None:
            return
        try:
            trace_context = {"trace_id": trace_id} if trace_id else None
            generation = self._client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="generation",
                model=model,
                input=input_messages,
                output=output,
                metadata=metadata,
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"total": cost_usd},
                completion_start_time=completion_start_time,
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
            generation.end()
        except Exception as exc:
            logger.debug("langfuse record failed: %s", exc)

    def score(self, *, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        if self._client is None:
            return
        try:
            self._client.create_score(name=name, value=value, trace_id=trace_id, comment=comment)
        except Exception as exc:
            logger.debug("langfuse score failed: %s", exc)

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.shutdown()
            except Exception:
                logger.debug("langfuse shutdown failed", exc_info=True)
