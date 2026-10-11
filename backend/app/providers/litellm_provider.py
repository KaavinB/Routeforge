"""LiteLLM-backed provider for OpenAI, Anthropic, Ollama and other compatible endpoints.

LiteLLM supplies provider connectivity and request/response translation. RouteForge
disables LiteLLM's own retries and routing so that retry, fallback and circuit-breaking
behaviour is owned (and measured) by ``app.reliability``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import litellm

from app.core.errors import ErrorKind, ProviderError
from app.providers.base import ProviderRequest, ProviderResponse, StreamChunk

litellm.suppress_debug_info = True
litellm.drop_params = False  # unsupported params are handled explicitly by the gateway
litellm.telemetry = False


def classify_litellm_error(exc: Exception, model_id: str) -> ProviderError:
    """Map LiteLLM / provider SDK exceptions onto RouteForge's error taxonomy."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    message = f"{name}: {str(exc)[:500]}"

    retry_after: float | None = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after")
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = None

    kind: ErrorKind
    if isinstance(exc, litellm.exceptions.ContextWindowExceededError):
        kind = ErrorKind.CONTEXT_LENGTH
    elif isinstance(exc, litellm.exceptions.ContentPolicyViolationError):
        kind = ErrorKind.CONTENT_FILTER
    elif isinstance(exc, litellm.exceptions.Timeout | TimeoutError):
        kind = ErrorKind.TIMEOUT
    elif isinstance(exc, litellm.exceptions.RateLimitError) or status == 429:
        kind = ErrorKind.RATE_LIMITED
    elif isinstance(
        exc,
        litellm.exceptions.AuthenticationError | litellm.exceptions.PermissionDeniedError,
    ) or status in (401, 403):
        kind = ErrorKind.AUTH
    elif isinstance(exc, litellm.exceptions.APIConnectionError):
        kind = ErrorKind.CONNECTION
    elif isinstance(
        exc,
        litellm.exceptions.ServiceUnavailableError | litellm.exceptions.InternalServerError,
    ) or (status is not None and status >= 500):
        kind = ErrorKind.SERVER_ERROR
    elif isinstance(
        exc,
        litellm.exceptions.BadRequestError
        | litellm.exceptions.UnsupportedParamsError
        | litellm.exceptions.NotFoundError,
    ) or status in (400, 404, 422):
        kind = ErrorKind.BAD_REQUEST
    else:
        kind = ErrorKind.UNKNOWN
    return ProviderError(
        kind, message, status_code=status, retry_after_s=retry_after, model_id=model_id
    )


class LiteLLMProvider:
    name = "litellm"

    def _kwargs(self, request: ProviderRequest) -> dict[str, Any]:
        spec = request.spec
        kwargs: dict[str, Any] = {
            "model": spec.upstream,
            "messages": request.messages,
            "timeout": request.timeout_s,
            "num_retries": 0,
            "max_retries": 0,
            **request.params,
            **spec.provider_params,
        }
        if spec.api_base:
            kwargs["api_base"] = spec.api_base
        return kwargs

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        started = time.perf_counter()
        try:
            resp = await litellm.acompletion(**self._kwargs(request))
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_litellm_error(exc, request.spec.id) from exc
        latency = time.perf_counter() - started

        try:
            choice = resp.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            usage = resp.usage
            input_tokens = int(usage.prompt_tokens or 0)
            output_tokens = int(usage.completion_tokens or 0)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderError(
                ErrorKind.INVALID_RESPONSE,
                f"malformed upstream response: {exc}",
                model_id=request.spec.id,
            ) from exc
        if content is None or (content == "" and finish_reason not in ("length", "content_filter")):
            raise ProviderError(
                ErrorKind.INVALID_RESPONSE, "empty completion", model_id=request.spec.id
            )
        if finish_reason == "content_filter":
            raise ProviderError(
                ErrorKind.CONTENT_FILTER, "upstream content filter", model_id=request.spec.id
            )
        return ProviderResponse(
            content=content,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            upstream_model=getattr(resp, "model", None),
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamChunk]:
        kwargs = self._kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise classify_litellm_error(exc, request.spec.id) from exc
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                delta = ""
                finish = None
                if chunk.choices:
                    c0 = chunk.choices[0]
                    delta = (c0.delta.content or "") if c0.delta else ""
                    finish = c0.finish_reason
                yield StreamChunk(
                    delta=delta,
                    finish_reason=finish,
                    input_tokens=int(usage.prompt_tokens) if usage else None,
                    output_tokens=int(usage.completion_tokens) if usage else None,
                )
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_litellm_error(exc, request.spec.id) from exc
