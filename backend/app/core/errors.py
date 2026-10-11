"""Error taxonomy.

Two families:

* ``ProviderError`` — something went wrong talking to an upstream model. Carries an
  ``ErrorKind`` that the reliability layer uses to decide retry / fallback / breaker
  behaviour. Never returned to clients verbatim (messages may contain upstream detail).
* ``GatewayError`` — a client-facing error rendered in the OpenAI error envelope.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CONNECTION = "connection"
    INVALID_RESPONSE = "invalid_response"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


# Transient failures worth retrying on the same model.
RETRYABLE_KINDS = frozenset(
    {
        ErrorKind.TIMEOUT,
        ErrorKind.RATE_LIMITED,
        ErrorKind.SERVER_ERROR,
        ErrorKind.CONNECTION,
        ErrorKind.INVALID_RESPONSE,
    }
)
# Failures where another model may succeed even though this one will not.
FALLBACKABLE_KINDS = RETRYABLE_KINDS | {
    ErrorKind.AUTH,  # provider credential problem is ours, not the client's
    ErrorKind.CONTEXT_LENGTH,
    ErrorKind.CIRCUIT_OPEN,
    ErrorKind.UNKNOWN,
}
# Failures attributable to provider health; these feed the circuit breaker.
BREAKER_KINDS = RETRYABLE_KINDS | {ErrorKind.AUTH}


class ProviderError(Exception):
    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.model_id = model_id

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    @property
    def fallbackable(self) -> bool:
        return self.kind in FALLBACKABLE_KINDS

    def __repr__(self) -> str:
        return f"ProviderError({self.kind}, status={self.status_code}, model={self.model_id})"


class GatewayError(Exception):
    """Client-facing error, rendered as ``{"error": {...}}`` (OpenAI-compatible)."""

    status_code: int = 500
    error_type: str = "server_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = headers or {}


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "authentication_error"


class PermissionDeniedError(GatewayError):
    status_code = 403
    error_type = "permission_error"


class NotFoundError(GatewayError):
    status_code = 404
    error_type = "not_found_error"


class RateLimitExceededError(GatewayError):
    status_code = 429
    error_type = "rate_limit_error"


class UpstreamUnavailableError(GatewayError):
    """All candidate models failed or were unavailable."""

    status_code = 503
    error_type = "upstream_unavailable"


class UpstreamTimeoutError(GatewayError):
    status_code = 504
    error_type = "upstream_timeout"


def gateway_error_from_provider(err: ProviderError) -> GatewayError:
    """Map a final (unrecovered) provider error to a sanitized client error."""
    if err.kind == ErrorKind.BAD_REQUEST:
        return InvalidRequestError(
            "The upstream model rejected the request as invalid.", code="upstream_bad_request"
        )
    if err.kind == ErrorKind.CONTEXT_LENGTH:
        return InvalidRequestError(
            "The request exceeds the context window of every eligible model.",
            code="context_length_exceeded",
        )
    if err.kind == ErrorKind.CONTENT_FILTER:
        return InvalidRequestError(
            "The upstream model declined to answer this request.", code="content_filtered"
        )
    if err.kind == ErrorKind.TIMEOUT:
        return UpstreamTimeoutError("Upstream model timed out.", code="upstream_timeout")
    if err.kind == ErrorKind.RATE_LIMITED:
        return UpstreamUnavailableError(
            "Upstream models are rate limited; retry later.", code="upstream_rate_limited"
        )
    return UpstreamUnavailableError(
        "No upstream model could serve the request.", code=f"upstream_{err.kind.value}"
    )
