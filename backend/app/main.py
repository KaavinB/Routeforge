"""FastAPI application factory."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from threadpoolctl import threadpool_limits

from app.api import admin, analytics, benchmarks, chat, evaluations, health
from app.core.config import Settings, get_settings
from app.core.errors import GatewayError
from app.core.logging import configure_logging, request_id_var
from app.state import AppState, build_state
from app.telemetry.loop_monitor import LoopLagMonitor, freeze_startup_heap
from app.telemetry.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger("routeforge")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def _error_body(
    message: str, error_type: str, code: str | None, param: str | None, request_id: str | None
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "param": param,
            "request_id": request_id,
        }
    }


class BodySizeLimitMiddleware:
    """Rejects request bodies above ``max_bytes`` (also when Content-Length is absent)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return
        received = 0
        too_large = False

        async def limited_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    raise _BodyTooLargeError()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLargeError:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = JSONResponse(
            _error_body(
                "Request body too large.", "invalid_request_error", "request_too_large", None, None
            ),
            status_code=413,
        )
        await send({"type": "http.response.start", "status": 413, "headers": body.raw_headers})
        await send({"type": "http.response.body", "body": body.body})


class _BodyTooLargeError(Exception):
    pass


def create_app(settings: Settings | None = None, state: AppState | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Router inference is single-row; multi-threaded BLAS/OpenMP pools only add
        # thread-spawn overhead and oversubscribe CPUs shared with the event loop.
        threadpool_limits(1)
        if settings.otel_enabled:
            setup_tracing(settings.service_name, settings.version, settings.otel_exporter_endpoint)
        app_state = state or build_state(settings)
        app.state.rf = app_state
        app_state.log_writer.start()
        loop_monitor = LoopLagMonitor()
        loop_monitor.start()
        if settings.gc_freeze_after_startup:
            frozen = freeze_startup_heap()
            logger.info("froze startup heap", extra={"frozen_objects": frozen})
        logger.info(
            "routeforge started",
            extra={
                "mode": app_state.registry.mode,
                "models": [m.id for m in app_state.registry.enabled()],
                "default_strategy": app_state.engine.default_strategy,
                "router_artifact": app_state.engine.artifact.version
                if app_state.engine.artifact
                else None,
            },
        )
        try:
            yield
        finally:
            await loop_monitor.stop()
            await app_state.close()
            shutdown_tracing()

    app = FastAPI(
        title="RouteForge",
        version=settings.version,
        description=(
            "Quality-aware LLM gateway with adaptive routing, continuous evaluation "
            "and observability."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "authorization",
            "content-type",
            "x-api-key",
            "x-request-id",
            "x-routeforge-cache",
            "x-routeforge-no-store",
        ],
        expose_headers=[
            "x-request-id",
            "x-routeforge-model",
            "x-routeforge-strategy",
            "x-routeforge-cache",
            "x-routeforge-cost-usd",
            "x-routeforge-trace-id",
            "x-routeforge-fallback",
        ],
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers.setdefault("x-request-id", rid)
        return response

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            _error_body(exc.message, exc.error_type, exc.code, exc.param, rid),
            status_code=exc.status_code,
            headers={**exc.headers, **({"x-request-id": rid} if rid else {})},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg", "Invalid request")
        if first.get("type") == "extra_forbidden":
            msg = f"Unsupported parameter '{loc}'. RouteForge rejects parameters it cannot honour."
        return JSONResponse(
            _error_body(
                f"{msg} ({loc})" if loc else msg,
                "invalid_request_error",
                first.get("type"),
                loc or None,
                rid,
            ),
            status_code=400,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        logger.exception("unhandled error")
        return JSONResponse(
            _error_body("Internal server error.", "server_error", None, None, rid), status_code=500
        )

    for r in (
        health.router,
        chat.router,
        analytics.router,
        evaluations.router,
        benchmarks.router,
        admin.router,
    ):
        app.include_router(r)
    return app
