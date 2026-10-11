from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.exceptions import RedisError

from app.api.deps import get_state
from app.state import AppState
from app.telemetry import metrics

router = APIRouter(tags=["operations"])


@router.get("/health")
async def health(state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Liveness: the process is up and serving."""
    return {"status": "ok", "version": state.settings.version}


@router.get("/ready")
async def ready(response: Response, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Readiness. The gateway can serve inference without Redis (cache/rate-limit degrade)
    and without PostgreSQL (logging degrades; bootstrap keys still authenticate), so those
    report ``degraded`` rather than failing readiness."""
    components: dict[str, str] = {}
    components["database"] = (
        "ok" if state.db is not None and await state.db.ping() else "unavailable"
    )
    try:
        components["redis"] = (
            "ok" if state.redis is not None and await state.redis.ping() else "unavailable"
        )
    except (RedisError, OSError, TimeoutError):
        components["redis"] = "unavailable"
    components["models"] = f"{len(state.registry.enabled())} enabled ({state.registry.mode})"
    components["router_artifact"] = (
        state.engine.artifact.version if state.engine.artifact else "none"
    )
    available = [m.id for m in state.registry.enabled() if state.breakers.is_available(m.id)]
    ok = bool(available)
    status = (
        "ready"
        if ok and all(components[c] == "ok" for c in ("database", "redis"))
        else ("degraded" if ok else "unavailable")
    )
    if not ok:
        response.status_code = 503
    return {
        "status": status,
        "components": components,
        "available_models": available,
        "default_strategy": state.engine.default_strategy,
        "simulated": state.registry.simulated,
    }


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request, state: AppState = Depends(get_state)) -> Response:
    if state.settings.metrics_require_auth:
        from app.api.deps import get_principal, require_admin

        await require_admin(await get_principal(request, state))
    metrics.IN_FLIGHT.set(state.gateway.concurrency.in_flight)
    return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)
