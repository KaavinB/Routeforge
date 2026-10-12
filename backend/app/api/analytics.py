from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics import queries
from app.api.deps import get_session, get_state, require_admin
from app.core.errors import NotFoundError
from app.services.auth import Principal
from app.state import AppState

router = APIRouter(
    prefix="/api/analytics", tags=["analytics"], dependencies=[Depends(require_admin)]
)

Hours = Query(24.0, gt=0, le=24 * 90)
TrafficQ = Query("all", pattern="^(all|real|simulated)$")


@router.get("/summary")
async def summary(
    hours: float = Hours,
    traffic: str = TrafficQ,
    session: AsyncSession = Depends(get_session),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    data = await queries.summary(session, hours, traffic)
    data["mode"] = state.registry.mode
    data["simulated_mode"] = state.registry.simulated
    data["pricing"] = state.registry.data.pricing.model_dump()
    data["router_artifact"] = state.engine.artifact.version if state.engine.artifact else None
    data["default_strategy"] = state.engine.default_strategy
    data["breakers"] = state.breakers.snapshot()
    return data


@router.get("/models")
async def models(
    hours: float = Hours,
    traffic: str = TrafficQ,
    session: AsyncSession = Depends(get_session),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    rows = await queries.models(session, hours, traffic)
    registry = {
        m.id: {
            "family": m.family,
            "tier": m.tier,
            "input_cost_per_mtok": m.input_cost_per_mtok,
            "output_cost_per_mtok": m.output_cost_per_mtok,
            "enabled": m.enabled,
        }
        for m in state.registry.all()
    }
    return {"models": rows, "registry": registry, "breakers": state.breakers.snapshot()}


@router.get("/requests")
async def requests(
    hours: float = Hours,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    model: str | None = None,
    strategy: str | None = None,
    status: str | None = None,
    cache: str | None = None,
    search: str | None = Query(None, max_length=64),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    return await queries.requests(
        session,
        hours=hours,
        limit=limit,
        offset=offset,
        model=model,
        strategy=strategy,
        status=status,
        cache=cache,
        search=search,
    )


@router.get("/requests/{request_id}")
async def request_detail(
    request_id: str, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    detail = await queries.request_detail(session, request_id)
    if detail is None:
        raise NotFoundError("Request not found.")
    return detail


@router.get("/timeseries")
async def timeseries(
    hours: float = Hours,
    buckets: int = Query(48, ge=4, le=500),
    traffic: str = TrafficQ,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    return {"buckets": await queries.timeseries(session, hours, buckets, traffic)}


@router.get("/routing/preview")
async def routing_preview(
    prompt: str = Query(..., min_length=1, max_length=8000),
    strategy: str | None = None,
    quality_threshold: float | None = Query(None, ge=0, le=1),
    latency_budget_ms: float | None = Query(None, gt=0),
    state: AppState = Depends(get_state),
    _: Principal = Depends(require_admin),
) -> dict[str, Any]:
    """Dry-run routing for a prompt without calling any model (dashboard 'why this model')."""
    messages = [{"role": "user", "content": prompt}]
    decision = state.engine.decide(
        strategy or state.engine.default_strategy,
        messages,
        {},
        quality_threshold=quality_threshold
        if quality_threshold is not None
        else state.settings.default_quality_threshold,
        latency_budget_ms=latency_budget_ms,
    )
    ctx = state.engine.build_context(messages, {}, quality_threshold=0.0, latency_budget_ms=None)
    return {**decision.summary(), "task_probs": ctx.task_probs, "features": ctx.features.as_dict()}
