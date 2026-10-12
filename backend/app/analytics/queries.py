"""Analytics aggregations over the request log.

Counts and sums run in SQL. Latency percentiles are computed in Python over the
window's latency column (bounded by ``MAX_PERCENTILE_ROWS``) so the same code works on
PostgreSQL and SQLite; at higher volume this would move to ``percentile_cont`` or be
served from Prometheus histograms.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import Select, and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import RequestLog

MAX_PERCENTILE_ROWS = 200_000


Traffic = str  # "all" | "real" | "simulated"


def _window_filter(hours: float, traffic: Traffic = "all") -> list[Any]:
    since = datetime.now(UTC) - timedelta(hours=hours)
    conds: list[Any] = [RequestLog.created_at >= since]
    # Simulated (mock-provider) requests carry simulated latencies and costs; mixing
    # them with real traffic would make spend and latency figures meaningless.
    if traffic == "real":
        conds.append(RequestLog.simulated.is_(False))
    elif traffic == "simulated":
        conds.append(RequestLog.simulated.is_(True))
    return conds


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    arr = np.asarray(values, dtype=float)
    out: dict[str, float | None] = {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
    # p99 is only meaningful with enough samples; report None rather than a noisy number.
    out["p99"] = float(np.percentile(arr, 99)) if len(arr) >= 200 else None
    return out


async def _latencies(
    session: AsyncSession, conds: list[Any], column: Any = RequestLog.latency_ms
) -> list[float]:
    stmt = select(column).where(and_(*conds), column.is_not(None)).limit(MAX_PERCENTILE_ROWS)
    return [float(v) for (v,) in (await session.execute(stmt)).all()]


async def summary(
    session: AsyncSession, hours: float = 24, traffic: Traffic = "all"
) -> dict[str, Any]:
    conds = _window_filter(hours, traffic)
    success = RequestLog.status == "success"
    row = (
        await session.execute(
            select(
                func.count(),
                func.sum(case((success, 1), else_=0)),
                func.coalesce(func.sum(RequestLog.cost_usd), 0.0),
                func.coalesce(func.sum(RequestLog.cache_saved_usd), 0.0),
                func.sum(case((RequestLog.cache_status == "hit", 1), else_=0)),
                func.sum(case((RequestLog.cache_status.in_(("hit", "miss")), 1), else_=0)),
                func.sum(case((RequestLog.fallback_used.is_(True), 1), else_=0)),
                func.coalesce(func.sum(RequestLog.input_tokens), 0),
                func.coalesce(func.sum(RequestLog.output_tokens), 0),
                func.sum(case((RequestLog.simulated.is_(True), 1), else_=0)),
                func.avg(RequestLog.routing_ms),
            ).where(and_(*conds))
        )
    ).one()
    total, ok, cost, saved, hits, lookups, fallbacks, tin, tout, simulated, routing_ms = row
    total = int(total or 0)
    ok = int(ok or 0)
    lat = _percentiles(await _latencies(session, [*conds, success]))
    strategies = (
        await session.execute(
            select(RequestLog.strategy, func.count())
            .where(and_(*conds))
            .group_by(RequestLog.strategy)
        )
    ).all()
    models = (
        await session.execute(
            select(RequestLog.final_model, func.count())
            .where(and_(*conds), RequestLog.final_model.is_not(None))
            .group_by(RequestLog.final_model)
        )
    ).all()
    return {
        "window_hours": hours,
        "total_requests": total,
        "successful_requests": ok,
        "success_rate": (ok / total) if total else None,
        "estimated_spend_usd": float(cost),
        "cost_per_successful_request_usd": (float(cost) / ok) if ok else None,
        "cache_hits": int(hits or 0),
        "cache_lookups": int(lookups or 0),
        "cache_hit_rate": (int(hits or 0) / int(lookups)) if lookups else None,
        "cache_saved_usd": float(saved),
        "fallback_requests": int(fallbacks or 0),
        "input_tokens": int(tin),
        "output_tokens": int(tout),
        "latency_ms": lat,
        "avg_routing_ms": float(routing_ms) if routing_ms is not None else None,
        "simulated_requests": int(simulated or 0),
        "active_models": sorted(m for m, _ in models if m),
        "strategy_distribution": {s: int(c) for s, c in strategies},
        "model_distribution": {m: int(c) for m, c in models},
        "cost_note": (
            "Estimated from provider-reported token usage x list prices; not billing data."
        ),
    }


async def models(
    session: AsyncSession, hours: float = 24, traffic: Traffic = "all"
) -> list[dict[str, Any]]:
    conds = _window_filter(hours, traffic)
    success = RequestLog.status == "success"
    rows = (
        await session.execute(
            select(
                RequestLog.final_model,
                func.count(),
                func.sum(case((success, 1), else_=0)),
                func.coalesce(func.sum(RequestLog.cost_usd), 0.0),
                func.coalesce(func.sum(RequestLog.input_tokens), 0),
                func.coalesce(func.sum(RequestLog.output_tokens), 0),
                func.sum(case((RequestLog.cache_status == "hit", 1), else_=0)),
                func.avg(RequestLog.provider_ms),
            )
            .where(and_(*conds), RequestLog.final_model.is_not(None))
            .group_by(RequestLog.final_model)
        )
    ).all()
    out = []
    for model, n, ok, cost, tin, tout, hits, prov in rows:
        lat = _percentiles(
            await _latencies(session, [*conds, success, RequestLog.final_model == model])
        )
        n, ok = int(n), int(ok or 0)
        out.append(
            {
                "model": model,
                "requests": n,
                "success_rate": ok / n if n else None,
                "error_rate": (n - ok) / n if n else None,
                "cost_usd": float(cost),
                "input_tokens": int(tin),
                "output_tokens": int(tout),
                "cache_hits": int(hits or 0),
                "avg_provider_ms": float(prov) if prov is not None else None,
                "latency_ms": lat,
            }
        )
    return sorted(out, key=lambda r: -r["requests"])


def _request_query(
    *,
    hours: float,
    model: str | None,
    strategy: str | None,
    status: str | None,
    cache: str | None,
    search: str | None,
) -> Select[Any]:
    conds = _window_filter(hours)
    if model:
        conds.append(RequestLog.final_model == model)
    if strategy:
        conds.append(RequestLog.strategy == strategy)
    if status:
        conds.append(RequestLog.status == status)
    if cache:
        conds.append(RequestLog.cache_status == cache)
    if search:
        conds.append(or_(RequestLog.request_id == search, RequestLog.trace_id == search))
    return select(RequestLog).where(and_(*conds))


_LIST_FIELDS = (
    "request_id",
    "trace_id",
    "created_at",
    "strategy",
    "selected_model",
    "final_model",
    "routing_reason",
    "predicted_task",
    "predicted_quality",
    "cache_status",
    "status",
    "error_type",
    "http_status",
    "attempts",
    "fallback_used",
    "stream",
    "simulated",
    "latency_ms",
    "routing_ms",
    "ttft_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
)


async def requests(
    session: AsyncSession,
    *,
    hours: float = 24,
    limit: int = 50,
    offset: int = 0,
    model: str | None = None,
    strategy: str | None = None,
    status: str | None = None,
    cache: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    base = _request_query(
        hours=hours, model=model, strategy=strategy, status=status, cache=cache, search=search
    )
    total = (await session.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        await session.execute(
            base.order_by(RequestLog.created_at.desc()).limit(limit).offset(offset)
        )
    ).scalars()
    return {
        "total": int(total),
        "items": [{f: getattr(r, f) for f in _LIST_FIELDS} for r in rows],
    }


async def request_detail(session: AsyncSession, request_id: str) -> dict[str, Any] | None:
    row = (
        await session.execute(select(RequestLog).where(RequestLog.request_id == request_id))
    ).scalar_one_or_none()
    if row is None:
        return None
    detail = {f: getattr(row, f) for f in _LIST_FIELDS}
    detail["routing_detail"] = row.routing_detail
    detail["attempt_detail"] = row.attempt_detail
    detail["quality_threshold"] = row.quality_threshold
    detail["queue_ms"] = row.queue_ms
    detail["provider_ms"] = row.provider_ms
    return detail


async def timeseries(
    session: AsyncSession, hours: float = 24, buckets: int = 48, traffic: Traffic = "all"
) -> list[dict[str, Any]]:
    """Fixed-width buckets computed in Python for portability across databases."""
    conds = _window_filter(hours, traffic)
    rows = (
        await session.execute(
            select(
                RequestLog.created_at,
                RequestLog.status,
                RequestLog.latency_ms,
                RequestLog.cost_usd,
                RequestLog.cache_status,
            )
            .where(and_(*conds))
            .limit(MAX_PERCENTILE_ROWS)
        )
    ).all()
    now = datetime.now(UTC)
    start = now - timedelta(hours=hours)
    width = (now - start) / buckets
    agg: list[dict[str, Any]] = [
        {
            "start": (start + width * i).isoformat(),
            "requests": 0,
            "errors": 0,
            "cost_usd": 0.0,
            "cache_hits": 0,
            "_lat": [],
        }
        for i in range(buckets)
    ]
    for created, status, lat, cost, cache_status in rows:
        ts = created if created.tzinfo else created.replace(tzinfo=UTC)
        i = min(int((ts - start) / width), buckets - 1)
        if i < 0:
            continue
        b = agg[i]
        b["requests"] += 1
        b["errors"] += int(status != "success")
        b["cost_usd"] += float(cost or 0)
        b["cache_hits"] += int(cache_status == "hit")
        if status == "success":
            b["_lat"].append(lat)
    for b in agg:
        lats = b.pop("_lat")
        b["p95_latency_ms"] = float(np.percentile(lats, 95)) if lats else None
    return agg
