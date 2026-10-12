"""Integration tests against real PostgreSQL and Redis.

    docker compose up -d postgres redis
    ROUTEFORGE_IT_PG_ADMIN_URL=postgresql+asyncpg://routeforge_owner:<pw>@localhost:5432/postgres \\
    ROUTEFORGE_IT_REDIS_URL=redis://localhost:6379/15 uv run pytest -m integration

CI provides both as service containers. Each run creates a fresh database, applies the
Alembic migrations (the real schema path, not ``create_all``), and flushes a dedicated
Redis DB.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.main import create_app
from app.models.db import Database
from app.reliability.rate_limit import RateLimiter
from app.state import build_state
from tests.conftest import ADMIN_KEY, CLIENT_KEY, make_settings

pytestmark = pytest.mark.integration

PG_ADMIN = os.environ.get("ROUTEFORGE_IT_PG_ADMIN_URL", "")
REDIS_URL = os.environ.get("ROUTEFORGE_IT_REDIS_URL", "redis://localhost:6379/15")
BACKEND = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def database_url() -> str:
    if not PG_ADMIN:
        pytest.skip("ROUTEFORGE_IT_PG_ADMIN_URL not set")
    db_name = "routeforge_it"

    async def recreate() -> None:
        engine = create_async_engine(PG_ADMIN, isolation_level="AUTOCOMMIT")
        async with engine.connect() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE)"))
            await conn.execute(text(f"CREATE DATABASE {db_name}"))
        await engine.dispose()

    asyncio.run(recreate())
    url = PG_ADMIN.rsplit("/", 1)[0] + f"/{db_name}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    os.environ["ROUTEFORGE_MIGRATION_DATABASE_URL"] = url
    command.upgrade(cfg, "head")
    return url


async def _harness(
    tmp_path: Path, database_url: str, **overrides: Any
) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    settings = make_settings(tmp_path, database_url=database_url, redis_url=REDIS_URL, **overrides)
    state = build_state(settings, db=Database(database_url), redis=Redis.from_url(REDIS_URL))
    app = create_app(settings, state=state)
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://it") as client,
    ):
        yield state, client


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    r = Redis.from_url(REDIS_URL)
    try:
        await r.ping()
    except Exception:
        pytest.skip("redis not reachable")
    await r.flushdb()
    yield r
    await r.aclose()


ADMIN = {"authorization": f"Bearer {ADMIN_KEY}"}
USER = {"authorization": f"Bearer {CLIENT_KEY}"}


def body(content: str, **kw: Any) -> dict[str, Any]:
    return {"model": "routeforge-auto", "messages": [{"role": "user", "content": content}], **kw}


async def test_request_logging_and_analytics_on_postgres(
    tmp_path: Path, database_url: str, redis: Redis
) -> None:
    async for state, client in _harness(tmp_path, database_url):
        for i in range(12):
            r = await client.post("/v1/chat/completions", json=body(f"q{i}"), headers=USER)
            assert r.status_code == 200
        await state.log_writer.flush()
        s = (await client.get("/api/analytics/summary", headers=ADMIN)).json()
        assert s["total_requests"] >= 12
        assert s["latency_ms"]["p50"] is not None
        ts = (await client.get("/api/analytics/timeseries?hours=1&buckets=6", headers=ADMIN)).json()
        assert sum(b["requests"] for b in ts["buckets"]) >= 12
        ready = (await client.get("/ready")).json()
        assert ready["status"] == "ready"


async def test_redis_rate_limiter_is_atomic_under_concurrency(redis: Redis) -> None:
    limiter = RateLimiter(redis)
    results = await asyncio.gather(*(limiter.check("burst-key", rpm=20) for _ in range(60)))
    assert sum(r.allowed for r in results) == 20  # exactly the bucket capacity, no races


async def test_cache_is_shared_between_gateway_replicas(
    tmp_path: Path, database_url: str, redis: Redis
) -> None:
    async for _, a in _harness(tmp_path / "a", database_url):
        async for _, b in _harness(tmp_path / "b", database_url):
            first = await a.post(
                "/v1/chat/completions",
                json=body("shared cache", temperature=0, model="mock-small"),
                headers=USER,
            )
            second = await b.post(
                "/v1/chat/completions",
                json=body("shared cache", temperature=0, model="mock-small"),
                headers=USER,
            )
            assert first.headers["x-routeforge-cache"] == "miss"
            assert second.headers["x-routeforge-cache"] == "hit"
            flushed = (await b.delete("/api/admin/cache", headers=ADMIN)).json()
            assert flushed["deleted"] >= 1


async def test_db_backed_api_keys(tmp_path: Path, database_url: str, redis: Redis) -> None:
    async for _, client in _harness(tmp_path, database_url):
        created = (
            await client.post(
                "/api/admin/keys", json={"name": "it", "tenant": "acme"}, headers=ADMIN
            )
        ).json()
        key_headers = {"authorization": f"Bearer {created['key']}"}
        r = await client.post("/v1/chat/completions", json=body("hello"), headers=key_headers)
        assert r.status_code == 200
        await client.delete(f"/api/admin/keys/{created['id']}", headers=ADMIN)
        # revocation clears the in-process auth cache immediately
        r = await client.post("/v1/chat/completions", json=body("hello"), headers=key_headers)
        assert r.status_code == 401


async def test_evaluation_run_lifecycle(tmp_path: Path, database_url: str, redis: Redis) -> None:
    async for _, client in _harness(tmp_path, database_url, eval_sandbox_backend="subprocess"):
        r = await client.post(
            "/api/evaluations/run",
            json={
                "targets": ["routeforge-heuristic"],
                "categories": ["classification", "json"],
                "limit": 6,
            },
            headers=ADMIN,
        )
        assert r.status_code == 202, r.text
        run_id = r.json()["run_ids"][0]
        for _ in range(100):
            runs = (await client.get("/api/evaluations/results", headers=ADMIN)).json()["runs"]
            run = next(x for x in runs if x["id"] == run_id)
            if run["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.1)
        assert run["status"] == "done", run.get("error")
        assert run["summary"]["overall"]["n"] == 6
        assert run["summary"]["overall"]["simulated"] is True  # mock answers are flagged
        cases = (await client.get(f"/api/evaluations/results/{run_id}", headers=ADMIN)).json()[
            "cases"
        ]
        assert len(cases) == 6
