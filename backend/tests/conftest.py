from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from app.core.config import REPO_ROOT, Settings
from app.main import create_app
from app.models.db import Database
from app.state import AppState, build_state

ADMIN_KEY = "rf-test-admin-key-0123456789abcdef"
CLIENT_KEY = "rf-test-client-key-0123456789abcdef"
MOCK_REGISTRY = REPO_ROOT / "config" / "models" / "mock.yaml"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "env": "test",
        "log_json": False,
        "log_level": "WARNING",
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        "redis_url": "",
        "model_config_path": MOCK_REGISTRY,
        "mock_latency_scale": 0.0,
        "bootstrap_admin_key": ADMIN_KEY,
        "bootstrap_client_key": CLIENT_KEY,
        "otel_enabled": True,
        "langfuse_enabled": False,
        "retry_base_delay_s": 0.0,
        "retry_max_delay_s": 0.01,
        "fault_injection_enabled": True,
        "default_strategy": "heuristic",
    }
    base.update(overrides)
    return Settings(**base)


@dataclass
class Harness:
    app: FastAPI
    state: AppState
    client: httpx.AsyncClient
    redis: Any

    @property
    def admin(self) -> dict[str, str]:
        return {"authorization": f"Bearer {ADMIN_KEY}"}

    @property
    def user(self) -> dict[str, str]:
        return {"authorization": f"Bearer {CLIENT_KEY}"}

    async def chat(
        self,
        content: str = "What is the capital of France?",
        headers: dict[str, str] | None = None,
        **body: Any,
    ) -> httpx.Response:
        payload: dict[str, Any] = {
            "model": "routeforge-auto",
            "messages": [{"role": "user", "content": content}],
        }
        payload.update(body)
        return await self.client.post(
            "/v1/chat/completions", json=payload, headers={**self.user, **(headers or {})}
        )


async def make_harness(
    tmp_path: Path, *, with_redis: bool = True, **overrides: Any
) -> AsyncIterator[Harness]:
    settings = make_settings(tmp_path, **overrides)
    db = Database(settings.database_url)
    await db.create_all()
    redis = fakeredis.aioredis.FakeRedis() if with_redis else None
    state = build_state(settings, db=db, redis=redis)
    if not with_redis:
        state.redis = None
    app = create_app(settings, state=state)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(app=app, state=state, client=client, redis=redis)


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    async for h in make_harness(tmp_path):
        yield h
