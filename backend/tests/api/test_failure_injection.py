"""Failure-injection tests: provider faults, dependency outages, partial streams."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from app.reliability.circuit_breaker import BreakerState
from tests.conftest import Harness, make_harness


async def _fault(h: Harness, model: str, **rule: object) -> None:
    r = await h.client.put(f"/api/admin/faults/{model}", json=rule, headers=h.admin)
    assert r.status_code == 200, r.text


async def test_http_500_retries_then_succeeds_or_falls_back(harness: Harness) -> None:
    await _fault(harness, "mock-premium", mode="http_500")
    r = await harness.chat(model="routeforge-premium")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] != "mock-premium"
    assert body["routeforge"]["fallback_used"] is True
    assert r.headers["x-routeforge-fallback"] == "true"


async def test_explicit_model_never_falls_back(harness: Harness) -> None:
    await _fault(harness, "mock-premium", mode="http_500")
    r = await harness.chat(model="mock-premium")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "upstream_server_error"


async def test_timeout_falls_back(tmp_path: Path) -> None:
    async for h in make_harness(tmp_path):
        spec = h.state.registry.require("mock-premium")
        object.__setattr__(spec, "timeout_s", 0.2)
        await _fault(h, "mock-premium", mode="timeout")
        r = await h.chat(model="routeforge-premium")
        assert r.status_code == 200
        assert r.json()["routeforge"]["fallback_used"] is True


async def test_429_and_invalid_response_are_recovered(harness: Harness) -> None:
    await _fault(harness, "mock-economy", mode="http_429", probability=1.0)
    r = await harness.chat(model="routeforge-economy")
    assert r.status_code == 200 and r.json()["model"] != "mock-economy"
    await _fault(harness, "mock-economy", mode="invalid_response")
    r = await harness.chat(model="routeforge-economy")
    assert r.status_code == 200 and r.json()["routeforge"]["fallback_used"]


async def test_outage_opens_circuit_breaker_and_routing_avoids_model(harness: Harness) -> None:
    await _fault(harness, "mock-premium", mode="outage")
    for _ in range(4):
        await harness.chat(model="routeforge-premium")
    breaker = harness.state.breakers.get("mock-premium")
    assert breaker.state == BreakerState.OPEN
    r = await harness.chat(model="routeforge-premium")
    # Routing now treats the premium model as ineligible: no upstream attempt is wasted on it.
    assert r.status_code == 200
    detail_id = r.json()["routeforge"]["request_id"]
    await harness.state.log_writer.flush()
    detail = (
        await harness.client.get(f"/api/analytics/requests/{detail_id}", headers=harness.admin)
    ).json()
    assert all(a["model_id"] != "mock-premium" for a in detail["attempt_detail"])
    assert "fixed_target_unavailable" in detail["routing_reason"]
    breakers = (await harness.client.get("/api/admin/breakers", headers=harness.admin)).json()
    assert breakers["breakers"]["mock-premium"]["state"] == "open"


async def test_all_models_failing_returns_503(harness: Harness) -> None:
    for m in ("mock-economy", "mock-small", "mock-medium", "mock-premium"):
        await _fault(harness, m, mode="http_500")
    r = await harness.chat()
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["type"] == "upstream_unavailable"
    assert "injected" not in err["message"]  # upstream detail is not leaked


async def test_retry_amplification_is_bounded(harness: Harness) -> None:
    for m in ("mock-economy", "mock-small", "mock-medium", "mock-premium"):
        await _fault(harness, m, mode="http_500")
    await harness.chat(model="routeforge-premium")
    await harness.state.log_writer.flush()
    items = (await harness.client.get("/api/analytics/requests", headers=harness.admin)).json()[
        "items"
    ]
    s = harness.state.settings
    max_calls = (1 + s.max_fallbacks) * s.retry_max_attempts_per_model
    assert items[0]["attempts"] <= max_calls


async def test_redis_outage_degrades_gracefully(tmp_path: Path) -> None:
    async for h in make_harness(tmp_path):

        async def broken(*_a: object, **_k: object) -> None:
            raise ConnectionError("redis down")

        h.redis.get = broken
        h.redis.set = broken
        h.redis.evalsha = broken
        h.redis.eval = broken
        r = await h.chat(temperature=0)
        assert r.status_code == 200
        assert r.headers["x-routeforge-cache"] in ("miss", "eligible")


async def test_database_outage_still_serves(tmp_path: Path) -> None:
    async for h in make_harness(tmp_path):
        # Point the session factory at an unreachable database.
        bad = create_async_engine(
            "postgresql+asyncpg://nobody:x@127.0.0.1:1/none", connect_args={"timeout": 0.2}
        )
        h.state.db.sessionmaker.configure(bind=bad)  # type: ignore[union-attr]
        r = await h.chat()  # bootstrap keys authenticate without the DB
        assert r.status_code == 200
        await h.state.log_writer.flush(timeout_s=3)
        ready = (await h.client.get("/ready")).json()
        assert ready["status"] in ("ready", "degraded")
        await bad.dispose()


async def test_stream_abort_after_first_chunk(harness: Harness) -> None:
    await _fault(harness, "mock-economy", mode="stream_abort")
    payload = {
        "model": "routeforge-economy",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    async with harness.client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=harness.user
    ) as r:
        assert r.status_code == 200
        events = [line[6:] async for line in r.aiter_lines() if line.startswith("data: ")]
    assert "[DONE]" not in events
    assert json.loads(events[-1])["error"]["code"] == "stream_aborted"


async def test_stream_failure_before_first_token_falls_back(harness: Harness) -> None:
    await _fault(harness, "mock-economy", mode="http_500")
    payload = {
        "model": "routeforge-economy",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }
    async with harness.client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=harness.user
    ) as r:
        assert r.status_code == 200
        assert r.headers["x-routeforge-fallback"] == "true"
        events = [line async for line in r.aiter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
