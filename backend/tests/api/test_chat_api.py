from __future__ import annotations

import json

import httpx
import openai
import pytest

from tests.conftest import CLIENT_KEY, Harness


async def test_health_and_ready(harness: Harness) -> None:
    r = await harness.client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    r = await harness.client.get("/ready")
    body = r.json()
    assert r.status_code == 200
    assert body["simulated"] is True
    assert body["components"]["database"] == "ok"
    assert body["components"]["redis"] == "ok"


async def test_requires_api_key(harness: Harness) -> None:
    r = await harness.client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"
    r = await harness.client.get("/v1/models", headers={"authorization": "Bearer rf-wrong"})
    assert r.status_code == 401


async def test_client_key_cannot_use_admin_api(harness: Harness) -> None:
    r = await harness.client.get("/api/analytics/summary", headers=harness.user)
    assert r.status_code == 403


async def test_chat_completion_routed(harness: Harness) -> None:
    r = await harness.chat(temperature=0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("[SIMULATED")
    assert (
        body["usage"]["total_tokens"]
        == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    meta = body["routeforge"]
    assert meta["strategy"] == "heuristic"
    assert meta["simulated"] is True
    assert r.headers["x-routeforge-model"] == body["model"]
    assert r.headers["x-routeforge-simulated"] == "true"
    assert len(r.headers["x-routeforge-trace-id"]) == 32
    assert r.headers["x-request-id"] == meta["request_id"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("routeforge-premium", "mock-premium"),
        ("routeforge-economy", "mock-economy"),
        ("mock-medium", "mock-medium"),
    ],
)
async def test_strategies_and_explicit_models(harness: Harness, model: str, expected: str) -> None:
    r = await harness.chat(model=model)
    assert r.status_code == 200
    assert r.json()["model"] == expected


async def test_strategy_via_extension_field(harness: Harness) -> None:
    r = await harness.chat(routeforge={"strategy": "economy"})
    assert r.json()["model"] == "mock-economy"


async def test_unknown_model_and_strategy(harness: Harness) -> None:
    r = await harness.chat(model="gpt-nonexistent")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = await harness.chat(model="routeforge-bogus")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_strategy"


async def test_unsupported_parameters_are_rejected_not_dropped(harness: Harness) -> None:
    r = await harness.chat(logprobs=True)
    assert r.status_code == 400
    assert "logprobs" in r.json()["error"]["message"]
    r = await harness.chat(n=2)
    assert r.status_code == 400
    r = await harness.chat(tools=[{"type": "function"}])
    assert r.status_code == 400


async def test_max_tokens_cap(harness: Harness) -> None:
    r = await harness.chat(max_tokens=100_000)
    assert r.status_code == 400 and r.json()["error"]["param"] == "max_tokens"


async def test_body_size_limit(harness: Harness) -> None:
    r = await harness.chat(content="x" * 600_000)
    assert r.status_code == 413


async def test_models_endpoint(harness: Harness) -> None:
    r = await harness.client.get("/v1/models", headers=harness.user)
    ids = {m["id"] for m in r.json()["data"]}
    assert {"routeforge-auto", "routeforge-heuristic", "mock-premium", "mock-economy"} <= ids


async def test_cache_hit_on_identical_deterministic_request(harness: Harness) -> None:
    first = await harness.chat(content="Cache me please", temperature=0)
    second = await harness.chat(content="Cache me please", temperature=0)
    assert first.headers["x-routeforge-cache"] == "miss"
    assert second.headers["x-routeforge-cache"] == "hit"
    assert (
        second.json()["choices"][0]["message"]["content"]
        == first.json()["choices"][0]["message"]["content"]
    )
    assert float(second.headers["x-routeforge-cost-usd"]) == 0.0


async def test_cache_not_used_for_nondeterministic_or_bypass(harness: Harness) -> None:
    r = await harness.chat(content="sampled", temperature=0.7)
    assert r.headers["x-routeforge-cache"] == "nondeterministic"
    await harness.chat(content="bypass me", temperature=0)
    r = await harness.chat(
        content="bypass me", temperature=0, headers={"x-routeforge-cache": "bypass"}
    )
    assert r.headers["x-routeforge-cache"] == "bypass"
    r = await harness.chat(content="bypass me", temperature=0, routeforge={"cache": False})
    assert r.headers["x-routeforge-cache"] == "bypass"


async def test_cache_isolated_between_models(harness: Harness) -> None:
    await harness.chat(content="same prompt", temperature=0, model="mock-small")
    r = await harness.chat(content="same prompt", temperature=0, model="mock-premium")
    assert r.headers["x-routeforge-cache"] == "miss"


async def test_streaming_sse(harness: Harness) -> None:
    payload = {
        "model": "routeforge-economy",
        "messages": [{"role": "user", "content": "stream please"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with harness.client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=harness.user
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line[6:] async for line in r.aiter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text.startswith("[SIMULATED mock-economy]")
    assert chunks[-1]["usage"]["completion_tokens"] > 0
    assert any(c["choices"] and c["choices"][0].get("finish_reason") == "stop" for c in chunks)


async def test_openai_python_sdk_compatibility(harness: Harness) -> None:
    transport = httpx.ASGITransport(app=harness.app)
    client = openai.AsyncOpenAI(
        api_key=CLIENT_KEY,
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=transport, base_url="http://test"),
    )
    resp = await client.chat.completions.create(
        model="routeforge-auto",
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}],
        max_tokens=32,
        extra_body={"routeforge": {"strategy": "premium"}},
    )
    assert resp.model == "mock-premium"
    assert resp.choices[0].message.content
    assert resp.model_extra and resp.model_extra["routeforge"]["strategy"] == "premium"

    stream = await client.chat.completions.create(
        model="routeforge-economy", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    pieces = [
        c.choices[0].delta.content async for c in stream if c.choices and c.choices[0].delta.content
    ]
    assert "".join(pieces).startswith("[SIMULATED")
    models = await client.models.list()
    assert any(m.id == "routeforge-learned" or m.id == "routeforge-heuristic" for m in models.data)


async def test_request_logs_and_analytics(harness: Harness) -> None:
    for i in range(5):
        await harness.chat(content=f"question {i}", model="routeforge-premium")
    await harness.chat(model="does-not-exist")
    await harness.state.log_writer.flush()
    r = await harness.client.get("/api/analytics/summary", headers=harness.admin)
    s = r.json()
    assert s["total_requests"] == 6
    assert s["successful_requests"] == 5
    assert s["simulated_requests"] >= 5
    assert s["estimated_spend_usd"] > 0
    assert s["latency_ms"]["p95"] is not None
    r = await harness.client.get("/api/analytics/requests?limit=3", headers=harness.admin)
    page = r.json()
    assert page["total"] == 6 and len(page["items"]) == 3
    rid = page["items"][0]["request_id"]
    detail = (
        await harness.client.get(f"/api/analytics/requests/{rid}", headers=harness.admin)
    ).json()
    assert "routing_detail" in detail
    assert "prompt_text" not in detail  # prompts are never exposed by the API
    models = (await harness.client.get("/api/analytics/models", headers=harness.admin)).json()
    assert models["models"][0]["model"] == "mock-premium"


async def test_routing_preview(harness: Harness) -> None:
    r = await harness.client.get(
        "/api/analytics/routing/preview",
        params={"prompt": "Write a python function that returns the nth prime"},
        headers=harness.admin,
    )
    body = r.json()
    assert body["predicted_task"] == "code"
    assert body["selected_model"] == "mock-premium"
    assert len(body["candidates"]) == 4


async def test_admin_key_lifecycle(harness: Harness) -> None:
    r = await harness.client.post(
        "/api/admin/keys", json={"name": "ci", "rate_limit_rpm": 2}, headers=harness.admin
    )
    assert r.status_code == 201
    created = r.json()
    key = created["key"]
    assert key.startswith("rf-")
    headers = {"authorization": f"Bearer {key}"}
    payload = {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]}
    ok = [
        await harness.client.post("/v1/chat/completions", json=payload, headers=headers)
        for _ in range(3)
    ]
    assert [x.status_code for x in ok] == [200, 200, 429]
    assert "retry-after" in ok[2].headers
    listed = (await harness.client.get("/api/admin/keys", headers=harness.admin)).json()
    assert any(k["id"] == created["id"] and "key" not in k for k in listed)
    r = await harness.client.delete(f"/api/admin/keys/{created['id']}", headers=harness.admin)
    assert r.status_code == 200
    r = await harness.client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r.status_code == 401


async def test_metrics_endpoint(harness: Harness) -> None:
    await harness.chat()
    r = await harness.client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "routeforge_requests_total" in text
    assert "routeforge_routing_decision_seconds_bucket" in text
    assert "request_id" not in text  # no high-cardinality labels
