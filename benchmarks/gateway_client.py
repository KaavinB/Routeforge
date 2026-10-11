"""Small async client used by the gateway benchmarks (B-E). Talks to a running gateway
over HTTP exactly like an application would, and records per-request measurements from
the client's point of view plus RouteForge's response headers."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def env_value(name: str) -> str:
    """Read a value from the process env, falling back to the repo's .env file."""
    if os.environ.get(name):
        return os.environ[name]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


@dataclass
class Result:
    status: int
    latency_ms: float
    model: str | None = None
    cache: str | None = None
    fallback: bool = False
    cost_usd: float = 0.0
    error: str | None = None
    request_id: str | None = None
    ttft_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


class Gateway:
    def __init__(
        self,
        base_url: str,
        client_key: str | None = None,
        admin_key: str | None = None,
        timeout_s: float = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_key = client_key or env_value("ROUTEFORGE_BOOTSTRAP_CLIENT_KEY")
        self.admin_key = admin_key or env_value("ROUTEFORGE_BOOTSTRAP_ADMIN_KEY")
        self.http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_s,
            # Expire idle connections before uvicorn's 5 s keep-alive timeout so a pooled
            # connection is never reused after the server has closed it.
            limits=httpx.Limits(
                max_connections=512, max_keepalive_connections=128, keepalive_expiry=3.0
            ),
        )

    async def close(self) -> None:
        await self.http.aclose()

    def _admin(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.admin_key}"}

    async def ready(self) -> dict[str, Any]:
        r = await self.http.get("/ready")
        data: dict[str, Any] = r.json()
        return data

    async def admin(self, method: str, path: str, json: Any = None) -> Any:
        r = await self.http.request(method, path, json=json, headers=self._admin())
        r.raise_for_status()
        return r.json()

    async def chat(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> Result:
        h = {"authorization": f"Bearer {self.client_key}", **(headers or {})}
        started = time.perf_counter()
        try:
            r = await self.http.post("/v1/chat/completions", json=body, headers=h)
        except httpx.HTTPError as exc:
            return Result(
                status=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=type(exc).__name__,
            )
        latency = (time.perf_counter() - started) * 1000
        res = Result(
            status=r.status_code,
            latency_ms=latency,
            model=r.headers.get("x-routeforge-model"),
            cache=r.headers.get("x-routeforge-cache"),
            fallback=r.headers.get("x-routeforge-fallback") == "true",
            cost_usd=float(r.headers.get("x-routeforge-cost-usd", 0) or 0),
            request_id=r.headers.get("x-request-id"),
        )
        if r.status_code != 200:
            try:
                res.error = r.json()["error"].get("code") or r.json()["error"].get("type")
            except Exception:
                res.error = f"http_{r.status_code}"
        return res

    async def chat_stream(self, body: dict[str, Any]) -> Result:
        h = {"authorization": f"Bearer {self.client_key}"}
        started = time.perf_counter()
        ttft = None
        try:
            async with self.http.stream(
                "POST", "/v1/chat/completions", json={**body, "stream": True}, headers=h
            ) as r:
                async for line in r.aiter_lines():
                    if (
                        line.startswith("data: ")
                        and '"content"' in line
                        and ttft is None
                        and '"content":""' not in line
                    ):
                        ttft = (time.perf_counter() - started) * 1000
                status = r.status_code
                model = r.headers.get("x-routeforge-model")
        except httpx.HTTPError as exc:
            return Result(
                status=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=type(exc).__name__,
            )
        return Result(
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=model,
            ttft_ms=ttft,
        )


async def run_concurrent(coros: Sequence[Any], concurrency: int) -> list[Any]:
    sem = asyncio.Semaphore(concurrency)

    async def guard(c: Any) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(guard(c) for c in coros))


def latency_stats(results: Sequence[Result], ok_only: bool = True) -> dict[str, float | None]:
    lat = [r.latency_ms for r in results if (r.ok or not ok_only)]
    if not lat:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    arr = np.asarray(lat)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)) if len(arr) >= 200 else None,
        "mean_ms": float(arr.mean()),
    }
