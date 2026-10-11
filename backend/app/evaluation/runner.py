"""Evaluation runner: send dataset cases to a target (concrete model or routing policy)
through the gateway, score the outputs, and summarise.

Two clients share one interface:
* ``InProcessClient`` calls ``GatewayService`` directly (used by the API's evaluation
  endpoint and CI replay runs) — identical code path to production, minus HTTP.
* ``HttpClient`` calls a running gateway over HTTP like any OpenAI-compatible client
  (used by end-to-end benchmarks against the Docker deployment).

Cache is bypassed by default so quality and latency reflect real inference.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from app.core.errors import GatewayError
from app.evaluation.datasets import EvalCase
from app.evaluation.scoring import Scorer
from app.evaluation.stats import wilson_interval
from app.schemas.openai import ChatCompletionRequest
from app.services.auth import Principal
from app.services.gateway import GatewayService, RequestContext

INTERNAL_PRINCIPAL = Principal(
    key_id="internal-eval",
    name="evaluation runner",
    role="admin",
    tenant="evaluation",
    rate_limit_rpm=1_000_000,
    max_concurrency=1024,
)


@dataclass
class CaseOutcome:
    case_id: str
    category: str
    split: str
    target: str
    content: str | None = None
    strategy: str | None = None
    selected_model: str | None = None
    final_model: str | None = None
    cache: str | None = None
    fallback_used: bool = False
    simulated: bool = False
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    score: float = 0.0
    acceptable: bool = False
    score_details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, include_content: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_content:
            d.pop("content")
        return d


@dataclass(frozen=True)
class TargetOptions:
    quality_threshold: float | None = None
    latency_budget_ms: float | None = None
    bypass_cache: bool = True


class CompletionClient(Protocol):
    async def complete(self, case: EvalCase, target: str, opts: TargetOptions) -> CaseOutcome: ...


def _request(case: EvalCase, target: str, opts: TargetOptions) -> dict[str, Any]:
    body: dict[str, Any] = {"model": target, "messages": case.messages, **case.request_params()}
    rf: dict[str, Any] = {"tags": ["evaluation"]}
    if opts.quality_threshold is not None:
        rf["quality_threshold"] = opts.quality_threshold
    if opts.latency_budget_ms is not None:
        rf["latency_budget_ms"] = opts.latency_budget_ms
    body["routeforge"] = rf
    return body


class InProcessClient:
    def __init__(self, gateway: GatewayService) -> None:
        self.gateway = gateway

    async def complete(self, case: EvalCase, target: str, opts: TargetOptions) -> CaseOutcome:
        out = CaseOutcome(case_id=case.id, category=case.category, split=case.split, target=target)
        req = ChatCompletionRequest.model_validate(_request(case, target, opts))
        ctx = RequestContext(
            principal=INTERNAL_PRINCIPAL,
            cache_bypass=opts.bypass_cache,
            skip_rate_limit=True,
            source="evaluation",
        )
        started = time.perf_counter()
        try:
            result = await self.gateway.chat(req, ctx)
        except GatewayError as exc:
            out.error = exc.code or exc.error_type
            out.latency_ms = (time.perf_counter() - started) * 1000
            return out
        meta = result.response.routeforge
        assert meta is not None
        out.content = result.response.choices[0].message.content
        out.strategy, out.selected_model, out.final_model = (
            meta.strategy,
            meta.selected_model,
            result.response.model,
        )
        out.cache, out.fallback_used, out.simulated = meta.cache, meta.fallback_used, meta.simulated
        out.latency_ms = result.latency_ms
        out.input_tokens = result.response.usage.prompt_tokens
        out.output_tokens = result.response.usage.completion_tokens
        out.cost_usd = meta.cost_usd
        return out


class HttpClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 180.0) -> None:
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def complete(self, case: EvalCase, target: str, opts: TargetOptions) -> CaseOutcome:
        out = CaseOutcome(case_id=case.id, category=case.category, split=case.split, target=target)
        headers = {"x-routeforge-cache": "bypass"} if opts.bypass_cache else {}
        started = time.perf_counter()
        try:
            resp = await self.client.post(
                "/v1/chat/completions", json=_request(case, target, opts), headers=headers
            )
        except httpx.HTTPError as exc:
            out.error = f"http:{type(exc).__name__}"
            out.latency_ms = (time.perf_counter() - started) * 1000
            return out
        out.latency_ms = (time.perf_counter() - started) * 1000
        data = resp.json()
        if resp.status_code != 200:
            out.error = (data.get("error") or {}).get("code") or f"http_{resp.status_code}"
            return out
        meta = data.get("routeforge") or {}
        out.content = data["choices"][0]["message"]["content"]
        out.strategy, out.selected_model = meta.get("strategy"), meta.get("selected_model")
        out.final_model = data.get("model")
        out.cache, out.fallback_used = meta.get("cache"), bool(meta.get("fallback_used"))
        out.simulated = bool(meta.get("simulated"))
        out.input_tokens = data["usage"]["prompt_tokens"]
        out.output_tokens = data["usage"]["completion_tokens"]
        out.cost_usd = float(meta.get("cost_usd", 0.0))
        return out


async def run_eval(
    cases: list[EvalCase],
    targets: list[str],
    client: CompletionClient,
    scorer: Scorer,
    *,
    opts: TargetOptions | None = None,
    concurrency: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[CaseOutcome]:
    opts = opts or TargetOptions()
    sem = asyncio.Semaphore(concurrency)
    jobs = [(c, t) for t in targets for c in cases]
    done = 0

    async def one(case: EvalCase, target: str) -> CaseOutcome:
        nonlocal done
        async with sem:
            res = await client.complete(case, target, opts)
        done += 1
        if progress:
            progress(done, len(jobs))
        return res

    outcomes = await asyncio.gather(*(one(c, t) for c, t in jobs))
    by_id = {c.id: c for c in cases}
    scores = await scorer.score_many(
        [(by_id[o.case_id], o.content if o.error is None else None) for o in outcomes]
    )
    for o, s in zip(outcomes, scores, strict=True):
        o.score, o.acceptable, o.score_details = s.score, s.acceptable, s.details
    return list(outcomes)


def _block(rows: list[CaseOutcome]) -> dict[str, Any]:
    n = len(rows)
    acc = sum(r.acceptable for r in rows)
    lo, hi = wilson_interval(acc, n)
    lat = [r.latency_ms for r in rows if r.latency_ms is not None and r.error is None]
    cost = sum(r.cost_usd for r in rows)
    return {
        "n": n,
        "acceptable": acc,
        "acceptable_rate": acc / n if n else None,
        "acceptable_rate_ci95": [lo, hi],
        "mean_score": float(np.mean([r.score for r in rows])) if n else None,
        "errors": sum(r.error is not None for r in rows),
        "total_cost_usd": cost,
        "cost_per_request_usd": cost / n if n else None,
        "cost_per_acceptable_usd": cost / acc if acc else None,
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat else None,
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else None,
        "fallbacks": sum(r.fallback_used for r in rows),
        "model_distribution": dict(Counter(r.final_model or "error" for r in rows)),
        "simulated": any(r.simulated for r in rows),
    }


def summarize(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    by_target: dict[str, list[CaseOutcome]] = defaultdict(list)
    for o in outcomes:
        by_target[o.target].append(o)
    summary: dict[str, Any] = {}
    for target, rows in by_target.items():
        cats: dict[str, list[CaseOutcome]] = defaultdict(list)
        for r in rows:
            cats[r.category].append(r)
        summary[target] = {
            "overall": _block(rows),
            "by_category": {c: _block(v) for c, v in sorted(cats.items())},
        }
    return summary
