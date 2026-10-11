"""Benchmark B: response-cache effectiveness.

Phases (same N unique held-out prompts, temperature=0, one explicit model):
  1. no_cache  — every request sent with ``X-RouteForge-Cache: bypass``.
  2. cold      — cache flushed, each prompt sent once (all misses: measures the cost of a
                 cache lookup + write on the miss path).
  3. warm      — the same prompts again (all hits: best case).
  4. mixed     — a synthetic workload where each request repeats an earlier prompt with
                 probability ``--repeat-rate`` (default 0.3), cache flushed first. The hit
                 rate here is a property of that assumed repetition rate, NOT a prediction
                 for real traffic.

    uv run python -m benchmarks.cache_bench --base-url http://localhost:8000 --model gpt-4.1-nano
"""

from __future__ import annotations

import argparse
import asyncio
import random
from typing import Any

from app.evaluation.datasets import Dataset
from benchmarks.common import experiment_metadata, write_csv, write_summary
from benchmarks.gateway_client import Gateway, Result, latency_stats, run_concurrent
from router_training.collect import DATASETS


def phase_summary(results: list[Result]) -> dict[str, Any]:
    n = len(results)
    hits = sum(r.cache == "hit" for r in results)
    lookups = sum(r.cache in ("hit", "miss") for r in results)
    backend_calls = sum(r.cache != "hit" for r in results if r.ok)
    return {
        "requests": n,
        "success_rate": sum(r.ok for r in results) / n,
        "cache_hits": hits,
        "eligible_lookups": lookups,
        "hit_rate": hits / lookups if lookups else None,
        "backend_invocations": backend_calls,
        "estimated_cost_usd": sum(r.cost_usd for r in results),
        **latency_stats(results),
    }


async def main_async(args: argparse.Namespace) -> None:
    gw = Gateway(args.base_url)
    ready = await gw.ready()
    ds = Dataset(DATASETS / args.dataset)
    cases = [c for c in ds.select("test") if c.category not in ("code", "summarization")][: args.n]

    def body(c: Any) -> dict[str, Any]:
        return {
            "model": args.model,
            "messages": c.messages,
            "max_tokens": min(c.max_tokens, 128),
            "temperature": 0,
        }

    async def send(cases_: list[Any], bypass: bool = False) -> list[Result]:
        headers = {"x-routeforge-cache": "bypass"} if bypass else None
        return await run_concurrent([gw.chat(body(c), headers) for c in cases_], args.concurrency)

    phases: dict[str, list[Result]] = {}
    phases["no_cache"] = await send(cases, bypass=True)
    await gw.admin("DELETE", "/api/admin/cache")
    phases["cold"] = await send(cases)
    phases["warm"] = await send(cases)
    await gw.admin("DELETE", "/api/admin/cache")
    rng = random.Random(7)
    stream: list[Any] = []
    seen: list[Any] = []
    pool = list(cases)
    for _ in range(args.mixed_requests):
        if seen and rng.random() < args.repeat_rate:
            stream.append(rng.choice(seen))
        else:
            c = pool.pop(0) if pool else rng.choice(cases)
            seen.append(c)
            stream.append(c)
    # sequential-in-order so a repeat can only hit after its first occurrence completed
    mixed: list[Result] = []
    for i in range(0, len(stream), args.concurrency):
        mixed += await send(stream[i : i + args.concurrency])
    phases["mixed"] = mixed
    await gw.close()

    summ = {k: phase_summary(v) for k, v in phases.items()}
    nc, warm, cold = summ["no_cache"], summ["warm"], summ["cold"]
    headline = [
        {
            "label": "Warm-cache p50 latency",
            "value": f"{warm['p50_ms']:.1f} ms",
            "detail": f"vs {nc['p50_ms']:.0f} ms uncached",
        },
        {
            "label": "Latency reduction on hits (p50)",
            "value": f"{100 * (1 - warm['p50_ms'] / nc['p50_ms']):.1f}%",
        },
        {
            "label": "Miss-path overhead (p50)",
            "value": f"{cold['p50_ms'] - nc['p50_ms']:+.1f} ms",
            "detail": "cold vs uncached; includes provider variance",
        },
        {
            "label": f"Hit rate at {int(100 * args.repeat_rate)}% synthetic repeats",
            "value": f"{100 * (summ['mixed']['hit_rate'] or 0):.1f}%",
            "detail": f"{summ['mixed']['backend_invocations']} backend calls for {summ['mixed']['requests']} requests",
        },
    ]
    summary = {
        "metadata": experiment_metadata(
            "B_cache",
            mode=ready.get("components", {}).get("models"),
            base_url=args.base_url,
            model=args.model,
            n_unique_prompts=len(cases),
            concurrency=args.concurrency,
            repeat_rate=args.repeat_rate,
            dataset=ds.name,
            dataset_version=ds.version,
            generation={"temperature": 0, "max_tokens": "min(case,128)"},
            simulated=ready.get("simulated"),
        ),
        "simulated": ready.get("simulated"),
        "status": "complete",
        "headline": headline,
        "cache_modes": summ,
        "limitations": [
            "Exact-match cache: only byte-identical (tenant, model, messages, params) requests hit.",
            "The mixed workload's repetition rate is an assumption; real hit rates depend entirely on traffic.",
            "Warm-phase latency measures the cache path, not model quality; cost avoided is list-price estimated.",
        ],
    }
    name = f"B_cache_{'mock' if ready.get('simulated') else args.label}"
    write_summary(name, summary)
    write_csv(
        name, "requests.csv", [{"phase": k, **r.__dict__} for k, v in phases.items() for r in v]
    )
    for k, v in summ.items():
        print(
            f"{k:9s} hit_rate={v['hit_rate']} p50={v['p50_ms']:.1f}ms p95={v['p95_ms']:.1f}ms backend={v['backend_invocations']} cost=${v['estimated_cost_usd']:.5f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gpt-4.1-nano")
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--mixed-requests", type=int, default=200)
    ap.add_argument("--repeat-rate", type=float, default=0.3)
    ap.add_argument("--label", default="hosted")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
