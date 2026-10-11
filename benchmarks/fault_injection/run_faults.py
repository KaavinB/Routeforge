"""Benchmark D: failure injection and recovery.

Runs against a gateway started with ROUTEFORGE_FAULT_INJECTION_ENABLED=true (mock mode by
default, so faults are injected in front of simulated providers at zero cost). For each
scenario: reset breakers, install fault rules, send N requests with the ``premium``
policy (primary = premium model, fallbacks by tier proximity), then reconstruct what
happened from the gateway's own request log (attempt-by-attempt records).

Definitions (per scenario)
  eligible_for_recovery   requests whose first upstream attempt on the primary model failed
                          or whose primary was skipped because its circuit was open
  recovered_by_retry      eligible requests that succeeded on the primary after a retry
  recovered_by_fallback   eligible requests that succeeded on a different model
  fallback_recovery_rate  recovered_by_fallback / eligible requests that needed a fallback
                          (i.e. excluding those already recovered by retry)
  recovery_rate           (recovered_by_retry + recovered_by_fallback) / eligible_for_recovery
  upstream_calls_per_req  mean number of real upstream calls (retry amplification)
  detection_requests      index (in submission order) of the first request routed away from
                          the primary; approximate under concurrent load
  recovery_overhead       per recovered request: end-to-end latency minus the successful
                          attempt's latency (failed attempts + backoff + re-routing)
Dependency outages (Redis, PostgreSQL) are reported as graceful degradation: requests are
served but caching/rate limiting fall back to local mode, or request logs are dropped.

    uv run python -m benchmarks.fault_injection.run_faults --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
from typing import Any

import numpy as np

from benchmarks.common import REPO, experiment_metadata, write_csv, write_summary
from benchmarks.gateway_client import Gateway, Result, latency_stats, run_concurrent

PROMPTS = [
    "Summarize why circuit breakers help distributed systems.",
    "Who wrote the novel Beloved?",
    "Classify the topic of: 'Stocks rallied after the central bank held rates.'",
    "Extract the city from: 'The meetup moved to Lisbon this year.'",
    "What is 17 * 23?",
]


async def send_batch(
    gw: Gateway, n: int, concurrency: int, strategy: str, tag: str
) -> list[Result]:
    bodies = [
        {
            "model": f"routeforge-{strategy}",
            "messages": [{"role": "user", "content": f"{PROMPTS[i % len(PROMPTS)]} (#{tag}-{i})"}],
            "max_tokens": 64,
            "temperature": 0,
        }
        for i in range(n)
    ]
    return await run_concurrent(
        [gw.chat(b, {"x-routeforge-cache": "bypass"}) for b in bodies], concurrency
    )


async def details(gw: Gateway, results: list[Result]) -> list[dict[str, Any]]:
    await asyncio.sleep(1.0)  # let the async request-log writer flush
    out = []
    for r in results:
        if not r.request_id:
            out.append({})
            continue
        try:
            out.append(await gw.admin("GET", f"/api/analytics/requests/{r.request_id}"))
        except Exception:
            out.append({})
    return out


def analyse(
    primary: str, results: list[Result], dets: list[dict[str, Any]], baseline_p50: float | None
) -> dict[str, Any]:
    eligible = retry_rec = fb_rec = 0
    calls: list[int] = []
    overhead: list[float] = []
    detection = None
    for i, (r, d) in enumerate(zip(results, dets, strict=True)):
        attempts = d.get("attempt_detail") or []
        real = [a for a in attempts if a["outcome"] not in ("circuit_open", "saturated")]
        if d:
            calls.append(len(real))
        first = attempts[0] if attempts else None
        primary_skipped = d.get("selected_model") not in (None, primary) or (
            first and first["outcome"] == "circuit_open"
        )
        failed_first = bool(
            first and first["model_id"] == primary and first["outcome"] != "success"
        )
        if detection is None and primary_skipped:
            detection = i
        if not (failed_first or primary_skipped):
            continue
        eligible += 1
        if r.ok and r.model == primary:
            retry_rec += 1
        elif r.ok and r.model != primary:
            fb_rec += 1
        ok_attempt = next((a for a in reversed(attempts) if a["outcome"] == "success"), None)
        if r.ok and ok_attempt is not None:
            # time spent on failed attempts, backoff sleeps and re-routing
            overhead.append(max(r.latency_ms - ok_attempt["latency_s"] * 1000, 0.0))
    lat = latency_stats(results)
    needing_fallback = eligible - retry_rec
    return {
        "requests": len(results),
        "success_rate": sum(r.ok for r in results) / len(results),
        "eligible_for_recovery": eligible,
        "recovered_by_retry": retry_rec,
        "recovered_by_fallback": fb_rec,
        "recovery_rate": (retry_rec + fb_rec) / eligible if eligible else None,
        "fallback_recovery_rate": fb_rec / needing_fallback if needing_fallback else None,
        "upstream_calls_per_request": float(np.mean(calls)) if calls else None,
        "max_upstream_calls": int(max(calls)) if calls else None,
        "detection_requests": detection,
        "p50_ms": lat["p50_ms"],
        "p95_ms": lat["p95_ms"],
        "recovery_overhead_p50_ms": float(np.percentile(overhead, 50)) if overhead else None,
        "recovery_overhead_p95_ms": float(np.percentile(overhead, 95)) if overhead else None,
        "errors": sorted({r.error for r in results if r.error}),
    }


def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=REPO, check=True, capture_output=True)


async def main_async(args: argparse.Namespace) -> None:
    gw = Gateway(args.base_url)
    ready = await gw.ready()
    if not ready.get("simulated") and not args.allow_real_providers:
        raise SystemExit(
            "refusing to inject faults against real providers without --allow-real-providers"
        )
    primary = args.primary
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    async def reset() -> None:
        await gw.admin("DELETE", "/api/admin/faults")
        await gw.admin("POST", "/api/admin/breakers/reset")

    scenarios: list[tuple[str, str, list[tuple[str, dict[str, Any]]]]] = [
        ("baseline", "no faults", []),
        (
            "transient_500_20pct",
            "20% of calls to the primary return HTTP 500",
            [(primary, {"mode": "http_500", "probability": 0.2})],
        ),
        (
            "rate_limited_429_50pct",
            "50% of calls return 429 (Retry-After 0.2 s)",
            [(primary, {"mode": "http_429", "probability": 0.5, "retry_after_s": 0.2})],
        ),
        (
            "invalid_response_30pct",
            "30% of calls return a malformed body",
            [(primary, {"mode": "invalid_response", "probability": 0.3})],
        ),
        (
            "persistent_500",
            "primary returns HTTP 500 on every call",
            [(primary, {"mode": "http_500"})],
        ),
        (
            "provider_outage",
            "primary refuses connections (outage)",
            [(primary, {"mode": "outage"})],
        ),
        (
            "provider_timeout",
            "primary hangs until the 3 s gateway timeout",
            [(primary, {"mode": "timeout"})],
        ),
        (
            "two_provider_outage",
            "primary and first fallback both down",
            [(primary, {"mode": "outage"}), (args.secondary, {"mode": "outage"})],
        ),
    ]
    baseline_p50: float | None = None
    for name, desc, rules in scenarios:
        await reset()
        for model, rule in rules:
            await gw.admin("PUT", f"/api/admin/faults/{model}", rule)
        res = await send_batch(gw, args.n, args.concurrency, "premium", name)
        dets = await details(gw, res)
        row = {
            "scenario": name,
            "description": desc,
            "kind": "provider_fault",
            **analyse(primary, res, dets, baseline_p50),
        }
        if name == "baseline":
            baseline_p50 = row["p50_ms"]
        rows.append(row)
        raw += [{"scenario": name, **r.__dict__} for r in res]
        print(
            f"{name:26s} success={row['success_rate']:.3f} recovery={row['recovery_rate']} fb={row['fallback_recovery_rate']} "
            f"calls/req={row['upstream_calls_per_request']} detect={row['detection_requests']} p50={row['p50_ms']:.0f}ms",
            flush=True,
        )
    await reset()

    # Dependency outages: graceful degradation (served without cache/rate-limit store or logs).
    if not args.skip_dependency_outages:
        for name, service, desc in (
            ("redis_outage", "redis", "Redis stopped: cache misses + local rate limiter"),
            (
                "database_outage",
                "postgres",
                "PostgreSQL stopped: bootstrap-key auth, request logs dropped",
            ),
        ):
            compose("stop", service)
            await asyncio.sleep(2)
            res = await send_batch(gw, args.n, args.concurrency, "economy", name)
            compose("start", service)
            lat = latency_stats(res)
            rows.append(
                {
                    "scenario": name,
                    "description": desc,
                    "kind": "dependency_outage",
                    "requests": len(res),
                    "success_rate": sum(r.ok for r in res) / len(res),
                    "p50_ms": lat["p50_ms"],
                    "p95_ms": lat["p95_ms"],
                    "errors": sorted({r.error for r in res if r.error}),
                    "outcome": "graceful degradation"
                    if all(r.ok for r in res)
                    else "partial failure",
                }
            )
            raw += [{"scenario": name, **r.__dict__} for r in res]
            print(
                f"{name:26s} success={rows[-1]['success_rate']:.3f} p50={lat['p50_ms']}", flush=True
            )
            await asyncio.sleep(5)

    # Mid-stream failure: not recoverable once tokens were sent.
    await reset()
    await gw.admin("PUT", f"/api/admin/faults/{args.economy}", {"mode": "stream_abort"})
    streams = await run_concurrent(
        [
            gw.chat_stream(
                {
                    "model": f"routeforge-{'economy'}",
                    "messages": [{"role": "user", "content": f"stream {i}"}],
                    "max_tokens": 32,
                }
            )
            for i in range(10)
        ],
        5,
    )
    await reset()
    rows.append(
        {
            "scenario": "stream_abort_after_first_token",
            "kind": "partial_stream",
            "requests": len(streams),
            "description": "economy model aborts streams after the first chunk",
            "success_rate": None,
            "outcome": "client receives an SSE error event; no transparent recovery after first token",
        }
    )
    await gw.close()

    prov = [r for r in rows if r["kind"] == "provider_fault" and r["scenario"] != "baseline"]
    elig = sum(r["eligible_for_recovery"] for r in prov)
    recovered = sum(r["recovered_by_retry"] + r["recovered_by_fallback"] for r in prov)
    fb_need = sum(r["eligible_for_recovery"] - r["recovered_by_retry"] for r in prov)
    fb_ok = sum(r["recovered_by_fallback"] for r in prov)
    summary = {
        "metadata": experiment_metadata(
            "D_faults",
            mode=ready.get("components", {}).get("models"),
            base_url=args.base_url,
            primary=primary,
            secondary=args.secondary,
            requests_per_scenario=args.n,
            concurrency=args.concurrency,
            simulated=ready.get("simulated"),
            reliability_config="2 attempts/model, 2 fallbacks, full-jitter backoff, retry budget 20%, breaker 5 consecutive / 50% of 20",
        ),
        "simulated": ready.get("simulated"),
        "status": "complete",
        "headline": [
            {
                "label": "Recovery rate (all injected provider faults)",
                "value": f"{100 * recovered / elig:.1f}%" if elig else "n/a",
                "detail": f"{recovered}/{elig} affected requests",
            },
            {
                "label": "Fallback recovery rate",
                "value": f"{100 * fb_ok / fb_need:.1f}%" if fb_need else "n/a",
                "detail": f"{fb_ok}/{fb_need} requests needing another model",
            },
            {
                "label": "Max upstream calls / request",
                "value": str(max((r["max_upstream_calls"] or 0) for r in prov)),
                "detail": "bounded retry amplification",
            },
            {
                "label": "Dependency outages",
                "value": ", ".join(
                    f"{r['scenario']}: {100 * r['success_rate']:.0f}%"
                    for r in rows
                    if r["kind"] == "dependency_outage"
                )
                or "skipped",
            },
        ],
        "totals": {
            "eligible": elig,
            "recovered": recovered,
            "needing_fallback": fb_need,
            "recovered_by_fallback": fb_ok,
        },
        "scenarios": rows,
        "limitations": [
            "Faults are injected in front of simulated providers; recovery latencies reflect mock model timings plus real gateway logic.",
            "Fallback models are other models in the pool: recovery preserves availability, not necessarily identical answer quality.",
            "Circuit-breaker state is per gateway process.",
        ],
    }
    name = f"D_faults_{'mock' if ready.get('simulated') else 'real'}"
    write_summary(name, summary)
    write_csv(name, "requests.csv", raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--primary", default="mock-premium")
    ap.add_argument("--secondary", default="mock-medium")
    ap.add_argument("--economy", default="mock-economy")
    ap.add_argument("--skip-dependency-outages", action="store_true")
    ap.add_argument("--allow-real-providers", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
