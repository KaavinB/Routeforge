"""Benchmark C: concurrency sweep with Locust (headless).

    uv run python -m benchmarks.load_tests.run_load --base-url http://localhost:8000 \
        --users 1 5 10 25 50 --duration 30

Creates a dedicated API key with a very high rate limit so the gateway's own per-key
limiter does not cap the measurement, then runs one Locust process per concurrency level.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.common import RESULTS, experiment_metadata, write_summary
from benchmarks.gateway_client import Gateway

HERE = Path(__file__).resolve().parent


async def setup(base_url: str) -> tuple[str, dict[str, Any]]:
    gw = Gateway(base_url)
    ready = await gw.ready()
    key = await gw.admin(
        "POST",
        "/api/admin/keys",
        {"name": "load-test", "rate_limit_rpm": 100000, "max_concurrency": 1024},
    )
    await gw.close()
    return key["key"], ready


def parse_stats(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        rows = {r["Name"]: r for r in csv.DictReader(fh)}
    agg = rows["Aggregated"]

    def f(k: str) -> float:
        try:
            return float(agg[k])
        except (KeyError, ValueError):
            return float("nan")

    n, fail = f("Request Count"), f("Failure Count")
    return {
        "requests": int(n),
        "failures": int(fail),
        "error_rate": fail / n if n else None,
        "rps": f("Requests/s"),
        "success_rps": f("Requests/s") * (1 - fail / n) if n else 0.0,
        "p50_ms": f("50%"),
        "p95_ms": f("95%"),
        "p99_ms": f("99%"),
        "mean_ms": f("Average Response Time"),
        "max_ms": f("Max Response Time"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--duration", type=int, default=30, help="seconds per level")
    ap.add_argument("--model", default="routeforge-auto")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    key, ready = asyncio.run(setup(args.base_url))
    simulated = bool(ready.get("simulated"))
    name = f"C_load_{args.label or ('mock' if simulated else 'hosted')}"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    levels = []
    for u in args.users:
        prefix = out / f"locust_u{u}"
        env = {**os.environ, "RF_LOAD_KEY": key, "RF_LOAD_MODEL": args.model}
        cmd = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(HERE / "locustfile.py"),
            "--headless",
            "-u",
            str(u),
            "-r",
            str(u),
            "--run-time",
            f"{args.duration}s",
            "--host",
            args.base_url,
            "--csv",
            str(prefix),
            "--only-summary",
            "--stop-timeout",
            "10",
        ]
        print(f"level {u} users ...", flush=True)
        subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
        stats = parse_stats(Path(f"{prefix}_stats.csv"))
        levels.append({"users": u, **stats})
        print(
            f"  {stats['success_rps']:.1f} ok req/s  p50 {stats['p50_ms']:.0f} ms  p95 {stats['p95_ms']:.0f} ms  errors {stats['error_rate']:.3%}",
            flush=True,
        )

    best = max(levels, key=lambda r: r["success_rps"])
    summary = {
        "metadata": experiment_metadata(
            "C_load",
            mode=ready.get("components", {}).get("models"),
            base_url=args.base_url,
            model_field=args.model,
            duration_s_per_level=args.duration,
            default_strategy=ready.get("default_strategy"),
            cache="bypassed",
            workload="closed-loop, zero think time, random held-out prompts, max_tokens<=128",
            deployment="docker compose backend: 1 uvicorn worker, cpus=2.0, memory=1g",
            simulated=simulated,
        ),
        "simulated": simulated,
        "status": "complete",
        "headline": [
            {
                "label": "Peak successful throughput",
                "value": f"{best['success_rps']:.1f} req/s",
                "detail": f"at {best['users']} concurrent users",
            },
            {"label": "p95 latency at peak", "value": f"{best['p95_ms']:.0f} ms"},
            {"label": "Error rate at peak", "value": f"{100 * (best['error_rate'] or 0):.2f}%"},
            {"label": "p50 latency at 1 user", "value": f"{levels[0]['p50_ms']:.0f} ms"},
        ],
        "levels": levels,
        "limitations": [
            "Single gateway replica with one uvicorn worker on a laptop (Apple M4) sharing CPU with the load generator.",
            "In mock mode the providers are simulated: this measures gateway capacity and overhead, not provider throughput.",
        ],
    }
    write_summary(name, summary)


if __name__ == "__main__":
    main()
