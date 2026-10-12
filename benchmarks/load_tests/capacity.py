"""Benchmark C, part 1: gateway capacity with zero provider latency.

Uses ``hey`` (a compiled Go load generator) rather than a Python client: during
development a single-process Python/httpx client turned out to be the bottleneck above
~10 concurrent connections and produced a false "throughput collapse" (see
docs/BENCHMARKS.md). Run against a gateway in replay mode (recorded real outputs,
zero provider latency) so throughput and latency reflect only the gateway: auth, rate
limiting, routing, cache lookup, execution, telemetry and request logging.

    ROUTEFORGE_MODE=replay-latest ... docker compose up -d backend
    uv run python -m benchmarks.load_tests.capacity --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.evaluation.datasets import Dataset
from benchmarks.common import experiment_metadata, write_summary
from benchmarks.gateway_client import Gateway
from router_training.collect import DATASETS


def parse_hey(out: str) -> dict[str, Any]:
    def grab(pattern: str) -> float | None:
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    codes = {int(c): int(n) for c, n in re.findall(r"\[(\d{3})\]\s+(\d+) responses", out)}
    total = sum(codes.values())
    return {
        "rps": grab(r"Requests/sec:\s+([\d.]+)"),
        "p50_ms": (grab(r"50%+ in ([\d.]+) secs") or 0) * 1000,
        "p95_ms": (grab(r"95%+ in ([\d.]+) secs") or 0) * 1000,
        "p99_ms": (grab(r"99%+ in ([\d.]+) secs") or 0) * 1000,
        "requests": total,
        "error_rate": (total - codes.get(200, 0)) / total if total else None,
        "status_codes": codes,
    }


async def setup(base_url: str) -> tuple[str, dict[str, Any]]:
    gw = Gateway(base_url)
    ready = await gw.ready()
    key = await gw.admin(
        "POST",
        "/api/admin/keys",
        {"name": "capacity-test", "rate_limit_rpm": 100000, "max_concurrency": 1024},
    )
    await gw.close()
    return key["key"], ready


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 25, 50, 100])
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=["claude-haiku-5-5", "routeforge-heuristic", "routeforge-learned"],
    )
    ap.add_argument("--label", default="replay")
    args = ap.parse_args()

    hey = shutil.which("hey")
    if hey is None:
        raise SystemExit("install hey (brew install hey / go install github.com/rakyll/hey@latest)")
    key, ready = asyncio.run(setup(args.base_url))
    # One recorded held-out prompt (replay needs prompts that were actually collected).
    case = next(
        c
        for c in Dataset(DATASETS / "routeforge-bench-v1").select("test")
        if c.category == "factual_qa"
    )
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for target in args.targets:
            body = Path(tmp) / f"{target}.json"
            body.write_text(
                json.dumps({"model": target, "messages": case.messages, **case.request_params()})
            )
            # warm-up
            subprocess.run(
                [
                    hey,
                    "-n",
                    "300",
                    "-c",
                    "10",
                    "-m",
                    "POST",
                    "-H",
                    f"authorization: Bearer {key}",
                    "-H",
                    "x-routeforge-cache: bypass",
                    "-T",
                    "application/json",
                    "-D",
                    str(body),
                    f"{args.base_url}/v1/chat/completions",
                ],
                capture_output=True,
                check=False,
            )
            for c in args.concurrency:
                out = subprocess.run(
                    [
                        hey,
                        "-n",
                        str(args.requests),
                        "-c",
                        str(c),
                        "-m",
                        "POST",
                        "-H",
                        f"authorization: Bearer {key}",
                        "-H",
                        "x-routeforge-cache: bypass",
                        "-T",
                        "application/json",
                        "-D",
                        str(body),
                        f"{args.base_url}/v1/chat/completions",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                row = {"target": target, "concurrency": c, **parse_hey(out)}
                results.append(row)
                print(
                    f"{target:22s} c={c:3d} {row['rps']:8.1f} req/s  p50 {row['p50_ms']:6.1f} ms  p95 {row['p95_ms']:6.1f} ms  "
                    f"p99 {row['p99_ms']:6.1f} ms  errors {row['error_rate']:.3%}",
                    flush=True,
                )

    peak = {
        t: max((r for r in results if r["target"] == t), key=lambda r: r["rps"] or 0)
        for t in args.targets
    }
    c1 = {
        t: next(
            r for r in results if r["target"] == t and r["concurrency"] == min(args.concurrency)
        )
        for t in args.targets
    }
    pinned, learned = args.targets[0], args.targets[-1]
    summary = {
        "metadata": experiment_metadata(
            "C_capacity",
            mode=ready.get("components", {}).get("models"),
            base_url=args.base_url,
            load_generator="hey 0.1.5 (Go), closed loop",
            requests_per_level=args.requests,
            prompt_case=case.id,
            router_artifact=ready.get("components", {}).get("router_artifact"),
            cache="bypassed",
            deployment="docker compose backend: 1 uvicorn worker (uvloop), cpus=2.0, memory=1g; request logging on",
        ),
        "status": "complete",
        "headline": [
            {
                "label": f"Peak throughput ({learned})",
                "value": f"{peak[learned]['rps']:.0f} req/s",
                "detail": f"at {peak[learned]['concurrency']} clients, p95 {peak[learned]['p95_ms']:.1f} ms",
            },
            {
                "label": f"Peak throughput ({pinned}, no routing)",
                "value": f"{peak[pinned]['rps']:.0f} req/s",
            },
            {
                "label": "Single-client gateway latency (p50)",
                "value": f"{c1[learned]['p50_ms']:.2f} ms learned / {c1[pinned]['p50_ms']:.2f} ms pinned",
                "detail": "zero provider latency: pure gateway overhead",
            },
            {
                "label": "Errors across all levels",
                "value": f"{sum((r['error_rate'] or 0) * r['requests'] for r in results):.0f}",
            },
        ],
        "levels": [
            {
                "users": r["concurrency"],
                "target": r["target"],
                "success_rps": r["rps"] * (1 - (r["error_rate"] or 0)),
                "p50_ms": r["p50_ms"],
                "p95_ms": r["p95_ms"],
                "p99_ms": r["p99_ms"],
                "error_rate": r["error_rate"],
                "requests": r["requests"],
            }
            for r in results
            if r["target"] == learned
        ],
        "all_levels": results,
        "limitations": [
            "Single replica, single worker, on a laptop (Apple M4) that also runs the load generator.",
            "Replay mode: provider latency is zero, so this is the gateway's own capacity, not end-to-end throughput.",
            "One prompt repeated (cache bypassed); routing cost varies little with prompt length for typical prompts.",
        ],
    }
    write_summary(f"C_capacity_{args.label}", summary)


if __name__ == "__main__":
    main()
