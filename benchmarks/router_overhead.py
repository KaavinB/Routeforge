"""Benchmark E: routing overhead.

Part 1 (in-process): time each stage of a routing decision on the held-out prompts —
feature extraction, task classification, quality models, full decision — per strategy.
Part 2 (end-to-end, optional): against a gateway in *replay* mode (recorded outputs, zero
provider latency), compare client-observed latency for a pinned model (no routing) vs.
heuristic and learned routing on identical prompts. The difference is the routing cost
as seen by a client, including HTTP and serialisation noise.

    uv run python -m benchmarks.router_overhead --registry config/models/hosted.yaml \
        --artifact router_training/artifacts/router-hosted-2026-10-11 [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from app.core.registry import ModelRegistry
from app.evaluation.datasets import Dataset
from app.routing.artifact import RouterArtifact
from app.routing.engine import RoutingEngine
from app.routing.features import extract_features, routing_text
from benchmarks.common import experiment_metadata, write_summary
from benchmarks.gateway_client import Gateway, latency_stats
from router_training.collect import DATASETS


def pct(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "n": len(a),
    }


def time_it(fn: Any, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


async def e2e(base_url: str, cases: list[Any], explicit: str, repeats: int) -> dict[str, Any]:
    gw = Gateway(base_url)
    ready = await gw.ready()
    if "replay" not in str(ready.get("components", {}).get("models")):
        await gw.close()
        return {
            "skipped": "gateway is not in replay mode; end-to-end comparison would be dominated by provider latency"
        }
    targets = [explicit, "routeforge-heuristic", "routeforge-learned"]
    out: dict[str, Any] = {}
    for c in cases[:5]:  # warm-up
        for t in targets:
            await gw.chat(
                {"model": t, "messages": c.messages, **c.request_params()},
                {"x-routeforge-cache": "bypass"},
            )
    for t in targets:
        res = []
        for _ in range(repeats):
            for c in cases:
                res.append(
                    await gw.chat(
                        {"model": t, "messages": c.messages, **c.request_params()},
                        {"x-routeforge-cache": "bypass"},
                    )
                )
        ok = [r for r in res if r.ok]
        out[t] = {"requests": len(res), "ok": len(ok), **latency_stats(res)}
    await gw.close()
    base = out[explicit]["p50_ms"]
    for t in targets[1:]:
        out[t]["added_p50_ms_vs_pinned"] = out[t]["p50_ms"] - base
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=Path("config/models/hosted.yaml"))
    ap.add_argument(
        "--artifact", type=Path, default=Path("router_training/artifacts/router-hosted-2026-10-11")
    )
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--base-url")
    ap.add_argument("--explicit-model", default="claude-haiku-5-5")
    args = ap.parse_args()

    threadpool_limits(1)
    registry = ModelRegistry.from_yaml(args.registry)
    artifact = RouterArtifact.load(args.artifact)
    engine = RoutingEngine(registry, artifact=artifact)
    cases = Dataset(DATASETS / args.dataset).select("test")

    stages: dict[str, list[float]] = {
        k: []
        for k in (
            "features",
            "task_classifier",
            "quality_models",
            "decide_learned",
            "decide_heuristic",
            "decide_premium",
        )
    }
    for _ in range(args.repeats):
        for c in cases:
            params = c.request_params()
            f = extract_features(c.messages, params)
            text = routing_text(c.messages)
            tp = artifact.task_probs(text)
            stages["features"] += time_it(lambda: extract_features(c.messages, params), 1)
            stages["task_classifier"] += time_it(lambda: artifact.task_probs(text), 1)
            stages["quality_models"] += time_it(
                lambda: artifact.quality_probs(f, tp, artifact.default_variant), 1
            )
            stages["decide_learned"] += time_it(
                lambda: engine.decide("learned", c.messages, params, quality_threshold=0.8), 1
            )
            stages["decide_heuristic"] += time_it(
                lambda: engine.decide("heuristic", c.messages, params, quality_threshold=0.8), 1
            )
            stages["decide_premium"] += time_it(
                lambda: engine.decide("premium", c.messages, params, quality_threshold=0.8), 1
            )
    inproc = {k: pct(v) for k, v in stages.items()}
    e2e_res = (
        asyncio.run(e2e(args.base_url, cases, args.explicit_model, 2))
        if args.base_url
        else {"skipped": "no --base-url"}
    )

    learned = inproc["decide_learned"]
    headline = [
        {
            "label": "Learned routing decision (p50)",
            "value": f"{learned['p50_ms']:.2f} ms",
            "detail": f"p95 {learned['p95_ms']:.2f} ms, in-process",
        },
        {
            "label": "Heuristic decision (p50)",
            "value": f"{inproc['decide_heuristic']['p50_ms']:.2f} ms",
        },
        {
            "label": "Fast-path compiled models",
            "value": f"{artifact.compiled_models}",
            "detail": "calibrated logistic models served by the NumPy fast path",
        },
    ]
    if "skipped" not in e2e_res:
        headline.append(
            {
                "label": "Client-observed added p50 (learned vs pinned)",
                "value": f"{e2e_res['routeforge-learned']['added_p50_ms_vs_pinned']:+.2f} ms",
                "detail": "replay mode, HTTP, sequential",
            }
        )
    summary = {
        "metadata": experiment_metadata(
            "E_router_overhead",
            mode="in-process + replay",
            registry_version=registry.version,
            router_artifact=artifact.version,
            n_prompts=len(cases),
            repeats=args.repeats,
            threads="BLAS/OpenMP limited to 1",
            optimisation_history=[
                "v0: CalibratedClassifierCV 5-fold ensembles + HistGradientBoosting with default OpenMP threads + char n-gram TF-IDF: 362 ms per learned decision (measured under load during development)",
                "v1: OMP_NUM_THREADS=1: 51 ms",
                "v2: ensemble=False calibration, word-only TF-IDF, NumPy fast path for calibrated logistic models, linear-model tie-break: see results",
            ],
        ),
        "status": "complete",
        "headline": headline,
        "overhead": {
            **{f"inprocess_{k}": v for k, v in inproc.items()},
            **({f"e2e_{k}": v for k, v in e2e_res.items()} if "skipped" not in e2e_res else {}),
        },
        "e2e_note": e2e_res.get("skipped"),
        "limitations": [
            "Laptop measurement; other processes (e.g. local Ollama inference) may share the CPU.",
            "End-to-end comparison runs in replay mode so provider latency is zero.",
        ],
    }
    write_summary("E_router_overhead", summary)
    for k, v in inproc.items():
        print(
            f"{k:18s} p50 {v['p50_ms']:.3f} ms  p95 {v['p95_ms']:.3f} ms  p99 {v['p99_ms']:.3f} ms"
        )
    print("e2e:", e2e_res)


if __name__ == "__main__":
    main()
