"""Step 2 of the router pipeline: run every dataset prompt on every candidate model.

    uv run python -m router_training.collect --registry config/models/hosted.yaml --dry-run
    uv run python -m router_training.collect --registry config/models/hosted.yaml

* Calls the same provider adapters and the same per-model parameter adaptation as the
  gateway (temperature=0 where supported), without the gateway's cache, so outputs are
  fresh.
* Resumable: completed (case, model) pairs in the output file are skipped.
* Paid calls are checked against the spend ledger cap (ROUTEFORGE_PAID_BUDGET_USD)
  using a conservative pre-call estimate (input estimate + full max_tokens), and the
  actual cost is recorded after each call.
* Latency is wall-clock per request measured inside the concurrency slot.

Output: router_training/data/collections/<registry-version>/outputs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.errors import ErrorKind, ProviderError
from app.core.registry import ModelRegistry, ModelSpec
from app.evaluation.budget import BudgetExceededError, SpendLedger
from app.evaluation.datasets import Dataset, EvalCase
from app.providers.base import ProviderRequest, messages_fingerprint
from app.providers.factory import ProviderPool
from app.reliability.executor import adapt_params
from app.routing.tokens import estimate_message_tokens

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets" / "data"
COLLECTIONS = ROOT / "data" / "collections"
LEDGER = ROOT.parent / "benchmarks" / "results" / "spend_ledger.jsonl"
FAMILY_CONCURRENCY = {"openai": 8, "anthropic": 6, "ollama": 1, "mock": 16}


def estimate_cost(spec: ModelSpec, case: EvalCase) -> float:
    return spec.cost_usd(int(estimate_message_tokens(case.messages) * 1.3), case.max_tokens)


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    done.add((row["case_id"], row["model_id"]))
    return done


async def collect(
    registry: ModelRegistry,
    cases: list[EvalCase],
    out_path: Path,
    *,
    models: list[str] | None,
    ledger: SpendLedger | None,
    attempts: int = 3,
) -> dict[str, Any]:
    pool = ProviderPool(registry)
    specs = [m for m in registry.candidates() if not models or m.id in models]
    done = load_done(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sems = {fam: asyncio.Semaphore(n) for fam, n in FAMILY_CONCURRENCY.items()}
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lock = asyncio.Lock()
    stop = asyncio.Event()

    async def one(case: EvalCase, spec: ModelSpec) -> None:
        if stop.is_set():
            return
        params, dropped = adapt_params(spec, case.request_params(), cap=4096)
        req = ProviderRequest(spec, case.messages, params, timeout_s=spec.timeout_s)
        paid = spec.input_cost_per_mtok > 0 and spec.family in ("openai", "anthropic")
        row: dict[str, Any] = {
            "case_id": case.id,
            "split": case.split,
            "category": case.category,
            "model_id": spec.id,
            "messages_sha256": messages_fingerprint(case.messages),
            "params": params,
            "dropped_params": dropped,
        }
        async with sems.get(spec.family, sems["mock"]):
            if paid and ledger is not None:
                try:
                    ledger.check(estimate_cost(spec, case))
                except BudgetExceededError as exc:
                    print(f"STOP: {exc}")
                    stop.set()
                    return
            provider = pool.for_model(spec)
            last_err: ProviderError | None = None
            for attempt in range(1, attempts + 1):
                started = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        provider.complete(req), timeout=spec.timeout_s + 5
                    )
                except (ProviderError, TimeoutError) as exc:
                    last_err = (
                        exc
                        if isinstance(exc, ProviderError)
                        else ProviderError(ErrorKind.TIMEOUT, "timeout")
                    )
                    if not last_err.retryable or attempt == attempts:
                        break
                    await asyncio.sleep(2**attempt)
                    continue
                cost = spec.cost_usd(resp.input_tokens, resp.output_tokens)
                row.update(
                    content=resp.content,
                    finish_reason=resp.finish_reason,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    latency_s=round(time.perf_counter() - started, 4),
                    cost_usd=cost,
                    upstream_model=resp.upstream_model,
                    attempts=attempt,
                    error=None,
                    ts=datetime.now(UTC).isoformat(),
                )
                if paid and ledger is not None:
                    ledger.record(
                        activity="router_training_collect",
                        model=spec.id,
                        cost_usd=cost,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                    )
                break
            if "content" not in row:
                assert last_err is not None
                row.update(
                    error=f"{last_err.kind.value}: {last_err.message[:300]}",
                    attempts=attempts,
                    ts=datetime.now(UTC).isoformat(),
                )
        async with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            s = stats[spec.id]
            s["n"] += 1
            s["errors"] += 1 if row.get("error") else 0
            s["cost_usd"] += row.get("cost_usd") or 0.0
            total = sum(v["n"] for v in stats.values())
            if total % 50 == 0:
                spent = f"${ledger.spent:.3f}" if ledger else "n/a"
                print(f"  progress {total} calls; ledger spent {spent}", flush=True)

    todo = [(c, s) for s in specs for c in cases if (c.id, s.id) not in done]
    print(f"{len(todo)} calls to make ({len(done)} already done)")
    # Model-major ordering keeps local (Ollama) models from thrashing GPU memory.
    by_model: dict[str, list[tuple[EvalCase, ModelSpec]]] = defaultdict(list)
    for c, s in todo:
        by_model[s.id].append((c, s))
    local = [m for m in by_model if registry.require(m).family == "ollama"]
    remote = [m for m in by_model if m not in local]
    await asyncio.gather(*(one(c, s) for m in remote for c, s in by_model[m]))
    for m in local:
        await asyncio.gather(*(one(c, s) for c, s in by_model[m]))
    return {k: dict(v) for k, v in stats.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--dataset", default="routeforge-bench-v1")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    registry = ModelRegistry.from_yaml(args.registry)
    ds = Dataset(DATASETS / args.dataset)
    cases = ds.select(args.split)
    if args.limit:
        # stratified smoke subset: first N per category
        per: dict[str, int] = defaultdict(int)
        subset = []
        for c in cases:
            per[c.category] += 1
            if per[c.category] <= args.limit:
                subset.append(c)
        cases = subset
    specs = [m for m in registry.candidates() if not args.models or m.id in args.models]
    est = {s.id: sum(estimate_cost(s, c) for c in cases) for s in specs}
    print(f"dataset {ds.name}@{ds.version}: {len(cases)} cases x {len(specs)} models")
    print("pessimistic cost estimate (input x1.3, output = full max_tokens):")
    for k, v in est.items():
        print(f"  {k:24s} ${v:.3f}")
    print(f"  {'TOTAL':24s} ${sum(est.values()):.3f}")
    if args.dry_run:
        return
    out = COLLECTIONS / registry.version / "outputs.jsonl"
    ledger = SpendLedger(LEDGER) if registry.mode == "hosted" else None
    if ledger is not None:
        print(f"ledger: spent ${ledger.spent:.4f} of cap ${ledger.cap_usd:.2f}")
        if ledger.cap_usd <= 0:
            raise SystemExit("ROUTEFORGE_PAID_BUDGET_USD is not set; refusing to make paid calls")
    stats = asyncio.run(collect(registry, cases, out, models=args.models, ledger=ledger))
    print(json.dumps(stats, indent=2))
    meta = {
        "registry_version": registry.version,
        "registry_fingerprint": registry.fingerprint,
        "dataset": ds.name,
        "dataset_version": ds.version,
        "dataset_sha256": ds.manifest.sha256,
        "finished_at": datetime.now(UTC).isoformat(),
        "pricing": registry.data.pricing.model_dump(),
    }
    (out.parent / "collection_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
