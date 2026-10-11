"""Background evaluation runs triggered through the API and persisted to PostgreSQL."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Any

from sqlalchemy import insert, select, update

from app.core.config import REPO_ROOT
from app.evaluation.budget import SpendLedger
from app.evaluation.datasets import resolve_dataset
from app.evaluation.judge import SummaryJudge
from app.evaluation.runner import InProcessClient, TargetOptions, run_eval, summarize
from app.evaluation.scoring import Scorer
from app.models.db import EvaluationResult, EvaluationRun, utcnow
from app.state import AppState

logger = logging.getLogger(__name__)
_TASKS: set[asyncio.Task[None]] = set()


def git_sha() -> str | None:
    env_sha = os.environ.get("ROUTEFORGE_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


async def start_run(
    state: AppState,
    *,
    dataset: str,
    split: str,
    targets: list[str],
    categories: list[str] | None,
    limit: int | None,
    quality_threshold: float | None,
    use_judge: bool,
    concurrency: int,
) -> list[str]:
    assert state.db is not None
    ds = resolve_dataset(state.settings.datasets_dir, dataset)
    cases = ds.select(split, categories, limit)
    run_ids: list[str] = []
    async with state.db.sessionmaker() as session:
        for target in targets:
            run = EvaluationRun(
                name=f"{dataset}:{split}:{target}",
                dataset=ds.name,
                dataset_version=ds.version,
                split=split,
                target=target,
                status="running",
                config={
                    "categories": categories,
                    "limit": limit,
                    "n_cases": len(cases),
                    "quality_threshold": quality_threshold,
                    "judge": use_judge,
                    "registry": state.registry.version,
                    "mode": state.registry.mode,
                    "router_artifact": state.engine.artifact.version
                    if state.engine.artifact
                    else None,
                },
                git_sha=git_sha(),
            )
            session.add(run)
            await session.flush()
            run_ids.append(run.id)
        await session.commit()

    async def _execute() -> None:
        judge = None
        if use_judge:
            judge = SummaryJudge(
                state.settings.judge_model,
                cache_path=REPO_ROOT / "router_training" / "data" / "judge_cache.jsonl",
                ledger=SpendLedger(state.settings.benchmark_results_dir / "spend_ledger.jsonl"),
            )
        scorer = Scorer(judge=judge, sandbox_backend=state.settings.eval_sandbox_backend)
        client = InProcessClient(state.gateway)
        for run_id, target in zip(run_ids, targets, strict=True):
            try:
                outcomes = await run_eval(
                    cases,
                    [target],
                    client,
                    scorer,
                    opts=TargetOptions(quality_threshold=quality_threshold),
                    concurrency=concurrency,
                )
                summary = summarize(outcomes)[target]
                rows: list[dict[str, Any]] = [
                    {
                        "run_id": run_id,
                        "case_id": o.case_id,
                        "category": o.category,
                        "model": o.final_model,
                        "score": o.score,
                        "acceptable": o.acceptable,
                        "latency_ms": o.latency_ms,
                        "cost_usd": o.cost_usd,
                        "input_tokens": o.input_tokens,
                        "output_tokens": o.output_tokens,
                        "error": o.error,
                        "details": o.score_details,
                    }
                    for o in outcomes
                ]
                async with state.db.sessionmaker() as session:  # type: ignore[union-attr]
                    if rows:
                        await session.execute(insert(EvaluationResult), rows)
                    await session.execute(
                        update(EvaluationRun)
                        .where(EvaluationRun.id == run_id)
                        .values(status="done", summary=summary, finished_at=utcnow())
                    )
                    await session.commit()
            except Exception as exc:
                logger.exception("evaluation run %s failed", run_id)
                async with state.db.sessionmaker() as session:  # type: ignore[union-attr]
                    await session.execute(
                        update(EvaluationRun)
                        .where(EvaluationRun.id == run_id)
                        .values(status="failed", error=str(exc)[:2000], finished_at=utcnow())
                    )
                    await session.commit()

    task = asyncio.create_task(_execute())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return run_ids


async def list_runs(state: AppState, limit: int = 50) -> list[dict[str, Any]]:
    assert state.db is not None
    async with state.db.sessionmaker() as session:
        rows = (
            await session.execute(
                select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
            )
        ).scalars()
        return [
            {
                "id": r.id,
                "name": r.name,
                "dataset": r.dataset,
                "dataset_version": r.dataset_version,
                "split": r.split,
                "target": r.target,
                "status": r.status,
                "config": r.config,
                "summary": r.summary,
                "error": r.error,
                "git_sha": r.git_sha,
                "created_at": r.created_at,
                "finished_at": r.finished_at,
            }
            for r in rows
        ]


async def run_results(state: AppState, run_id: str) -> list[dict[str, Any]]:
    assert state.db is not None
    async with state.db.sessionmaker() as session:
        rows = (
            await session.execute(select(EvaluationResult).where(EvaluationResult.run_id == run_id))
        ).scalars()
        return [
            {
                "case_id": r.case_id,
                "category": r.category,
                "model": r.model,
                "score": r.score,
                "acceptable": r.acceptable,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
                "error": r.error,
                "score_details": r.details,
            }
            for r in rows
        ]
