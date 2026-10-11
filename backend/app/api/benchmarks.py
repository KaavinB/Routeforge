"""Serves benchmark result artifacts (JSON files written by the benchmarks/ suite) so the
dashboard displays exactly what the scripts measured — no hand-entered numbers."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import get_state, require_admin
from app.core.errors import NotFoundError
from app.state import AppState

router = APIRouter(
    prefix="/api/benchmarks", tags=["benchmarks"], dependencies=[Depends(require_admin)]
)
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@router.get("")
async def list_benchmarks(state: AppState = Depends(get_state)) -> dict[str, Any]:
    root = state.settings.benchmark_results_dir
    items = []
    if root.exists():
        for path in sorted(root.glob("*/summary.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            meta = data.get("metadata", {})
            items.append(
                {
                    "name": path.parent.name,
                    "benchmark": meta.get("benchmark"),
                    "experiment_id": meta.get("experiment_id"),
                    "timestamp": meta.get("timestamp"),
                    "git_sha": meta.get("git_sha"),
                    "mode": meta.get("mode"),
                    "status": data.get("status", "complete"),
                }
            )
    return {"benchmarks": items}


@router.get("/{name}")
async def get_benchmark(name: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if not _NAME.match(name):
        raise NotFoundError("Unknown benchmark.")
    path = state.settings.benchmark_results_dir / name / "summary.json"
    if not path.exists():
        raise NotFoundError("Unknown benchmark.")
    data: dict[str, Any] = json.loads(path.read_text())
    return data
