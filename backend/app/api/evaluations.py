from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import get_state, require_admin
from app.core.errors import GatewayError, InvalidRequestError
from app.evaluation import regression, service
from app.state import AppState

router = APIRouter(
    prefix="/api/evaluations", tags=["evaluations"], dependencies=[Depends(require_admin)]
)


class RunRequest(BaseModel):
    dataset: str = Field(default="routeforge-bench-v1", pattern=r"^[A-Za-z0-9._-]+$")
    split: str = Field(default="test", pattern="^(train|val|test)$")
    targets: list[str] = Field(min_length=1, max_length=10)
    categories: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=5000)
    quality_threshold: float | None = Field(default=None, ge=0, le=1)
    use_judge: bool = False  # paid LLM-judge calls are opt-in
    concurrency: int = Field(default=4, ge=1, le=64)


@router.post("/run", status_code=202)
async def run(body: RunRequest, state: AppState = Depends(get_state)) -> dict[str, Any]:
    if state.db is None:
        raise GatewayError("Database required for evaluation runs.", status_code=503)
    try:
        ids = await service.start_run(state, **body.model_dump())
    except (FileNotFoundError, ValueError) as exc:
        raise InvalidRequestError(f"Cannot start evaluation: {exc}") from None
    return {"run_ids": ids, "status": "running"}


@router.get("/results")
async def results(
    limit: int = Query(50, ge=1, le=500), state: AppState = Depends(get_state)
) -> dict[str, Any]:
    return {"runs": await service.list_runs(state, limit)}


@router.get("/results/{run_id}")
async def result_detail(run_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"run_id": run_id, "cases": await service.run_results(state, run_id)}


class CompareRequest(BaseModel):
    reference_run_id: str
    candidate_run_id: str


@router.post("/compare")
async def compare(body: CompareRequest, state: AppState = Depends(get_state)) -> dict[str, Any]:
    ref = await service.run_results(state, body.reference_run_id)
    cand = await service.run_results(state, body.candidate_run_id)
    return regression.evaluate_gates(ref, cand)
