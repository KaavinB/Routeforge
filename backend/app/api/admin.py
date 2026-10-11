from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_state, require_admin
from app.core.errors import GatewayError, NotFoundError
from app.providers.faults import FaultRule
from app.state import AppState

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    role: str = Field(default="client", pattern="^(client|admin)$")
    tenant: str = Field(default="default", min_length=1, max_length=64)
    rate_limit_rpm: int | None = Field(default=None, ge=1, le=100_000)
    max_concurrency: int | None = Field(default=None, ge=1, le=1024)


@router.post("/keys", status_code=201)
async def create_key(
    body: CreateKeyRequest, state: AppState = Depends(get_state)
) -> dict[str, Any]:
    raw, row = await state.auth.create_key(**body.model_dump())
    return {
        "id": row.id,
        "key": raw,  # shown exactly once; only the hash is stored
        "prefix": row.key_prefix,
        "role": row.role,
        "tenant": row.tenant,
        "rate_limit_rpm": row.rate_limit_rpm,
        "max_concurrency": row.max_concurrency,
    }


@router.get("/keys")
async def list_keys(state: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    return [
        {
            "id": k.id,
            "name": k.name,
            "prefix": k.key_prefix,
            "role": k.role,
            "tenant": k.tenant,
            "rate_limit_rpm": k.rate_limit_rpm,
            "max_concurrency": k.max_concurrency,
            "created_at": k.created_at,
            "revoked_at": k.revoked_at,
        }
        for k in await state.auth.list_keys()
    ]


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, state: AppState = Depends(get_state)) -> dict[str, bool]:
    if not await state.auth.revoke(key_id):
        raise NotFoundError("Key not found or already revoked.")
    return {"revoked": True}


@router.delete("/cache")
async def flush_cache(state: AppState = Depends(get_state)) -> dict[str, int]:
    return {"deleted": await state.cache.invalidate_all()}


@router.get("/breakers")
async def breakers(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {
        "breakers": state.breakers.snapshot(),
        "retry_budget_tokens": state.retry_budget.available,
    }


@router.post("/breakers/reset")
async def reset_breakers(state: AppState = Depends(get_state)) -> dict[str, bool]:
    state.breakers.reset_all()
    return {"reset": True}


def _injector(state: AppState) -> Any:
    if state.fault_injector is None:
        raise GatewayError(
            "Fault injection is disabled (set ROUTEFORGE_FAULT_INJECTION_ENABLED=true).",
            status_code=409,
            error_type="conflict",
        )
    return state.fault_injector


@router.get("/faults")
async def get_faults(state: AppState = Depends(get_state)) -> dict[str, Any]:
    inj = _injector(state)
    return {"rules": {k: v.model_dump() for k, v in inj.rules().items()}, "injected": inj.injected}


@router.put("/faults/{model_id}")
async def set_fault(
    model_id: str, rule: FaultRule, state: AppState = Depends(get_state)
) -> dict[str, Any]:
    if state.registry.get(model_id) is None:
        raise NotFoundError(f"Unknown model {model_id}.")
    _injector(state).set_rule(model_id, rule)
    return {"model": model_id, "rule": rule.model_dump()}


@router.delete("/faults")
async def clear_faults(state: AppState = Depends(get_state)) -> dict[str, bool]:
    inj = _injector(state)
    inj.clear()
    inj.injected.clear()
    return {"cleared": True}
