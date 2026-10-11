"""OpenAI-compatible inference endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_principal, get_state
from app.schemas.openai import ChatCompletionRequest, ModelCard, ModelList
from app.services.auth import Principal
from app.services.gateway import RequestContext
from app.state import AppState

router = APIRouter(prefix="/v1", tags=["inference"])


def _ctx(request: Request, principal: Principal) -> RequestContext:
    cache_header = request.headers.get("x-routeforge-cache", "").lower()
    return RequestContext(
        principal=principal,
        request_id=request.state.request_id,
        cache_bypass=cache_header == "bypass",
        no_store=request.headers.get("x-routeforge-no-store", "").lower() in ("1", "true"),
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
    state: AppState = Depends(get_state),
) -> JSONResponse | StreamingResponse:
    ctx = _ctx(request, principal)
    if body.stream:
        headers, stream = await state.gateway.chat_stream(body, ctx)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={**headers, "cache-control": "no-cache", "x-accel-buffering": "no"},
        )
    result = await state.gateway.chat(body, ctx)
    return JSONResponse(result.response.model_dump(exclude_none=True), headers=result.headers)


@router.get("/models", response_model=ModelList)
async def list_models(
    _: Principal = Depends(get_principal), state: AppState = Depends(get_state)
) -> ModelList:
    cards = [ModelCard(id=name, owned_by="routeforge") for name in state.engine.virtual_models()]
    cards += [ModelCard(id=m.id, owned_by=m.family) for m in state.registry.enabled()]
    return ModelList(data=cards)
