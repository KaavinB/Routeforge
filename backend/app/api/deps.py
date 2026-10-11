from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import GatewayError, PermissionDeniedError
from app.services.auth import Principal
from app.state import AppState


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.rf
    return state


def _extract_key(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


async def get_principal(request: Request, state: AppState = Depends(get_state)) -> Principal:
    principal = await state.auth.authenticate(_extract_key(request))
    request.state.principal = principal
    return principal


async def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if not principal.is_admin:
        raise PermissionDeniedError("Admin API key required.")
    return principal


async def get_session(state: AppState = Depends(get_state)) -> AsyncIterator[AsyncSession]:
    if state.db is None:
        raise GatewayError(
            "Database not configured.", status_code=503, error_type="service_unavailable"
        )
    async with state.db.sessionmaker() as session:
        yield session
