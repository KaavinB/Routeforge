"""Alembic environment (async). Migrations run as the schema-owner role; the application
connects as a least-privilege role that only has DML rights (see infrastructure/postgres)."""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.models.db import Base

config = context.config
target_metadata = Base.metadata


def database_url() -> str:
    url = os.environ.get("ROUTEFORGE_MIGRATION_DATABASE_URL") or os.environ.get("ROUTEFORGE_DATABASE_URL")
    if not url:
        raise RuntimeError("set ROUTEFORGE_MIGRATION_DATABASE_URL to run migrations")
    return url


def run_migrations_offline() -> None:
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url())
    async with engine.connect() as conn:
        await conn.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
