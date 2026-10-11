"""SQLAlchemy 2.0 ORM models and async engine/session helpers."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JsonType = JSON().with_variant(JSONB(), "postgresql")


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    key_prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(16), default="client")  # client | admin
    tenant: Mapped[str] = mapped_column(String(64), default="default")
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, default=600)
    max_concurrency: Mapped[int] = mapped_column(Integer, default=32)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RequestLog(Base):
    """One row per chat-completion request. Prompt and completion text are NOT stored
    unless ROUTEFORGE_LOG_PROMPTS=true; only a digest of the messages is kept."""

    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    api_key_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    tenant: Mapped[str] = mapped_column(String(64), default="default")
    requested_model: Mapped[str] = mapped_column(String(128))
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    selected_model: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    final_model: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    routing_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    predicted_task: Mapped[str | None] = mapped_column(String(32), nullable=True)
    predicted_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    routing_detail: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    cache_status: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)  # success | error
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    http_status: Mapped[int] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    attempt_detail: Mapped[list[dict[str, Any]] | None] = mapped_column(JsonType, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float)
    routing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    queue_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cache_saved_usd: Mapped[float] = mapped_column(Float, default=0.0)
    messages_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JsonType, nullable=True)

    __table_args__ = (Index("ix_request_logs_created_strategy", "created_at", "strategy"),)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    dataset: Mapped[str] = mapped_column(String(128))
    dataset_version: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(16))
    target: Mapped[str] = mapped_column(String(128))  # model id or routeforge-<strategy>
    status: Mapped[str] = mapped_column(
        String(16), default="pending"
    )  # pending|running|done|failed
    config: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score: Mapped[float] = mapped_column(Float)
    acceptable: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String(256), nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        kwargs: dict[str, Any] = {"echo": echo, "pool_pre_ping": True}
        if url.startswith("postgresql"):
            kwargs.update(pool_size=10, max_overflow=10, pool_timeout=2.0)
            kwargs["connect_args"] = {"timeout": 2.0, "command_timeout": 5.0}
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as session:
            yield session

    async def create_all(self) -> None:
        """Used by tests (SQLite). Production schemas are managed by Alembic."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def ping(self) -> bool:
        from sqlalchemy import text

        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def dispose(self) -> None:
        await self.engine.dispose()
