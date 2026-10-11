"""API-key authentication and key management.

* Keys are stored as HMAC-SHA256 digests (``app.core.security``).
* Lookups are cached in-process for ``cache_ttl_s`` (positive and negative results) so
  authentication does not add a database round trip to every request.
* Bootstrap keys from the environment are always valid, even while the database is
  down, so operators retain access during an incident.
* Unknown keys during a database outage yield 503 (cannot verify), not 401.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import AuthenticationError, GatewayError
from app.core.security import generate_api_key, hash_api_key, key_display_prefix
from app.models.db import ApiKey, Database, utcnow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    key_id: str
    name: str
    role: str
    tenant: str
    rate_limit_rpm: int
    max_concurrency: int

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class AuthService:
    def __init__(
        self,
        db: Database | None,
        *,
        pepper: str,
        default_rpm: int,
        default_concurrency: int,
        bootstrap: dict[str, str] | None = None,  # role -> raw key
        cache_ttl_s: float = 60.0,
    ) -> None:
        self.db = db
        self.pepper = pepper
        self.default_rpm = default_rpm
        self.default_concurrency = default_concurrency
        self.cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[Principal | None, float]] = {}
        self._static: dict[str, Principal] = {}
        for role, raw in (bootstrap or {}).items():
            digest = hash_api_key(raw, pepper)
            self._static[digest] = Principal(
                key_id=f"bootstrap-{role}",
                name=f"bootstrap {role}",
                role=role,
                tenant="default",
                rate_limit_rpm=default_rpm * (10 if role == "admin" else 1),
                max_concurrency=default_concurrency,
            )

    async def authenticate(self, raw_key: str | None) -> Principal:
        if not raw_key:
            raise AuthenticationError("Missing API key. Send 'Authorization: Bearer <key>'.")
        digest = hash_api_key(raw_key, self.pepper)
        if digest in self._static:
            return self._static[digest]
        cached = self._cache.get(digest)
        if cached is not None and cached[1] > time.monotonic():
            if cached[0] is None:
                raise AuthenticationError("Invalid API key.")
            return cached[0]
        if self.db is None:
            raise AuthenticationError("Invalid API key.")
        try:
            async with self.db.sessionmaker() as session:
                row = (
                    await session.execute(select(ApiKey).where(ApiKey.key_hash == digest))
                ).scalar_one_or_none()
        except (SQLAlchemyError, OSError, TimeoutError) as exc:
            logger.warning("auth backend unavailable: %s", type(exc).__name__)
            raise GatewayError(
                "Authentication backend unavailable.",
                status_code=503,
                error_type="service_unavailable",
            ) from None
        principal = None
        if row is not None and row.revoked_at is None:
            principal = Principal(
                key_id=row.id,
                name=row.name,
                role=row.role,
                tenant=row.tenant,
                rate_limit_rpm=row.rate_limit_rpm,
                max_concurrency=row.max_concurrency,
            )
        self._cache[digest] = (principal, time.monotonic() + self.cache_ttl_s)
        if principal is None:
            raise AuthenticationError("Invalid API key.")
        return principal

    async def create_key(
        self,
        *,
        name: str,
        role: str = "client",
        tenant: str = "default",
        rate_limit_rpm: int | None = None,
        max_concurrency: int | None = None,
    ) -> tuple[str, ApiKey]:
        assert self.db is not None
        raw = generate_api_key()
        row = ApiKey(
            name=name,
            key_prefix=key_display_prefix(raw),
            key_hash=hash_api_key(raw, self.pepper),
            role=role,
            tenant=tenant,
            rate_limit_rpm=rate_limit_rpm or self.default_rpm,
            max_concurrency=max_concurrency or self.default_concurrency,
        )
        async with self.db.sessionmaker() as session:
            session.add(row)
            await session.commit()
        return raw, row

    async def list_keys(self) -> list[ApiKey]:
        assert self.db is not None
        async with self.db.sessionmaker() as session:
            return list(
                (await session.execute(select(ApiKey).order_by(ApiKey.created_at))).scalars()
            )

    async def revoke(self, key_id: str) -> bool:
        assert self.db is not None
        async with self.db.sessionmaker() as session:
            result = await session.execute(
                update(ApiKey)
                .where(ApiKey.id == key_id, ApiKey.revoked_at.is_(None))
                .values(revoked_at=utcnow())
                .returning(ApiKey.key_hash)
            )
            digests = [r[0] for r in result]
            await session.commit()
        for d in digests:
            self._cache.pop(d, None)
        return bool(digests)
