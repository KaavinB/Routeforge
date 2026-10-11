"""Replay provider: serves previously recorded real model outputs.

Recordings are produced by ``router_training.collect`` against real models. Replaying
them lets CI run the evaluation and regression-gate pipeline deterministically on genuine
model outputs without making paid API calls. Latency is reported as the originally
recorded latency (no sleeping), and responses are flagged ``simulated=False`` because the
*content* is real, but every replay run is labelled as a replay in its metadata.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app.core.errors import ErrorKind, ProviderError
from app.providers.base import (
    ProviderRequest,
    ProviderResponse,
    StreamChunk,
    messages_fingerprint,
)


class ReplayProvider:
    name = "replay"

    def __init__(self, recordings: dict[tuple[str, str], dict[str, Any]]) -> None:
        self._recordings = recordings

    @classmethod
    def from_jsonl(cls, path: Path) -> ReplayProvider:
        recordings: dict[tuple[str, str], dict[str, Any]] = {}
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error"):
                    continue
                recordings[(row["model_id"], row["messages_sha256"])] = row
        return cls(recordings)

    def __len__(self) -> int:
        return len(self._recordings)

    def _lookup(self, request: ProviderRequest) -> dict[str, Any]:
        key = (request.spec.id, messages_fingerprint(request.messages))
        row = self._recordings.get(key)
        if row is None:
            raise ProviderError(
                ErrorKind.BAD_REQUEST,
                "no recording for this (model, messages) pair",
                model_id=request.spec.id,
            )
        return row

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        row = self._lookup(request)
        return ProviderResponse(
            content=row["content"],
            finish_reason=row.get("finish_reason"),
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            latency_s=float(row["latency_s"]),
            upstream_model=row.get("upstream_model"),
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamChunk]:
        row = self._lookup(request)
        yield StreamChunk(delta=row["content"])
        yield StreamChunk(
            finish_reason=row.get("finish_reason"),
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
        )
