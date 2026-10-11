"""Provider abstraction: one interface for every inference backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.registry import ModelSpec


@dataclass(frozen=True)
class ProviderRequest:
    spec: ModelSpec
    messages: list[dict[str, str]]
    params: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 60.0

    def prompt_fingerprint(self) -> str:
        """Stable hash of (model, messages, params) used by replay recordings."""
        return request_fingerprint(self.spec.id, self.messages, self.params)


def request_fingerprint(
    model_id: str, messages: list[dict[str, str]], params: dict[str, Any]
) -> str:
    # max_tokens/temperature are part of the identity of a recorded generation.
    payload = json.dumps(
        {"model": model_id, "messages": messages, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def messages_fingerprint(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ProviderResponse:
    content: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    simulated: bool = False
    upstream_model: str | None = None


@dataclass
class StreamChunk:
    delta: str = ""
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(Protocol):
    name: str

    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...

    def stream(self, request: ProviderRequest) -> AsyncIterator[StreamChunk]: ...
