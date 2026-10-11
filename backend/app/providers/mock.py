"""Deterministic mock provider (Mode A).

Everything this provider returns is SIMULATED and is flagged as such end-to-end
(``simulated=True`` in responses, logs and metrics). It exists for development, CI,
UI demos, load-testing the gateway itself, and fault injection — never as evidence of
model quality or real inference latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator

from app.core.registry import MockProfile
from app.providers.base import ProviderRequest, ProviderResponse, StreamChunk
from app.routing.tokens import estimate_message_tokens

_WORDS = [
    "routing",
    "latency",
    "quality",
    "budget",
    "cache",
    "model",
    "request",
    "token",
    "gateway",
    "policy",
    "fallback",
    "evaluation",
    "response",
    "provider",
    "signal",
    "metric",
    "trace",
    "inference",
    "estimate",
    "threshold",
]


class MockProvider:
    name = "mock"

    def __init__(self, latency_scale: float = 1.0) -> None:
        self.latency_scale = latency_scale

    @staticmethod
    def _profile(request: ProviderRequest) -> MockProfile:
        return request.spec.mock or MockProfile()

    def _render(self, request: ProviderRequest) -> tuple[str, int]:
        profile = self._profile(request)
        max_tokens = int(request.params.get("max_tokens") or profile.typical_output_tokens)
        n_tokens = max(1, min(max_tokens, profile.typical_output_tokens))
        rf = request.params.get("response_format") or {}
        if rf.get("type") in ("json_object", "json_schema"):
            text = json.dumps({"simulated": True, "model": request.spec.id})
            return text, max(1, len(text) // 4)
        digest = hashlib.sha256(
            (request.spec.id + json.dumps(request.messages, sort_keys=True)).encode()
        ).digest()
        words = [_WORDS[b % len(_WORDS)] for b in (digest * 8)[: max(1, n_tokens - 6)]]
        text = f"[SIMULATED {request.spec.id}] " + " ".join(words)
        return text, n_tokens

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        started = time.perf_counter()
        profile = self._profile(request)
        text, out_tokens = self._render(request)
        delay = (profile.ttft_ms / 1000 + out_tokens / profile.tokens_per_s) * self.latency_scale
        if delay > 0:
            await asyncio.sleep(delay)
        return ProviderResponse(
            content=text,
            finish_reason="stop",
            input_tokens=estimate_message_tokens(request.messages),
            output_tokens=out_tokens,
            latency_s=time.perf_counter() - started,
            simulated=True,
            upstream_model=request.spec.id,
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamChunk]:
        profile = self._profile(request)
        text, out_tokens = self._render(request)
        if self.latency_scale > 0:
            await asyncio.sleep(profile.ttft_ms / 1000 * self.latency_scale)
        pieces = text.split(" ")
        per_piece = out_tokens / max(1, len(pieces)) / profile.tokens_per_s * self.latency_scale
        for i, piece in enumerate(pieces):
            if per_piece > 0:
                await asyncio.sleep(per_piece)
            yield StreamChunk(delta=piece if i == 0 else " " + piece)
        yield StreamChunk(
            finish_reason="stop",
            input_tokens=estimate_message_tokens(request.messages),
            output_tokens=out_tokens,
        )
