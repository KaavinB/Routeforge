"""OpenAI-compatible request/response schemas (Chat Completions subset).

Compatibility policy: parameters RouteForge cannot honour are rejected with HTTP 400
rather than silently ignored (see docs/ARCHITECTURE.md#api-compatibility). Tool calling,
multimodal inputs, ``logprobs`` and ``n > 1`` are currently unsupported.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "developer", "user", "assistant"]


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str | list[TextContentPart]
    name: str | None = Field(default=None, max_length=64)

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict[str, Any] | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class RouteForgeOptions(BaseModel):
    """RouteForge extension, passed as ``extra_body={"routeforge": {...}}`` by OpenAI SDKs."""

    model_config = ConfigDict(extra="forbid")
    strategy: str | None = None
    quality_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_budget_ms: float | None = Field(default=None, gt=0)
    cache: bool | None = None  # False = bypass cache read & write for this request
    tags: list[str] | None = Field(default=None, max_length=8)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = None
    response_format: ResponseFormat | None = None
    user: str | None = Field(default=None, max_length=256)
    routeforge: RouteForgeOptions | None = None

    @field_validator("n")
    @classmethod
    def _single_choice(cls, v: int) -> int:
        if v != 1:
            raise ValueError("n > 1 is not supported by RouteForge")
        return v

    @field_validator("stop")
    @classmethod
    def _stop_limit(cls, v: str | list[str] | None) -> str | list[str] | None:
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("at most 4 stop sequences are supported")
        return v

    @model_validator(mode="after")
    def _token_alias(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("specify only one of max_tokens / max_completion_tokens")
        return self

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_tokens if self.max_tokens is not None else self.max_completion_tokens

    def sampling_params(self) -> dict[str, Any]:
        """Generation parameters explicitly set by the client (None values omitted)."""
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.effective_max_tokens,
            "stop": self.stop,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "response_format": (
                self.response_format.model_dump(exclude_none=True) if self.response_format else None
            ),
        }
        return {k: v for k, v in params.items() if v is not None}


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None
    logprobs: None = None


class RouteForgeMeta(BaseModel):
    """Non-standard response extension; OpenAI SDKs expose it via ``response.model_extra``."""

    request_id: str
    strategy: str
    selected_model: str
    cache: str
    cost_usd: float
    fallback_used: bool
    simulated: bool
    dropped_params: list[str] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None
    routeforge: RouteForgeMeta | None = None


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
