"""Model registry: the single source of truth for which models exist, how to reach
them, what they cost, and which request parameters they accept.

Loaded from a versioned YAML file (``config/models/*.yaml``). Pricing lives next to the
model definition together with its source and retrieval date so every cost figure in
logs, dashboards and benchmark reports can be traced to an explicit assumption.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ProviderKind = Literal["litellm", "mock", "replay"]


_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references (e.g. the Ollama base URL, which
    differs between the host and a container)."""
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)


class MockProfile(BaseModel):
    """Simulated behaviour for mock models. Timings are synthetic, never measurements."""

    ttft_ms: float = 150.0
    tokens_per_s: float = 80.0
    typical_output_tokens: int = 60


class ModelSpec(BaseModel):
    id: str
    provider: ProviderKind
    upstream: str | None = None  # LiteLLM model string, e.g. "openai/gpt-4.1-nano"
    family: str  # vendor family used for reporting, e.g. "openai", "anthropic", "ollama"
    tier: int = Field(ge=0)  # 0 = cheapest; used for heuristic ordering only
    enabled: bool = True
    context_window: int = 128_000
    max_output_tokens: int = 4096
    input_cost_per_mtok: float = Field(ge=0)
    output_cost_per_mtok: float = Field(ge=0)
    timeout_s: float = 60.0
    max_concurrency: int = 32
    api_base: str | None = None
    # Parameters this model accepts. Anything else is dropped (and reported) or rejected.
    supported_params: list[str] = Field(
        default_factory=lambda: [
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
        ]
    )
    # Extra provider-specific parameters always sent upstream (e.g. thinking config).
    provider_params: dict[str, Any] = Field(default_factory=dict)
    mock: MockProfile | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_upstream(self) -> ModelSpec:
        if self.provider == "litellm" and not self.upstream:
            raise ValueError(f"model {self.id}: litellm provider requires 'upstream'")
        return self

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_mtok + output_tokens * self.output_cost_per_mtok
        ) / 1_000_000


class PricingInfo(BaseModel):
    version: str
    currency: str = "USD"
    sources: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = (
        "Costs are estimates computed from token usage and list prices; "
        "they are not provider billing data."
    )


class RoutingDefaults(BaseModel):
    premium_model: str
    economy_model: str
    # Candidate pool considered by routed strategies (defaults to all enabled models).
    candidates: list[str] | None = None


class RegistryFile(BaseModel):
    version: str
    mode: Literal["mock", "local", "hosted", "replay"]
    simulated: bool = False
    pricing: PricingInfo
    routing: RoutingDefaults
    models: list[ModelSpec]

    @field_validator("models")
    @classmethod
    def _unique_ids(cls, v: list[ModelSpec]) -> list[ModelSpec]:
        ids = [m.id for m in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model ids in registry")
        return v


class ModelRegistry:
    def __init__(self, data: RegistryFile, source_path: Path | None = None) -> None:
        self.data = data
        self.source_path = source_path
        self._models = {m.id: m for m in data.models}
        for ref in (data.routing.premium_model, data.routing.economy_model):
            if ref not in self._models:
                raise ValueError(f"routing default {ref!r} is not a registered model")
        # A content hash of the registry participates in cache keys, so changing any
        # model/provider configuration naturally invalidates previously cached outputs.
        raw = data.model_dump_json().encode()
        self.fingerprint = hashlib.sha256(raw).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: Path) -> ModelRegistry:
        raw = yaml.safe_load(expand_env(path.read_text()))
        return cls(RegistryFile.model_validate(raw), source_path=path)

    @property
    def version(self) -> str:
        return self.data.version

    @property
    def mode(self) -> str:
        return self.data.mode

    @property
    def simulated(self) -> bool:
        return self.data.simulated

    @property
    def premium_model(self) -> str:
        return self.data.routing.premium_model

    @property
    def economy_model(self) -> str:
        return self.data.routing.economy_model

    def get(self, model_id: str) -> ModelSpec | None:
        return self._models.get(model_id)

    def require(self, model_id: str) -> ModelSpec:
        spec = self._models.get(model_id)
        if spec is None:
            raise KeyError(model_id)
        return spec

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def enabled(self) -> list[ModelSpec]:
        return [m for m in self._models.values() if m.enabled]

    def candidates(self) -> list[ModelSpec]:
        pool = self.data.routing.candidates
        models = self.enabled()
        if pool is not None:
            models = [m for m in models if m.id in pool]
        return sorted(models, key=lambda m: (m.tier, m.id))
