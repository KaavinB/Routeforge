"""Centralized, environment-driven configuration.

Every setting can be overridden with an environment variable prefixed ``ROUTEFORGE_``
(e.g. ``ROUTEFORGE_DEFAULT_STRATEGY=heuristic``). Provider credentials use the
provider-standard names (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) and are read by
LiteLLM directly; RouteForge never logs or returns them.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Environment(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class CacheMode(StrEnum):
    # Only cache requests that explicitly ask for temperature=0 (client opted into
    # deterministic semantics). Default.
    DETERMINISTIC = "deterministic"
    # Cache every eligible request; a hit returns one previously sampled answer.
    ALL = "all"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTEFORGE_", env_file=".env", extra="ignore", case_sensitive=False
    )

    env: Environment = Environment.DEV
    service_name: str = "routeforge"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_json: bool = True

    # --- storage -------------------------------------------------------------
    database_url: str = (
        "postgresql+asyncpg://routeforge_app:routeforge_app@localhost:5432/routeforge"
    )
    redis_url: str = "redis://localhost:6379/0"
    redis_socket_timeout_s: float = 0.25

    # --- models & routing -----------------------------------------------------
    model_config_path: Path = REPO_ROOT / "config" / "models" / "mock.yaml"
    router_artifact_path: Path | None = None
    default_strategy: str = "heuristic"
    default_quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    default_latency_budget_ms: float | None = None
    # Extra safety margin added to the calibrated margin when a request looks
    # out-of-distribution relative to router training data.
    router_uncertainty_margin: float = 0.10
    random_seed: int = 1234
    # How to treat sampling params a model rejects (e.g. temperature on Claude 5.5):
    # "drop" removes them and reports them in the x-routeforge-dropped-params header;
    # "reject" returns HTTP 400.
    unsupported_param_policy: str = Field(default="drop", pattern="^(drop|reject)$")

    # --- request limits -------------------------------------------------------
    max_request_bytes: int = 512_000
    max_messages: int = 64
    max_output_tokens_cap: int = 4096
    request_deadline_s: float = 120.0
    global_max_concurrency: int = 256
    concurrency_wait_s: float = 2.0

    # --- caching --------------------------------------------------------------
    cache_enabled: bool = True
    cache_mode: CacheMode = CacheMode.DETERMINISTIC
    cache_ttl_s: int = 3600
    cache_max_value_bytes: int = 262_144

    # --- reliability ----------------------------------------------------------
    retry_max_attempts_per_model: int = 2
    retry_base_delay_s: float = 0.25
    retry_max_delay_s: float = 4.0
    # Retries may consume at most this fraction of successful-request volume
    # (plus retry_budget_min_per_s), bounding cost amplification during incidents.
    retry_budget_ratio: float = 0.2
    retry_budget_min_per_s: float = 1.0
    max_fallbacks: int = 2
    breaker_failure_threshold: int = 5
    breaker_failure_rate: float = 0.5
    breaker_window: int = 20
    breaker_min_calls: int = 10
    breaker_reset_timeout_s: float = 30.0

    # --- rate limiting --------------------------------------------------------
    default_rate_limit_rpm: int = 600
    default_key_max_concurrency: int = 32
    rate_limit_fail_open: bool = True

    # --- security -------------------------------------------------------------
    api_key_pepper: SecretStr = SecretStr("dev-only-pepper-change-me")
    bootstrap_admin_key: SecretStr | None = None
    bootstrap_client_key: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]
    metrics_require_auth: bool = False

    # --- telemetry ------------------------------------------------------------
    log_prompts: bool = False
    otel_enabled: bool = True
    otel_exporter_endpoint: str | None = None
    langfuse_enabled: bool = True
    request_log_queue_size: int = 10_000
    request_log_batch_size: int = 200

    # --- fault injection & evaluation -----------------------------------------
    fault_injection_enabled: bool = False
    mock_latency_scale: float = 1.0
    replay_recordings_path: Path | None = None
    datasets_dir: Path = REPO_ROOT / "router_training" / "datasets" / "data"
    benchmark_results_dir: Path = REPO_ROOT / "benchmarks" / "results"
    eval_sandbox_backend: str = Field(default="subprocess", pattern="^(subprocess|docker)$")
    judge_model: str = "openai/gpt-4.1"

    @property
    def is_prod(self) -> bool:
        return self.env == Environment.PROD


@lru_cache
def get_settings() -> Settings:
    return Settings()
