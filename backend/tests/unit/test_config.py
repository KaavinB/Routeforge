from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.registry import ModelRegistry, expand_env


def test_empty_env_values_mean_unset(monkeypatch) -> None:
    monkeypatch.setenv("ROUTEFORGE_REPLAY_RECORDINGS_PATH", "")
    monkeypatch.setenv("ROUTEFORGE_ROUTER_ARTIFACT_PATH", "  ")
    monkeypatch.setenv("ROUTEFORGE_BOOTSTRAP_ADMIN_KEY", "")
    s = Settings(_env_file=None)
    assert s.replay_recordings_path is None
    assert s.router_artifact_path is None
    assert s.bootstrap_admin_key is None


def test_registry_env_expansion(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    assert (
        expand_env("${OLLAMA_BASE_URL:-http://localhost:11434}")
        == "http://host.docker.internal:11434"
    )
    monkeypatch.delenv("OLLAMA_BASE_URL")
    assert expand_env("${OLLAMA_BASE_URL:-http://localhost:11434}") == "http://localhost:11434"


def test_all_registries_load() -> None:
    root = Path(__file__).resolve().parents[3] / "config" / "models"
    for path in root.glob("*.yaml"):
        reg = ModelRegistry.from_yaml(path)
        assert reg.premium_model in {m.id for m in reg.all()}
        assert reg.data.pricing.sources, f"{path.name} must document its pricing source"
