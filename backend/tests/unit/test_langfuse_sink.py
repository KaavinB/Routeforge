from __future__ import annotations

from typing import Any

from app.telemetry.langfuse_sink import LangfuseSink


class FakeGeneration:
    def __init__(self) -> None:
        self.ended = False

    def end(self) -> None:
        self.ended = True


class FakeLangfuse:
    """Records calls with the Langfuse v4 SDK's ``start_observation`` signature."""

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.generation = FakeGeneration()

    def start_observation(self, **kwargs: Any) -> FakeGeneration:
        self.observations.append(kwargs)
        return self.generation


def make_sink(client: Any) -> LangfuseSink:
    sink = LangfuseSink(enabled=False, environment="test", release="0")
    sink._client = client
    return sink


def test_disabled_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not LangfuseSink(enabled=True, environment="test", release="0").enabled


def test_records_generation_on_gateway_trace() -> None:
    fake = FakeLangfuse()
    sink = make_sink(fake)
    sink.record_generation(
        trace_id="0123456789abcdef0123456789abcdef",
        name="routeforge.learned",
        model="claude-haiku-5-5",
        input_messages=None,  # prompts are not sent unless prompt logging is enabled
        output=None,
        input_tokens=120,
        output_tokens=30,
        cost_usd=0.000027,
        completion_start_time=None,
        metadata={"strategy": "learned", "cache": "miss"},
    )
    obs = fake.observations[0]
    assert obs["as_type"] == "generation"
    assert obs["trace_context"] == {"trace_id": "0123456789abcdef0123456789abcdef"}
    assert obs["usage_details"] == {"input": 120, "output": 30}
    assert obs["cost_details"] == {"total": 0.000027}
    assert obs["input"] is None and obs["output"] is None
    assert obs["level"] == "DEFAULT"
    assert fake.generation.ended


def test_errors_are_marked_and_sink_failures_never_raise() -> None:
    fake = FakeLangfuse()
    sink = make_sink(fake)
    sink.record_generation(
        trace_id=None,
        name="x",
        model="m",
        input_messages=None,
        output=None,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        completion_start_time=None,
        metadata={},
        error="upstream_timeout",
    )
    assert fake.observations[0]["level"] == "ERROR"
    assert fake.observations[0]["trace_context"] is None

    class Broken:
        def start_observation(self, **_: Any) -> None:
            raise RuntimeError("langfuse down")

    make_sink(Broken()).record_generation(  # must not propagate: observability never breaks serving
        trace_id=None,
        name="x",
        model="m",
        input_messages=None,
        output=None,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        completion_start_time=None,
        metadata={},
    )
