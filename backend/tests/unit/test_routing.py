from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.registry import ModelRegistry
from app.routing.artifact import ARTIFACT_FORMAT_VERSION, ArtifactError, RouterArtifact, save_bundle
from app.routing.cost import CostModel
from app.routing.engine import RoutingEngine, UnknownStrategyError
from app.routing.features import (
    FEATURE_NAMES,
    TASK_CATEGORIES,
    extract_features,
    rule_based_task_probs,
)
from app.routing.latency import LatencyModel, LatencyPrior
from app.routing.tokens import estimate_message_tokens, estimate_tokens
from app.routing.types import NoEligibleModelError
from tests.conftest import MOCK_REGISTRY

U = "user"


def msgs(text: str, system: str | None = None) -> list[dict[str, str]]:
    out = [{"role": "system", "content": system}] if system else []
    return [*out, {"role": U, "content": text}]


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_yaml(MOCK_REGISTRY)


# ------------------------------------------------------------------ tokens
def test_token_estimate_reasonable() -> None:
    assert estimate_tokens("") == 0
    assert 8 <= estimate_tokens("The quick brown fox jumps over the lazy dog.") <= 14
    long_ident = estimate_tokens("supercalifragilisticexpialidocious")
    assert long_ident > 3  # long words split into several pieces
    assert estimate_message_tokens(msgs("hi")) > estimate_tokens("hi")


# ---------------------------------------------------------------- features
def test_features_are_deterministic_and_complete() -> None:
    f1 = extract_features(msgs("Write a python function def add(a, b): return a + b"))
    f2 = extract_features(msgs("Write a python function def add(a, b): return a + b"))
    assert f1 == f2
    assert f1.vector().shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f1.vector()).all()


@pytest.mark.parametrize(
    ("text", "params", "expected"),
    [
        ("Write a Python function that reverses a linked list. def reverse(head):", {}, "code"),
        (
            "Natalia sold 48 clips in April and half as many in May. How many clips did she sell in total?",
            {},
            "reasoning",
        ),
        ("Summarize the following article in two sentences: " + "word " * 300, {}, "summarization"),
        (
            'Extract the fields into JSON matching this schema {"name": ...}',
            {"response_format": {"type": "json_object"}},
            "json",
        ),
        (
            "Classify the sentiment of this review into one of: positive, negative.",
            {},
            "classification",
        ),
        ("Who wrote the novel Beloved?", {}, "factual_qa"),
    ],
)
def test_rule_based_task_detection(text: str, params: dict[str, object], expected: str) -> None:
    probs = rule_based_task_probs(extract_features(msgs(text), params))
    assert abs(sum(probs.values()) - 1) < 1e-9
    assert max(probs, key=probs.__getitem__) == expected


def test_feature_extraction_is_fast() -> None:
    import time

    m = msgs("Explain step by step how many apples remain if " * 40)
    t = time.perf_counter()
    for _ in range(200):
        extract_features(m)
    per_call_ms = (time.perf_counter() - t) / 200 * 1000
    assert per_call_ms < 5


# -------------------------------------------------------------- cost/latency
def test_cost_model_uses_table_and_caps(registry: ModelRegistry) -> None:
    spec = registry.require("mock-premium")
    cm = CostModel({"mock-premium": {"code": 400.0}})
    probs = dict.fromkeys(TASK_CATEGORIES, 0.0) | {"code": 1.0}
    cost, out = cm.expected_cost(spec, 100, probs, max_tokens=None)
    assert out == 400.0
    assert cost == pytest.approx((100 * 2.0 + 400 * 10.0) / 1e6)
    _, capped = cm.expected_cost(spec, 100, probs, max_tokens=50)
    assert capped == 50


def test_latency_model_online_correction(registry: ModelRegistry) -> None:
    spec = registry.require("mock-small")
    lm = LatencyModel({"mock-small": LatencyPrior(100.0, 10.0)})
    assert lm.predict_ms(spec, 0, 10) == pytest.approx(200.0)
    for _ in range(50):
        lm.observe(spec, 0, 10, 400.0)  # provider is 2x slower than the prior
    assert 350 < lm.predict_ms(spec, 0, 10) < 410


# ---------------------------------------------------------------- strategies
def test_fixed_and_random_strategies(registry: ModelRegistry) -> None:
    eng = RoutingEngine(registry, seed=1)
    m = msgs("hello")
    assert eng.decide("premium", m, {}, quality_threshold=0.8).selected_model == "mock-premium"
    assert eng.decide("economy", m, {}, quality_threshold=0.8).selected_model == "mock-economy"
    # Random is reproducible per request, and covers several models across requests.
    a = eng.decide("random", m, {}, quality_threshold=0.8).selected_model
    assert all(
        eng.decide("random", m, {}, quality_threshold=0.8).selected_model == a for _ in range(5)
    )
    picks = {
        eng.decide("random", msgs(f"q{i}"), {}, quality_threshold=0.8).selected_model
        for i in range(40)
    }
    assert len(picks) >= 3


def test_heuristic_levels(registry: ModelRegistry) -> None:
    eng = RoutingEngine(registry)
    d = eng.decide(
        "heuristic",
        msgs("Classify the topic of this headline: one of World, Sports."),
        {},
        quality_threshold=0.8,
    )
    assert d.selected_model == "mock-economy"
    d = eng.decide(
        "heuristic",
        msgs("Write a python function def f(x): that returns x squared"),
        {},
        quality_threshold=0.8,
    )
    assert d.selected_model == "mock-premium"
    d = eng.decide(
        "heuristic", msgs("Who discovered penicillin?"), {"max_tokens": 2000}, quality_threshold=0.8
    )
    assert "large_max_tokens" in d.reason


def test_unavailable_models_are_skipped(registry: ModelRegistry) -> None:
    eng = RoutingEngine(registry, is_available=lambda mid: mid != "mock-premium")
    d = eng.decide("premium", msgs("hello"), {}, quality_threshold=0.8)
    assert d.selected_model == "mock-medium"
    assert d.reason.startswith("fixed_target_unavailable")
    assert "mock-premium" not in d.fallback_chain
    none = RoutingEngine(registry, is_available=lambda _m: False)
    with pytest.raises(NoEligibleModelError):
        none.decide("heuristic", msgs("x"), {}, quality_threshold=0.8)


def test_context_window_eligibility(registry: ModelRegistry) -> None:
    eng = RoutingEngine(registry)
    huge = msgs("word " * 140_000)
    ctx = eng.build_context(huge, {}, quality_threshold=0.8, latency_budget_ms=None)
    assert ctx.ineligible.get("mock-economy") == "context_window"
    assert "mock-premium" not in ctx.ineligible  # 200k window


def test_resolve_model_field(registry: ModelRegistry) -> None:
    eng = RoutingEngine(registry, default_strategy="economy")
    assert eng.resolve("routeforge-auto", None) == (None, "economy")
    assert eng.resolve("auto", "premium") == (None, "premium")
    assert eng.resolve("routeforge-random", None) == (None, "random")
    assert eng.resolve("mock-small", None) == ("mock-small", "explicit")
    with pytest.raises(UnknownStrategyError):
        eng.resolve("routeforge-nope", None)


# ----------------------------------------------------------- learned router
def make_artifact(tmp_path: Path, registry: ModelRegistry) -> Path:
    """Tiny synthetic artifact: cheaper models are only good at short prompts."""
    rng = np.random.default_rng(0)
    texts, cats = [], []
    for i in range(140):
        cat = TASK_CATEGORIES[i % len(TASK_CATEGORIES)]
        words = {
            "code": "def function python return",
            "reasoning": "how many total each apples",
        }.get(cat, cat)
        texts.append(f"{words} sample {i} " + "pad " * int(rng.integers(1, 200)))
        cats.append(cat)
    task_clf = Pipeline(
        [
            (
                "v",
                __import__(
                    "sklearn.feature_extraction.text", fromlist=["TfidfVectorizer"]
                ).TfidfVectorizer(),
            ),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    ).fit(texts, cats)
    feats = np.vstack([extract_features(msgs(t)).vector() for t in texts])
    tp = task_clf.predict_proba(texts)
    full = np.hstack([feats, tp])
    lengths = feats[:, FEATURE_NAMES.index("input_tokens_est")]
    quality: dict[str, dict[str, object]] = {"full": {}, "no_task": {}}
    for spec in registry.candidates():
        y = (lengths < 60 + 60 * spec.tier).astype(int)
        y[:2], y[2:4] = 1, 0  # ensure both classes present
        for variant, X in (("full", full), ("no_task", feats)):
            quality[variant][spec.id] = Pipeline(
                [("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]
            ).fit(X, y)
    bundle = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "router_version": "test-router",
        "feature_names": list(FEATURE_NAMES),
        "models": [m.id for m in registry.candidates()],
        "task_classifier": task_clf,
        "quality_models": quality,
        "margins": {"full": {}, "no_task": {}},
        "output_tokens": {},
        "latency_priors": {},
        "max_train_input_tokens": 10_000.0,
        "ood_task_confidence": 0.0,
    }
    out = tmp_path / "artifact"
    save_bundle(bundle, out, {"note": "unit test"})
    return out


def test_learned_router_prefers_cheapest_model_meeting_threshold(
    tmp_path: Path, registry: ModelRegistry
) -> None:
    artifact = RouterArtifact.load(make_artifact(tmp_path, registry))
    eng = RoutingEngine(registry, artifact=artifact)
    assert {"learned", "learned-no-task", "learned-no-cost", "learned-no-latency"} <= set(
        eng.strategies
    )
    short = eng.decide("learned", msgs("sort a list"), {}, quality_threshold=0.6)
    assert short.selected_model == "mock-economy"
    assert short.reason == "cheapest_meeting_constraints"
    long = eng.decide("learned", msgs("sort a list " + "pad " * 150), {}, quality_threshold=0.6)
    assert registry.require(long.selected_model).tier > 0
    # selected model always satisfies the threshold when the reason says so
    sel = next(c for c in long.candidates if c.model_id == long.selected_model)
    if long.reason == "cheapest_meeting_constraints":
        assert sel.p_quality_adjusted is not None and sel.p_quality_adjusted >= 0.6
    impossible = eng.decide("learned", msgs("x " * 400), {}, quality_threshold=1.0)
    assert impossible.reason.startswith("no_model_met")
    assert impossible.confidence == max(c.p_quality or 0 for c in impossible.candidates)


def test_learned_router_latency_budget(tmp_path: Path, registry: ModelRegistry) -> None:
    eng = RoutingEngine(registry, artifact=RouterArtifact.load(make_artifact(tmp_path, registry)))
    d = eng.decide("learned", msgs("sort a list"), {}, quality_threshold=0.0, latency_budget_ms=1.0)
    assert d.reason == "no_model_met_quality:best_within_latency" or d.reason.startswith(
        "no_model_met"
    )
    d2 = eng.decide(
        "learned-no-latency", msgs("sort a list"), {}, quality_threshold=0.0, latency_budget_ms=1.0
    )
    assert d2.reason == "cheapest_meeting_constraints"


def test_artifact_checksum_is_enforced(tmp_path: Path, registry: ModelRegistry) -> None:
    path = make_artifact(tmp_path, registry)
    (path / "router.joblib.sha256").write_text("0" * 64)
    with pytest.raises(ArtifactError):
        RouterArtifact.load(path)
    assert RoutingEngine.load_artifact(path) is None  # engine degrades to non-learned strategies
