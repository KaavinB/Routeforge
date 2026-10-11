from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.evaluation.budget import BudgetExceededError, SpendLedger
from app.evaluation.datasets import EvalCase
from app.evaluation.regression import GateConfig, evaluate_gates
from app.evaluation.sandbox import CodeJob, check_policy, run_jobs
from app.evaluation.scorers import (
    extract_final_number,
    rouge_l,
    score_alias,
    score_json_fields,
    score_label,
    score_numeric,
    score_span_f1,
    token_f1,
)
from app.evaluation.scoring import Scorer
from app.evaluation.stats import (
    bootstrap_ci,
    brier,
    cost_reduction_pct,
    ece,
    paired_bootstrap,
    quality_diff_pp,
    wilson_interval,
)


def case(scorer: str, reference: dict[str, Any], category: str = "x") -> EvalCase:
    return EvalCase(
        id="c1",
        category=category,
        source="test",
        license="MIT",
        provenance="hand_written",
        group_id="g",
        split="test",
        messages=[{"role": "user", "content": "q"}],
        scorer=scorer,
        reference=reference,
    )


# ------------------------------------------------------------------ scorers
def test_numeric_scorer() -> None:
    c = case("numeric", {"answer": "1250"})
    assert score_numeric(c, "So the total is 1,250 dollars.\nAnswer: $1,250").acceptable
    assert score_numeric(c, "first 3 then 1250").acceptable  # last number fallback
    assert not score_numeric(c, "Answer: 1200").acceptable
    assert extract_final_number("no numbers") is None


def test_alias_scorer_requires_concise_answer() -> None:
    c = case("alias_match", {"answers": ["George Washington", "Washington"]})
    assert score_alias(c, "George Washington.").acceptable
    assert score_alias(c, "It was Washington").acceptable
    assert not score_alias(c, "Washingtonian").acceptable
    assert not score_alias(c, "Washington " + "filler " * 40).acceptable


def test_span_f1() -> None:
    assert token_f1("the Eiffel Tower", "Eiffel Tower") == 1.0
    c = case("span_f1", {"answers": ["in 1889"]})
    r = score_span_f1(c, "1889")
    assert r.score == pytest.approx(2 / 3) and r.acceptable


def test_label_scorer_rejects_hedging() -> None:
    c = case("label", {"label": "Sports", "labels": ["World", "Sports", "Business", "Sci/Tech"]})
    assert score_label(c, "Sports").acceptable
    assert not score_label(c, "Sports or Business").acceptable
    assert not score_label(c, "World").acceptable
    c2 = case("label", {"label": "Sci/Tech", "labels": ["World", "Sports", "Business", "Sci/Tech"]})
    assert score_label(c2, "Sci/Tech").acceptable


def test_json_scorer() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    c = case("json_fields", {"schema": schema, "expected": {"a": 3, "b": "Hello"}})
    assert score_json_fields(c, '```json\n{"a": 3, "b": "hello"}\n```').acceptable
    partial = score_json_fields(c, '{"a": 4, "b": "hello"}')
    assert partial.score == 0.5 and not partial.acceptable
    invalid = score_json_fields(c, '{"a": "3", "b": "hello"}')
    assert invalid.details["schema_valid"] is False
    assert score_json_fields(c, "not json").details["valid_json"] is False


def test_rouge_l() -> None:
    r = rouge_l("the cat sat on the mat", "the cat was on the mat")
    assert 0.6 < r["f1"] < 1
    assert rouge_l("", "x")["f1"] == 0


# ------------------------------------------------------------------ sandbox
def test_sandbox_policy() -> None:
    assert check_policy("import math\ndef f(x):\n    return math.sqrt(x)") is None
    assert "import not allowed" in (check_policy("import os") or "")
    assert "import not allowed" in (check_policy("from subprocess import run") or "")
    assert "name not allowed" in (check_policy("open('/etc/passwd')") or "")
    assert "name not allowed" in (check_policy("eval('1')") or "")
    assert "attribute not allowed" in (check_policy("().__class__.__bases__") or "")
    assert "syntax error" in (check_policy("def (") or "")


async def test_subprocess_sandbox_runs_tests_with_timeout() -> None:
    jobs = [
        CodeJob("ok", "def add(a, b):\n    return a + b\nassert add(1, 2) == 3\n"),
        CodeJob("fail", "def add(a, b):\n    return a - b\nassert add(1, 2) == 3\n"),
        CodeJob("loop", "while True:\n    pass\n", timeout_s=1.0),
        CodeJob("env", "import sys\nassert 'OPENAI_API_KEY' not in str(sys.argv)\n"),
    ]
    results = {r.id: r for r in await run_jobs(jobs, backend="subprocess")}
    assert results["ok"].passed
    assert not results["fail"].passed
    assert not results["loop"].passed and "timeout" in results["loop"].detail
    assert results["env"].passed


async def test_scorer_batches_code_and_summaries_without_judge() -> None:
    code_case = case("code_tests", {"test_program": "assert sq(3) == 9\n"}, "code")
    summ = case(
        "summary_judge",
        {"highlights": "The council approved the new park.", "article": "..."},
        "summarization",
    )
    results = await Scorer(sandbox_backend="subprocess").score_many(
        [
            (code_case, "```python\ndef sq(x):\n    return x * x\n```"),
            (summ, "The council approved a new park."),
            (code_case, None),
            (code_case, "import os\n"),
        ]
    )
    assert results[0].acceptable
    assert results[1].details["method"] == "rouge_fallback"
    assert not results[2].acceptable
    assert "policy_violation" in results[3].details


# -------------------------------------------------------------------- stats
def test_wilson_and_bootstrap() -> None:
    lo, hi = wilson_interval(80, 100)
    assert 0.7 < lo < 0.8 < hi < 0.88
    point, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], resamples=2000)
    assert lo <= point <= hi


def test_paired_bootstrap_detects_real_difference_but_not_noise() -> None:
    rng = np.random.default_rng(1)
    n = 300
    base = {
        "acceptable": rng.random(n) < 0.8,
        "cost": np.full(n, 1.0),
        "latency_ms": np.full(n, 100.0),
    }
    cheaper = {
        "acceptable": base["acceptable"].copy(),
        "cost": np.full(n, 0.4),
        "latency_ms": base["latency_ms"],
    }
    r = paired_bootstrap(base, cheaper, cost_reduction_pct, resamples=1000)
    assert r["estimate"] == pytest.approx(60.0) and r["ci_low"] > 50
    flipped = cheaper["acceptable"].copy()
    flipped[:3] = ~flipped[:3]  # 1pp of noise
    q = paired_bootstrap(base, {**cheaper, "acceptable": flipped}, quality_diff_pp, resamples=1000)
    assert q["ci_low"] < 0 < q["ci_high"] or abs(q["estimate"]) <= 1.0


def test_calibration_metrics() -> None:
    y = np.array([1, 0, 1, 0])
    assert brier([1, 0, 1, 0], y) == 0
    assert ece([0.5] * 4, y) == pytest.approx(0.0)
    assert ece([0.9, 0.9, 0.9, 0.9], y) == pytest.approx(0.4)


# --------------------------------------------------------------- regression
def _rows(
    acc: list[bool], cost: float = 1.0, lat: float = 100.0, category: str = "qa"
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"c{i}",
            "acceptable": a,
            "cost_usd": cost,
            "latency_ms": lat,
            "category": category,
        }
        for i, a in enumerate(acc)
    ]


def test_regression_gate_blocks_real_quality_drop() -> None:
    ref = _rows([True] * 180 + [False] * 20)
    worse = _rows([True] * 140 + [False] * 60)
    report = evaluate_gates(ref, worse, GateConfig(resamples=1000))
    assert not report["passed"]
    assert report["gates"][0]["status"] == "fail"


def test_regression_gate_tolerates_noise_and_flags_cost() -> None:
    ref = _rows([True] * 180 + [False] * 20)
    noisy = _rows([True] * 178 + [False] * 22, cost=1.5)
    report = evaluate_gates(ref, noisy, GateConfig(resamples=1000))
    statuses = {g["name"]: g["status"] for g in report["gates"]}
    assert report["passed"]
    assert statuses["quality_drop"] == "pass"
    assert statuses["cost_per_acceptable_increase"] == "warn"


def test_regression_gate_structured_output() -> None:
    ref = _rows([True] * 50, category="json")
    cand = [dict(r, score_details={"schema_valid": i % 4 != 0}) for i, r in enumerate(ref)]
    report = evaluate_gates(ref, cand, GateConfig(resamples=500))
    assert {g["name"]: g["status"] for g in report["gates"]}["structured_output_validity"] == "fail"


# ------------------------------------------------------------------ budget
def test_spend_ledger_enforces_cap(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=0.01)
    ledger.check(0.005)
    ledger.record(activity="t", model="m", cost_usd=0.008, input_tokens=1, output_tokens=1)
    with pytest.raises(BudgetExceededError):
        ledger.check(0.005)
    reloaded = SpendLedger(tmp_path / "ledger.jsonl", cap_usd=0.01)
    assert reloaded.spent == pytest.approx(0.008)
    assert json.loads((tmp_path / "ledger.jsonl").read_text())["activity"] == "t"
