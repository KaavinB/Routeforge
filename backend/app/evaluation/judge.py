"""LLM-as-judge for summarization quality.

Reference-guided rubric (the judge sees the source article, the dataset's reference
highlights, and the candidate). The judge model is deliberately *not* one of the routing
candidates to avoid self-preference bias. Judge outputs are cached on disk keyed by a
hash of (judge model, rubric version, inputs): judging is deterministic at temperature 0
and caching it does not affect any inference-latency measurement.

The judge is not ground truth: agreement with a second judge and with manual labels is
measured by ``router_training.judge_agreement`` and reported in docs/EVALUATION.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import litellm

from app.evaluation.budget import SpendLedger
from app.evaluation.scorers import extract_json

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "summary-rubric-v1"

# USD per 1M tokens (input, output). Sources as in config/models/hosted.yaml (2026-10-11).
JUDGE_PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1": (2.00, 8.00),
    "anthropic/claude-fable-5-1": (10.00, 50.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00),
}

JUDGE_SYSTEM = (
    "You are a strict, consistent evaluator of news summaries. You grade a CANDIDATE summary "
    "of an ARTICLE. REFERENCE HIGHLIGHTS written by the article's editors show the main points. "
    "Score each criterion on an integer 1-5 scale:\n"
    "- faithfulness: 5 = every claim is supported by the article; 3 = one minor unsupported or "
    "distorted detail; 1 = major fabrication or contradiction.\n"
    "- coverage: 5 = captures all main points in the reference highlights; 3 = captures about "
    "half; 1 = misses the main point.\n"
    "- conciseness: 5 = no padding or repetition; 1 = mostly filler.\n"
    "Do not reward length. Respond with JSON only: "
    '{"faithfulness": int, "coverage": int, "conciseness": int, "rationale": "<one sentence>"}'
)


def judge_prompt(article: str, highlights: str, candidate: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"ARTICLE:\n{article}\n\nREFERENCE HIGHLIGHTS:\n{highlights}"
                f"\n\nCANDIDATE SUMMARY:\n{candidate}"
            ),
        },
    ]


def acceptable_from_scores(scores: dict[str, Any]) -> bool:
    return int(scores.get("faithfulness", 0)) >= 4 and int(scores.get("coverage", 0)) >= 3


class JudgeCacheMissError(LookupError):
    pass


class SummaryJudge:
    def __init__(
        self,
        model: str = "openai/gpt-4.1",
        *,
        cache_path: Path | None = None,
        ledger: SpendLedger | None = None,
        concurrency: int = 4,
        cache_only: bool = False,
        request_overrides: dict[str, Any] | None = None,
    ) -> None:
        """``cache_only``: never call the judge model; raise ``JudgeCacheMiss`` instead
        (used by CI, which replays recorded outputs whose verdicts are already cached)."""
        self.model = model
        self.cache_only = cache_only
        # Defaults suit OpenAI judges (JSON mode, temperature 0). Reasoning judges such as
        # Claude Fable 5.1 reject temperature/JSON mode and need room for thinking tokens.
        self.request_params: dict[str, Any] = {
            "temperature": 0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        if request_overrides is not None:
            self.request_params = request_overrides
        self.cache_path = cache_path
        self.ledger = ledger
        self._sem = None if concurrency <= 0 else concurrency
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["result"]

    def _key(self, messages: list[dict[str, str]]) -> str:
        raw = json.dumps(
            {"model": self.model, "rubric": RUBRIC_VERSION, "messages": messages}, sort_keys=True
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    async def judge(self, article: str, highlights: str, candidate: str) -> dict[str, Any]:
        messages = judge_prompt(article, highlights, candidate)
        key = self._key(messages)
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        if self.cache_only:
            raise JudgeCacheMissError(key)
        in_price, out_price = JUDGE_PRICING.get(self.model, (5.0, 25.0))
        if self.ledger is not None:
            est = (
                sum(len(m["content"]) for m in messages) / 3.5 * in_price + 120 * out_price
            ) / 1e6
            self.ledger.check(est)
        resp = await litellm.acompletion(
            model=self.model, messages=messages, num_retries=2, timeout=180, **self.request_params
        )
        usage = resp.usage
        cost = (usage.prompt_tokens * in_price + usage.completion_tokens * out_price) / 1e6
        if self.ledger is not None:
            self.ledger.record(
                activity="judge",
                model=self.model,
                cost_usd=cost,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )
        try:
            scores = extract_json(resp.choices[0].message.content or "")
            result = {
                "faithfulness": int(scores["faithfulness"]),
                "coverage": int(scores["coverage"]),
                "conciseness": int(scores["conciseness"]),
                "rationale": str(scores.get("rationale", ""))[:300],
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            result = {
                "faithfulness": 0,
                "coverage": 0,
                "conciseness": 0,
                "rationale": "unparseable judge output",
            }
        result["judge_cost_usd"] = cost
        with self._lock:
            self._cache[key] = result
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as fh:
                    fh.write(json.dumps({"key": key, "result": result}) + "\n")
        return result
