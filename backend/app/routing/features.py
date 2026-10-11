"""Request feature extraction for routing.

The same function is used at serving time and by ``router_training`` to build training
data, so there is no train/serve skew in feature computation. Features are cheap lexical
and structural signals (target: well under 1 ms per request); no network calls, no model
inference.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.routing.tokens import estimate_message_tokens

TASK_CATEGORIES: tuple[str, ...] = (
    "factual_qa",
    "extraction",
    "classification",
    "summarization",
    "code",
    "reasoning",
    "json",
)

# Keyword detectors. Single words are matched by set lookup over the tokenised prompt,
# multi-word phrases with str.count, and structural patterns with literal-prefixed
# regexes. (A single big alternation regex was ~10x slower on long prompts because
# Python's backtracking engine tries every alternative at every position.)
_KEYWORDS: dict[str, frozenset[str]] = {
    "code": frozenset(
        [
            "def",
            "return",
            "import",
            "function",
            "int",
            "str",
            "python",
            "javascript",
            "code",
            "assert",
            "lambda",
            "print",
            "array",
            "string",
            "boolean",
            "recursion",
            "algorithm",
        ]
    ),
    "reasoning": frozenset(
        [
            "calculate",
            "why",
            "prove",
            "total",
            "each",
            "per",
            "percent",
            "remaining",
            "times",
            "if",
            "then",
            "dollars",
            "dollar",
            "cost",
            "costs",
            "left",
            "altogether",
        ]
    ),
    "structured": frozenset(
        ["json", "schema", "key", "keys", "field", "fields", "yaml", "csv", "format"]
    ),
    "extraction": frozenset(["extract", "span", "quote", "identify", "verbatim"]),
    "summary": frozenset(
        ["summary", "summarize", "summarise", "summarization", "tldr", "highlights", "bullet"]
    ),
    "classify": frozenset(
        [
            "classify",
            "classification",
            "category",
            "categorize",
            "categories",
            "label",
            "sentiment",
            "topic",
        ]
    ),
}
_PHRASES: dict[str, tuple[str, ...]] = {
    "code": ("```", "=>", "write a function", "write a python function"),
    "reasoning": ("how many", "how much", "step by step", "step-by-step"),
    "structured": ("{", "["),
    "extraction": (
        "from the passage",
        "from the text",
        "from the context",
        "from the article",
        "context:",
        "passage:",
    ),
    "summary": ("key points", "tl;dr"),
    "classify": ("one of", "which of the following"),
}
_STRUCTURAL: dict[str, re.Pattern[str]] = {
    "code": re.compile(r"def \w+\(|class \w+|import \w+|list\["),
    "reasoning": re.compile(r"\$\d"),
}
_MATH_PATTERNS = re.compile("\\d+\\s*[-+*/x\\u00d7\\u00f7]\\s*\\d+|\\d+%|\\d+/\\d+|\\d+\\.\\d+")
_QUESTION_WORDS = frozenset(["who", "what", "when", "where", "which", "whom", "whose"])
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_WORD = re.compile(r"[a-z']+")
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]")


def _hits(kind: str, lower: str, word_counts: dict[str, int]) -> int:
    n = sum(word_counts.get(w, 0) for w in _KEYWORDS[kind])
    n += sum(lower.count(p) for p in _PHRASES.get(kind, ()))
    pat = _STRUCTURAL.get(kind)
    if pat is not None:
        n += len(pat.findall(lower))
    return n


LONG_CONTEXT_TOKENS = 2000


@dataclass(frozen=True)
class RequestFeatures:
    input_tokens_est: float
    log_input_tokens: float
    n_messages: float
    has_system: float
    prompt_chars: float
    n_words: float
    avg_word_len: float
    type_token_ratio: float
    digit_ratio: float
    punct_ratio: float
    n_lines: float
    n_sentences: float
    n_questions: float
    question_word_count: float
    code_score: float
    reasoning_score: float
    math_score: float
    structured_score: float
    extraction_score: float
    summary_score: float
    classify_score: float
    json_requested: float
    long_context: float
    max_tokens_requested: float
    log_max_tokens: float
    last_user_tokens: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=np.float64)


FEATURE_NAMES: tuple[str, ...] = tuple(RequestFeatures.__dataclass_fields__.keys())


def _density(hits: int, n_words: int) -> float:
    # log-scaled count, lightly normalised by length so long prompts don't dominate
    return math.log1p(hits) / math.log(max(n_words, 2) + 8, 10)


def routing_text(messages: list[dict[str, str]]) -> str:
    """Text the router looks at: system + user turns (assistant turns are context)."""
    return "\n".join(m["content"] for m in messages if m["role"] in ("system", "developer", "user"))


def extract_features(
    messages: list[dict[str, str]],
    params: dict[str, Any] | None = None,
) -> RequestFeatures:
    params = params or {}
    text = routing_text(messages)
    lower = text.lower()  # all keyword patterns are written in lower case
    words = _WORD.findall(lower)
    n_words = len(words)
    word_counts = Counter(words)
    digit_chars = sum(len(d) for d in _DIGITS.findall(text))
    n_chars = max(len(text), 1)
    input_tokens = estimate_message_tokens(messages)
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_user = user_msgs[-1] if user_msgs else ""
    rf = params.get("response_format") or {}
    json_requested = float(rf.get("type") in ("json_object", "json_schema"))
    max_tokens = float(params.get("max_tokens") or 0)

    return RequestFeatures(
        input_tokens_est=float(input_tokens),
        log_input_tokens=math.log1p(input_tokens),
        n_messages=float(len(messages)),
        has_system=float(any(m["role"] in ("system", "developer") for m in messages)),
        prompt_chars=float(len(text)),
        n_words=float(n_words),
        avg_word_len=(sum(len(w) for w in words) / n_words) if n_words else 0.0,
        type_token_ratio=(len(word_counts) / n_words) if n_words else 0.0,
        digit_ratio=digit_chars / n_chars,
        punct_ratio=len(_PUNCT.findall(text)) / n_chars,
        n_lines=float(text.count("\n") + 1),
        n_sentences=float(len(_SENTENCE_END.findall(text))),
        n_questions=float(text.count("?")),
        question_word_count=float(
            sum(1 for w in _WORD.findall(last_user.lower()) if w in _QUESTION_WORDS)
        ),
        code_score=_density(_hits("code", lower, word_counts), n_words),
        reasoning_score=_density(_hits("reasoning", lower, word_counts), n_words),
        math_score=_density(len(_MATH_PATTERNS.findall(text)) if digit_chars else 0, n_words),
        structured_score=_density(_hits("structured", lower, word_counts), n_words),
        extraction_score=_density(_hits("extraction", lower, word_counts), n_words),
        summary_score=_density(_hits("summary", lower, word_counts), n_words),
        classify_score=_density(_hits("classify", lower, word_counts), n_words),
        json_requested=json_requested,
        long_context=float(input_tokens >= LONG_CONTEXT_TOKENS),
        max_tokens_requested=max_tokens,
        log_max_tokens=math.log1p(max_tokens),
        last_user_tokens=float(
            input_tokens
            if len(messages) == 1
            else estimate_message_tokens([{"role": "user", "content": last_user}])
        ),
    )


def rule_based_task_probs(features: RequestFeatures) -> dict[str, float]:
    """Hand-written task scoring used by the heuristic router and as a fallback when no
    learned task classifier is available. Returns a normalised distribution."""
    f = features
    scores = {
        "code": 3.0 * f.code_score,
        "reasoning": 1.5 * f.reasoning_score + 2.0 * f.math_score,
        "json": 3.0 * f.json_requested + 1.5 * f.structured_score,
        "summarization": 3.0 * f.summary_score + (0.5 if f.input_tokens_est > 400 else 0.0),
        "extraction": 2.5 * f.extraction_score,
        "classification": 2.5 * f.classify_score,
        "factual_qa": 0.6 + 0.6 * f.question_word_count - 0.002 * f.input_tokens_est,
    }
    exps = {k: math.exp(max(min(v, 20.0), -20.0)) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: exps[k] / total for k in TASK_CATEGORIES}
