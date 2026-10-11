"""Task-specific scorers. Each returns a score in [0, 1] and a binary ``acceptable``
label; the acceptable label is what router training and quality metrics use.

| scorer        | task            | method                                                  |
|---------------|-----------------|---------------------------------------------------------|
| numeric       | reasoning       | final number == reference (tolerance 1e-6)              |
| alias_match   | factual QA      | normalised alias contained in a short answer            |
| span_f1       | extraction      | max token-F1 vs. reference spans; acceptable F1 >= 0.6  |
| label         | classification  | exactly one label mentioned and it is the gold label    |
| json_fields   | structured JSON | schema-valid AND every expected field correct           |
| code_tests    | code            | all unit tests pass in the sandbox                      |
| summary_judge | summarization   | LLM judge (rubric, reference-guided); ROUGE-L recorded  |
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from app.evaluation.datasets import EvalCase


@dataclass
class ScoreResult:
    score: float
    acceptable: bool
    details: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------- utilities
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _to_number(raw: str) -> float | None:
    try:
        return float(raw.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def extract_final_number(text: str) -> float | None:
    m = re.search(r"(?i)answer\s*[:=]\s*\**\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return _to_number(m.group(1))
    nums = _NUMBER.findall(text)
    return _to_number(nums[-1]) if nums else None


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:json|python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def extract_json(text: str) -> Any:
    candidate = strip_code_fences(text).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


def extract_code(text: str) -> str:
    blocks: list[str] = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text


def rouge_l(pred: str, ref: str) -> dict[str, float]:
    p, r = normalize_text(pred).split(), normalize_text(ref).split()
    if not p or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # LCS length via DP over the shorter sequence.
    prev = [0] * (len(r) + 1)
    for tok in p:
        cur = [0] * (len(r) + 1)
        for j, rt in enumerate(r, 1):
            cur[j] = prev[j - 1] + 1 if tok == rt else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    precision, recall = lcs / len(p), lcs / len(r)
    f1 = 0.0 if lcs == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ------------------------------------------------------------- sync scorers
def score_numeric(case: EvalCase, output: str) -> ScoreResult:
    expected = float(case.reference["answer"])
    got = extract_final_number(output)
    ok = got is not None and abs(got - expected) < 1e-6
    return ScoreResult(float(ok), ok, {"extracted": got, "expected": expected})


_ALIAS_STOPWORDS = frozenset(
    {"in", "on", "at", "of", "by", "to", "for", "from", "and", "or", "is", "was"}
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if t not in _ALIAS_STOPWORDS}


def score_alias(case: EvalCase, output: str) -> ScoreResult:
    """Scorer v2 (lenient match): an alias matches if it appears as a phrase in the
    normalised answer, or if all of its content tokens (stopwords removed) appear in it.
    v1 used phrase containment only, which rejected e.g. "Santa Monica, California" for
    the gold alias "in Santa Monica". Answers must stay concise (<= 30 words) so that
    listing many candidates is not rewarded."""
    norm = normalize_text(output)
    pred_tokens = _content_tokens(output)
    hit = False
    for alias in case.reference["answers"]:
        a = normalize_text(alias)
        gold_tokens = _content_tokens(alias)
        if (a and (a == norm or f" {a} " in f" {norm} ")) or (
            gold_tokens and gold_tokens <= pred_tokens
        ):
            hit = True
    concise = len(norm.split()) <= 30
    ok = hit and concise
    return ScoreResult(float(ok), ok, {"alias_hit": hit, "concise": concise, "scorer_version": 2})


def score_span_f1(case: EvalCase, output: str) -> ScoreResult:
    answers = case.reference["answers"]
    best = max(token_f1(output, a) for a in answers)
    em = any(normalize_text(output) == normalize_text(a) for a in answers)
    ok = em or best >= 0.6
    return ScoreResult(best, ok, {"f1": best, "exact_match": em})


def score_label(case: EvalCase, output: str) -> ScoreResult:
    labels = case.reference["labels"]
    gold = case.reference["label"]
    norm = f" {normalize_text(output)} "
    mentioned = [lab for lab in labels if f" {normalize_text(lab)} " in norm]
    ok = mentioned == [gold]
    return ScoreResult(float(ok), ok, {"mentioned": mentioned, "gold": gold})


def _field_equal(expected: Any, got: Any) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    if isinstance(expected, int | float) and isinstance(got, int | float):
        return abs(float(expected) - float(got)) < 1e-6
    if isinstance(expected, str) and isinstance(got, str):
        return normalize_text(expected) == normalize_text(got)
    if isinstance(expected, list) and isinstance(got, list):
        return len(expected) == len(got) and all(
            _field_equal(e, g) for e, g in zip(expected, got, strict=True)
        )
    if isinstance(expected, dict) and isinstance(got, dict):
        return expected.keys() == got.keys() and all(
            _field_equal(expected[k], got[k]) for k in expected
        )
    return bool(expected == got)


def score_json_fields(case: EvalCase, output: str) -> ScoreResult:
    try:
        obj = extract_json(output)
    except (json.JSONDecodeError, ValueError):
        return ScoreResult(0.0, False, {"valid_json": False, "schema_valid": False})
    schema = case.reference["schema"]
    try:
        jsonschema.validate(obj, schema)
        schema_valid = True
    except jsonschema.ValidationError as exc:
        return ScoreResult(
            0.0, False, {"valid_json": True, "schema_valid": False, "error": exc.message[:200]}
        )
    expected: dict[str, Any] = case.reference["expected"]
    correct = {k: _field_equal(v, obj.get(k)) for k, v in expected.items()}
    acc = sum(correct.values()) / len(correct)
    return ScoreResult(
        acc,
        all(correct.values()),
        {"valid_json": True, "schema_valid": schema_valid, "fields": correct},
    )


SYNC_SCORERS = {
    "numeric": score_numeric,
    "alias_match": score_alias,
    "span_f1": score_span_f1,
    "label": score_label,
    "json_fields": score_json_fields,
}
