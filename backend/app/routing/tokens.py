"""Fast, dependency-free token count estimation.

Routing needs an input-token estimate *before* calling any provider, in well under a
millisecond, for models with different tokenizers. We approximate BPE behaviour: each
word-ish run costs ceil(len/4) tokens (long identifiers split into several pieces) and
each punctuation/symbol is a token. Actual billing uses provider-reported usage; the
estimate's error against provider counts is measured in router_training (see
docs/EVALUATION.md) rather than assumed.
"""

from __future__ import annotations

import re

_ALPHA = re.compile(r"[A-Za-z]+")
_DIGITS = re.compile(r"\d+")
_SYMBOL = re.compile(r"[^\sA-Za-z\d]")
# Per-message overhead used by chat formats (role markers, separators).
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Letters cost ceil(len/5) per run, digits ceil(len/3) per run, each symbol 1."""
    if not text:
        return 0
    alpha = sum((len(p) + 4) // 5 for p in _ALPHA.findall(text))
    digits = sum((len(p) + 2) // 3 for p in _DIGITS.findall(text))
    return alpha + digits + len(_SYMBOL.findall(text))


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return (
        sum(estimate_tokens(m.get("content", "")) + MESSAGE_OVERHEAD_TOKENS for m in messages) + 2
    )
