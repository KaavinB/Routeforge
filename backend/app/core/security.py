"""API key generation and hashing.

Keys look like ``rf-<43 url-safe chars>`` (256 bits of entropy). Only an HMAC-SHA256
digest (keyed with a server-side pepper) is stored, so a database leak does not reveal
usable keys, and a stolen database without the pepper cannot be brute-forced offline.
A fast hash is appropriate here (unlike passwords) because keys are high-entropy.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

KEY_PREFIX = "rf-"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str, pepper: str) -> str:
    return hmac.new(pepper.encode(), key.encode(), hashlib.sha256).hexdigest()


def key_display_prefix(key: str) -> str:
    """Non-secret identifier shown in UIs and logs (first 10 chars)."""
    return key[:10]


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
