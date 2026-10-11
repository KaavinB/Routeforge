"""Ensure .env contains the secrets docker compose needs. Generates random values for any
missing entry and never overwrites existing ones (so provider API keys you added stay put).

    python3 infrastructure/scripts/init_env.py
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / ".env"
GENERATED = {
    "ROUTEFORGE_API_KEY_PEPPER": lambda: secrets.token_urlsafe(32),
    "ROUTEFORGE_BOOTSTRAP_ADMIN_KEY": lambda: "rf-admin-" + secrets.token_urlsafe(24),
    "ROUTEFORGE_BOOTSTRAP_CLIENT_KEY": lambda: "rf-client-" + secrets.token_urlsafe(24),
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(18),
    "ROUTEFORGE_APP_DB_PASSWORD": lambda: secrets.token_urlsafe(18),
    "GRAFANA_ADMIN_PASSWORD": lambda: secrets.token_urlsafe(12),
}
DEFAULTS = {"ROUTEFORGE_MODE": "mock", "ROUTEFORGE_PAID_BUDGET_USD": "20"}


def main() -> int:
    existing: dict[str, str] = {}
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            existing[k.strip()] = v.strip()
    added = []
    for key, make in GENERATED.items():
        if not existing.get(key):
            lines.append(f"{key}={make()}")
            added.append(key)
    for key, value in DEFAULTS.items():
        if key not in existing:
            lines.append(f"{key}={value}")
            added.append(key)
    if added:
        ENV.write_text("\n".join(lines) + "\n")
        ENV.chmod(0o600)
    print(f".env: added {', '.join(added) if added else 'nothing (all set)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
