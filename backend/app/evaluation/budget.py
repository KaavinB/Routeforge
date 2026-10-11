"""Spend ledger with a hard cap for paid API usage in offline tooling.

Every paid call made by data collection, judging and benchmarks is appended to a JSONL
ledger (committed to the repo as an audit trail). Before each call the tool checks that
``spent + estimated_next_cost <= cap``; the cap comes from ROUTEFORGE_PAID_BUDGET_USD.
Costs are estimates from provider-reported tokens x list price.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class BudgetExceededError(RuntimeError):
    pass


class SpendLedger:
    def __init__(self, path: Path, cap_usd: float | None = None) -> None:
        self.path = path
        self.cap_usd = (
            cap_usd
            if cap_usd is not None
            else float(os.environ.get("ROUTEFORGE_PAID_BUDGET_USD", "0") or 0)
        )
        self._lock = threading.Lock()
        self._spent = 0.0
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self._spent += float(json.loads(line)["cost_usd"])

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self.cap_usd - self._spent

    def check(self, estimated_cost_usd: float) -> None:
        with self._lock:
            if self._spent + estimated_cost_usd > self.cap_usd:
                raise BudgetExceededError(
                    f"paid budget exhausted: spent ${self._spent:.4f} "
                    f"+ next ~${estimated_cost_usd:.4f} "
                    f"> cap ${self.cap_usd:.2f} (ROUTEFORGE_PAID_BUDGET_USD)"
                )

    def record(
        self, *, activity: str, model: str, cost_usd: float, input_tokens: int, output_tokens: int
    ) -> None:
        if cost_usd <= 0:
            return
        with self._lock:
            self._spent += cost_usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "activity": activity,
                            "model": model,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost_usd": round(cost_usd, 8),
                        }
                    )
                    + "\n"
                )
