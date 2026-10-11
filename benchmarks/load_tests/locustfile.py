"""Locust workload for Benchmark C. Each simulated user sends held-out benchmark prompts
back-to-back (closed loop, no think time) through the gateway.

Environment:
  RF_LOAD_KEY       API key to use (the runner creates a dedicated high-limit key)
  RF_LOAD_MODEL     model field (default routeforge-auto)
  RF_LOAD_BYPASS    "1" to bypass the response cache (default) so every request is routed
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import FastHttpUser, constant, task

CASES = (
    Path(__file__).resolve().parents[2]
    / "router_training"
    / "datasets"
    / "data"
    / "routeforge-bench-v1"
    / "cases.jsonl"
)
PROMPTS = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
PROMPTS = [p for p in PROMPTS if p["split"] == "test"]


class GatewayUser(FastHttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        self.headers = {
            "authorization": f"Bearer {os.environ['RF_LOAD_KEY']}",
            "content-type": "application/json",
        }
        if os.environ.get("RF_LOAD_BYPASS", "1") == "1":
            self.headers["x-routeforge-cache"] = "bypass"
        self.model = os.environ.get("RF_LOAD_MODEL", "routeforge-auto")
        self.rng = random.Random()

    @task
    def chat(self) -> None:
        case = self.rng.choice(PROMPTS)
        body = {
            "model": self.model,
            "messages": case["messages"],
            "max_tokens": min(case["max_tokens"], 128),
            "temperature": 0,
        }
        with self.client.post(
            "/v1/chat/completions",
            data=json.dumps(body),
            headers=self.headers,
            name="chat",
            catch_response=True,
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
