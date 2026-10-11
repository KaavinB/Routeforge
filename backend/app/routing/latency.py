"""Latency model: offline prior + online correction.

predicted_ms = ttft_ms + in_tokens * ms_per_input_token + out_tokens * ms_per_output_token

Priors are fitted per model from router-training collection runs (least squares on
observed latencies). At serving time an exponentially weighted ratio of observed /
predicted latency adapts to current provider conditions (e.g. a provider slowing down
during peak hours) without retraining.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.registry import ModelSpec


@dataclass
class LatencyPrior:
    ttft_ms: float
    ms_per_output_token: float
    ms_per_input_token: float = 0.0


class LatencyModel:
    def __init__(self, priors: dict[str, LatencyPrior] | None = None, alpha: float = 0.1) -> None:
        self.priors = priors or {}
        self.alpha = alpha
        self._ratio: dict[str, float] = {}
        self._lock = threading.Lock()

    def _prior(self, spec: ModelSpec) -> LatencyPrior:
        if spec.id in self.priors:
            return self.priors[spec.id]
        if spec.mock is not None:
            return LatencyPrior(spec.mock.ttft_ms, 1000.0 / spec.mock.tokens_per_s)
        # Generic prior for an unprofiled hosted model; deliberately pessimistic.
        return LatencyPrior(ttft_ms=600.0, ms_per_output_token=25.0)

    def predict_ms(self, spec: ModelSpec, input_tokens: float, output_tokens: float) -> float:
        p = self._prior(spec)
        base = (
            p.ttft_ms + input_tokens * p.ms_per_input_token + output_tokens * p.ms_per_output_token
        )
        return base * self._ratio.get(spec.id, 1.0)

    def observe(
        self, spec: ModelSpec, input_tokens: float, output_tokens: float, actual_ms: float
    ) -> None:
        p = self._prior(spec)
        predicted = (
            p.ttft_ms + input_tokens * p.ms_per_input_token + output_tokens * p.ms_per_output_token
        )
        if predicted <= 0 or actual_ms <= 0:
            return
        # Clamp single observations so one outlier cannot swing routing.
        ratio = min(max(actual_ms / predicted, 0.2), 5.0)
        with self._lock:
            prev = self._ratio.get(spec.id, 1.0)
            self._ratio[spec.id] = (1 - self.alpha) * prev + self.alpha * ratio

    def correction(self, model_id: str) -> float:
        return self._ratio.get(model_id, 1.0)
