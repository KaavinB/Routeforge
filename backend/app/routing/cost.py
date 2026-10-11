"""Expected-cost model.

expected_cost(model, request) = in_tokens * in_price + E[out_tokens | model, task] * out_price

E[out_tokens] is the task-probability-weighted median output length observed for that
model during router training (models differ substantially in verbosity), capped by the
request's ``max_tokens``. Without a trained artifact, generic per-task priors are used.
"""

from __future__ import annotations

from app.core.registry import ModelSpec

DEFAULT_OUTPUT_TOKENS: dict[str, float] = {
    "factual_qa": 30.0,
    "extraction": 25.0,
    "classification": 4.0,
    "summarization": 110.0,
    "code": 180.0,
    "reasoning": 220.0,
    "json": 90.0,
}


class CostModel:
    def __init__(self, output_token_table: dict[str, dict[str, float]] | None = None) -> None:
        # model_id -> task -> median output tokens
        self.table = output_token_table or {}

    def expected_output_tokens(
        self, spec: ModelSpec, task_probs: dict[str, float], max_tokens: int | None
    ) -> float:
        per_task = self.table.get(spec.id, {})
        expected = sum(
            p * per_task.get(task, DEFAULT_OUTPUT_TOKENS.get(task, 100.0))
            for task, p in task_probs.items()
        )
        if max_tokens:
            expected = min(expected, float(max_tokens))
        return max(expected, 1.0)

    def expected_cost(
        self,
        spec: ModelSpec,
        input_tokens: float,
        task_probs: dict[str, float],
        max_tokens: int | None,
    ) -> tuple[float, float]:
        out = self.expected_output_tokens(spec, task_probs, max_tokens)
        cost = (input_tokens * spec.input_cost_per_mtok + out * spec.output_cost_per_mtok) / 1e6
        return cost, out
