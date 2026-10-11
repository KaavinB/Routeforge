"""Batch scoring: dispatches each (case, output) to its scorer, batching sandboxed code
execution and parallelising LLM-judge calls."""

from __future__ import annotations

import asyncio

from app.evaluation.datasets import EvalCase
from app.evaluation.judge import JudgeCacheMissError, SummaryJudge, acceptable_from_scores
from app.evaluation.sandbox import CodeJob, check_policy, run_jobs
from app.evaluation.scorers import SYNC_SCORERS, ScoreResult, extract_code, rouge_l


class Scorer:
    def __init__(
        self,
        *,
        judge: SummaryJudge | None = None,
        sandbox_backend: str = "docker",
        judge_concurrency: int = 4,
    ) -> None:
        self.judge = judge
        self.sandbox_backend = sandbox_backend
        self._judge_sem = asyncio.Semaphore(judge_concurrency)

    async def score_many(self, items: list[tuple[EvalCase, str | None]]) -> list[ScoreResult]:
        results: list[ScoreResult | None] = [None] * len(items)
        code_jobs: list[tuple[int, CodeJob]] = []
        judge_tasks: list[tuple[int, asyncio.Task[ScoreResult]]] = []

        for i, (case, output) in enumerate(items):
            if output is None:
                results[i] = ScoreResult(0.0, False, {"error": "no output"})
            elif case.scorer in SYNC_SCORERS:
                results[i] = SYNC_SCORERS[case.scorer](case, output)
            elif case.scorer == "code_tests":
                code = extract_code(output)
                violation = check_policy(code)
                if violation:
                    results[i] = ScoreResult(0.0, False, {"policy_violation": violation})
                else:
                    program = code + "\n\n" + case.reference["test_program"]
                    code_jobs.append((i, CodeJob(id=str(i), program=program, timeout_s=10.0)))
            elif case.scorer == "summary_judge":
                judge_tasks.append((i, asyncio.create_task(self._score_summary(case, output))))
            else:
                raise ValueError(f"unknown scorer {case.scorer}")

        if code_jobs:
            outcomes = await run_jobs([job for _, job in code_jobs], backend=self.sandbox_backend)
            by_id = {o.id: o for o in outcomes}
            for i, job in code_jobs:
                o = by_id.get(job.id)
                passed = bool(o and o.passed)
                results[i] = ScoreResult(
                    float(passed),
                    passed,
                    {"tests_passed": passed, "detail": o.detail[-200:] if o else "missing"},
                )
        for i, task in judge_tasks:
            results[i] = await task
        assert all(r is not None for r in results)
        return [r for r in results if r is not None]

    async def _score_summary(self, case: EvalCase, output: str) -> ScoreResult:
        ref = case.reference
        rouge = rouge_l(output, ref["highlights"])
        if self.judge is None:
            # Offline fallback only; documented as a weak proxy, never used for headline metrics.
            ok = rouge["recall"] >= 0.35
            return ScoreResult(rouge["f1"], ok, {"method": "rouge_fallback", "rouge_l": rouge})
        async with self._judge_sem:
            try:
                verdict = await self.judge.judge(ref["article"], ref["highlights"], output)
            except JudgeCacheMissError:
                ok = rouge["recall"] >= 0.35
                return ScoreResult(
                    rouge["f1"], ok, {"method": "rouge_fallback_cache_miss", "rouge_l": rouge}
                )
        ok = acceptable_from_scores(verdict)
        score = (verdict["faithfulness"] + verdict["coverage"]) / 10
        return ScoreResult(score, ok, {"method": "llm_judge", "judge": verdict, "rouge_l": rouge})
