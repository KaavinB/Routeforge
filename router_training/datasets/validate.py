"""Oracle validation of the dataset and scorers.

For every case we construct the *reference* answer (gold number, gold span, gold label,
expected JSON, canonical code solution) and check that its scorer marks it acceptable.
This catches broken answer keys, broken test programs and scorer bugs before any model
is evaluated. Summarization is excluded (no deterministic oracle; judged by an LLM).

    uv run python -m router_training.datasets.validate
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from app.evaluation.datasets import Dataset, EvalCase
from app.evaluation.scoring import Scorer
from router_training.datasets.build import OUT_ROOT, RAW_CACHE


def _canonical_code() -> dict[str, str]:
    """Reference solutions from the cached raw HumanEval / MBPP rows."""
    out: dict[str, str] = {}
    for path in RAW_CACHE.glob("humaneval_*.json"):
        for r in json.loads(path.read_text())["rows"]:
            row = r["row"]
            out[f"humaneval-{row['task_id'].split('/')[-1]}"] = (
                row["prompt"] + row["canonical_solution"]
            )
    for path in RAW_CACHE.glob("mbpp_*.json"):
        for r in json.loads(path.read_text())["rows"]:
            row = r["row"]
            out[f"mbpp-{row['task_id']}"] = row["code"]
    return out


def oracle_output(case: EvalCase, code: dict[str, str]) -> str | None:
    ref = case.reference
    match case.scorer:
        case "numeric":
            return f"Some reasoning.\nAnswer: {ref['answer']}"
        case "alias_match" | "span_f1":
            return str(ref["answers"][0])
        case "label":
            return str(ref["label"])
        case "json_fields":
            return json.dumps(ref["expected"])
        case "code_tests":
            sol = code.get(case.id)
            return f"```python\n{sol}\n```" if sol else None
    return None


async def main(name: str = "routeforge-bench-v1") -> None:
    ds = Dataset(OUT_ROOT / name)
    code = _canonical_code()
    items = [(c, oracle_output(c, code)) for c in ds.cases if c.scorer != "summary_judge"]
    results = await Scorer(sandbox_backend="docker").score_many(items)
    failures = [
        {"id": c.id, "scorer": c.scorer, "details": r.details}
        for (c, _), r in zip(items, results, strict=True)
        if not r.acceptable
    ]
    by_scorer = Counter(c.scorer for c, _ in items)
    fail_by = Counter(f["scorer"] for f in failures)
    report = {
        "dataset": ds.name,
        "version": ds.version,
        "checked": len(items),
        "by_scorer": {
            s: {"checked": n, "oracle_failures": fail_by.get(s, 0)} for s, n in by_scorer.items()
        },
        "failures": failures,
    }
    out = Path(ds.directory) / "oracle_validation.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    for f in failures[:15]:
        print("FAIL", f["id"], str(f["details"])[:300])


if __name__ == "__main__":
    asyncio.run(main())
