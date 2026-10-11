"""Build the versioned RouteForge benchmark/router-training dataset.

    uv run python -m router_training.datasets.build --per-category 120

Sources (public samples fetched from the Hugging Face datasets-server API, plus one
programmatically generated category):

| category       | source                                   | license       | scorer        |
|----------------|------------------------------------------|---------------|---------------|
| factual_qa     | google-research-datasets/nq_open (val)   | CC BY-SA 3.0  | alias_match   |
| extraction     | rajpurkar/squad v1.1 (validation)        | CC BY-SA 4.0  | span_f1       |
| classification | fancyzhx/ag_news (test)                  | unspecified*  | label         |
| summarization  | abisee/cnn_dailymail 3.0.0 (test)        | Apache-2.0    | summary_judge |
| code           | mbpp sanitized (test) + openai_humaneval | CC BY 4.0/MIT | code_tests    |
| reasoning      | openai/gsm8k main (test)                 | MIT           | numeric       |
| json           | synthetic templates (this file)          | MIT (ours)    | json_fields   |

*AG News has no explicit license on its dataset card; it is widely used for research
evaluation and is used here only for non-commercial benchmarking.

Splits are made **by group within each category** (60/20/20): SQuAD groups by source
article, JSON groups by template, everything else by source item. A cross-split
near-duplicate check (character 5-gram Jaccard) runs afterwards; any test/val case too
similar to a train case is dropped and recorded in ``leakage_report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.datasets import EvalCase, file_sha256, write_cases
from app.evaluation.scorers import normalize_text
from router_training.datasets.json_templates import generate_json_cases

ROOT = Path(__file__).resolve().parent
RAW_CACHE = ROOT.parent / "data" / "raw_cache"
OUT_ROOT = ROOT / "data"
API = "https://datasets-server.huggingface.co/rows"

SOURCES: dict[str, dict[str, Any]] = {
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "license": "MIT"},
    "nq_open": {
        "dataset": "google-research-datasets/nq_open",
        "config": "nq_open",
        "split": "validation",
        "license": "CC BY-SA 3.0",
    },
    "squad": {
        "dataset": "rajpurkar/squad",
        "config": "plain_text",
        "split": "validation",
        "license": "CC BY-SA 4.0",
    },
    "ag_news": {
        "dataset": "fancyzhx/ag_news",
        "config": "default",
        "split": "test",
        "license": "unspecified (research use)",
    },
    "cnn_dm": {
        "dataset": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "split": "test",
        "license": "Apache-2.0",
    },
    "mbpp": {
        "dataset": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "test",
        "license": "CC BY 4.0",
    },
    "humaneval": {
        "dataset": "openai/openai_humaneval",
        "config": "openai_humaneval",
        "split": "test",
        "license": "MIT",
    },
}

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def _fetch(source: str, offset: int, length: int = 100) -> dict[str, Any]:
    s = SOURCES[source]
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    cache = RAW_CACHE / f"{source}_{offset}_{length}.json"
    if cache.exists():
        data: dict[str, Any] = json.loads(cache.read_text())
        return data
    q = urllib.parse.urlencode(
        {
            "dataset": s["dataset"],
            "config": s["config"],
            "split": s["split"],
            "offset": offset,
            "length": length,
        }
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as resp:  # noqa: S310 - fixed https host
                data = json.loads(resp.read())
            cache.write_text(json.dumps(data))
            time.sleep(1.0)  # be polite to the public API
            return data
        except Exception as exc:
            wait = min(60, 5 * 2**attempt)
            print(f"  fetch {source}@{offset} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {source} offset={offset}")


def sample_rows(source: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample: random page offsets (seeded), then rows within pages."""
    first = _fetch(source, 0, 100)
    total = int(first["num_rows_total"])
    rng = random.Random(f"{seed}:{source}")
    pages = sorted(
        {rng.randrange(0, max(total - 100, 1)) // 100 * 100 for _ in range(max(3, n // 40 + 2))}
    )
    rows: list[dict[str, Any]] = []
    for off in pages:
        for r in _fetch(source, off, 100)["rows"]:
            rows.append({"row_idx": r["row_idx"], **r["row"]})
    uniq = {r["row_idx"]: r for r in rows}
    rows = [uniq[k] for k in sorted(uniq)]
    rng.shuffle(rows)
    return rows


def _case(**kw: Any) -> EvalCase:
    return EvalCase(split="train", **kw)


def build_reasoning(n: int, seed: int) -> list[EvalCase]:
    out = []
    for r in sample_rows("gsm8k", n, seed)[:n]:
        answer = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            _case(
                id=f"gsm8k-{r['row_idx']}",
                category="reasoning",
                source="openai/gsm8k:main:test",
                license=SOURCES["gsm8k"]["license"],
                provenance="public_benchmark",
                group_id=f"gsm8k-{r['row_idx']}",
                messages=[
                    {
                        "role": "system",
                        "content": "Solve the math word problem. Reason briefly step by step, then give the final answer on the last line as 'Answer: <number>'.",
                    },
                    {"role": "user", "content": r["question"]},
                ],
                max_tokens=512,
                scorer="numeric",
                reference={"answer": answer},
            )
        )
    return out


def build_factual(n: int, seed: int) -> list[EvalCase]:
    out = []
    rows = [
        r
        for r in sample_rows("nq_open", n * 2, seed)
        if any(normalize_text(a) for a in r["answer"])  # drop broken gold labels, e.g. ")"
    ]
    for r in rows[:n]:
        q = r["question"].strip()
        out.append(
            _case(
                id=f"nq-{r['row_idx']}",
                category="factual_qa",
                source="google-research-datasets/nq_open:validation",
                license=SOURCES["nq_open"]["license"],
                provenance="public_benchmark",
                group_id=f"nq-{r['row_idx']}",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question with a short phrase only (a few words). No explanation.",
                    },
                    {"role": "user", "content": q[0].upper() + q[1:] + "?"},
                ],
                max_tokens=48,
                scorer="alias_match",
                reference={"answers": r["answer"]},
            )
        )
    return out


def build_extraction(n: int, seed: int) -> list[EvalCase]:
    out = []
    seen_ctx: Counter[str] = Counter()
    for r in sample_rows("squad", n * 3, seed):
        ctx_key = hashlib.sha1(r["context"].encode()).hexdigest()[:12]  # noqa: S324 - grouping id only
        if seen_ctx[ctx_key] >= 2:  # at most 2 questions per passage
            continue
        seen_ctx[ctx_key] += 1
        out.append(
            _case(
                id=f"squad-{r['id']}",
                category="extraction",
                source="rajpurkar/squad:plain_text:validation",
                license=SOURCES["squad"]["license"],
                provenance="public_benchmark",
                group_id=f"squad-{r['title']}",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the exact span from the passage that answers the question. Reply with the span only, copied verbatim.",
                    },
                    {
                        "role": "user",
                        "content": f"Passage: {r['context']}\n\nQuestion: {r['question']}",
                    },
                ],
                max_tokens=48,
                scorer="span_f1",
                reference={"answers": list(dict.fromkeys(r["answers"]["text"]))},
            )
        )
        if len(out) >= n:
            break
    return out


def build_classification(n: int, seed: int) -> list[EvalCase]:
    rows = sample_rows("ag_news", n * 2, seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)
    out = []
    per = n // len(AG_LABELS)
    for label, items in sorted(by_label.items()):
        for r in items[:per]:
            out.append(
                _case(
                    id=f"agnews-{r['row_idx']}",
                    category="classification",
                    source="fancyzhx/ag_news:test",
                    license=SOURCES["ag_news"]["license"],
                    provenance="public_benchmark",
                    group_id=f"agnews-{r['row_idx']}",
                    messages=[
                        {
                            "role": "system",
                            "content": "Classify the news article into exactly one topic: World, Sports, Business, or Sci/Tech. Reply with the topic label only.",
                        },
                        {"role": "user", "content": f"Article: {r['text']}"},
                    ],
                    max_tokens=8,
                    scorer="label",
                    reference={"label": AG_LABELS[label], "labels": AG_LABELS},
                )
            )
    return out


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def build_summarization(n: int, seed: int) -> list[EvalCase]:
    out = []
    for r in sample_rows("cnn_dm", n, seed)[:n]:
        article = _truncate_words(re.sub(r"\s+", " ", r["article"]), 450)
        out.append(
            _case(
                id=f"cnndm-{r['id'][:16]}",
                category="summarization",
                source="abisee/cnn_dailymail:3.0.0:test",
                license=SOURCES["cnn_dm"]["license"],
                provenance="public_benchmark",
                group_id=f"cnndm-{r['id']}",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the news article in 2-3 sentences that capture its main points. Do not add information that is not in the article.",
                    },
                    {"role": "user", "content": f"Article: {article}"},
                ],
                max_tokens=200,
                scorer="summary_judge",
                reference={"highlights": r["highlights"], "article": article},
            )
        )
    return out


def build_code(n: int, seed: int) -> list[EvalCase]:
    out = []
    n_he = n // 3
    for r in sample_rows("humaneval", n_he, seed)[:n_he]:
        out.append(
            _case(
                id=f"humaneval-{r['task_id'].split('/')[-1]}",
                category="code",
                source="openai/openai_humaneval:test",
                license=SOURCES["humaneval"]["license"],
                provenance="public_benchmark",
                group_id=r["task_id"],
                messages=[
                    {
                        "role": "system",
                        "content": "Complete the Python function. Return the complete function definition (with any needed imports) in a single ```python code block. Use only the standard library.",
                    },
                    {"role": "user", "content": r["prompt"]},
                ],
                max_tokens=600,
                scorer="code_tests",
                reference={"test_program": r["test"] + f"\n\ncheck({r['entry_point']})\n"},
            )
        )
    for r in sample_rows("mbpp", n - n_he, seed)[: n - n_he]:
        tests = "\n".join(r["test_list"])
        imports = "\n".join(r.get("test_imports") or [])
        out.append(
            _case(
                id=f"mbpp-{r['task_id']}",
                category="code",
                source="google-research-datasets/mbpp:sanitized:test",
                license=SOURCES["mbpp"]["license"],
                provenance="public_benchmark",
                group_id=f"mbpp-{r['task_id']}",
                messages=[
                    {
                        "role": "system",
                        "content": "Write a Python function that solves the task. Return only the code in a single ```python code block. Use only the standard library.",
                    },
                    {
                        "role": "user",
                        "content": f"{r['prompt']}\nYour code should pass this test:\n{r['test_list'][0]}",
                    },
                ],
                max_tokens=600,
                scorer="code_tests",
                reference={"test_program": f"{imports}\n{tests}\n"},
            )
        )
    return out


# ------------------------------------------------------------------ splits
def split_by_group(
    cases: list[EvalCase], seed: int, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> list[EvalCase]:
    by_cat: dict[str, list[EvalCase]] = defaultdict(list)
    for c in cases:
        by_cat[c.category].append(c)
    out: list[EvalCase] = []
    for cat, items in sorted(by_cat.items()):
        groups: dict[str, list[EvalCase]] = defaultdict(list)
        for c in items:
            groups[c.group_id].append(c)
        keys = sorted(groups)
        random.Random(f"{seed}:{cat}").shuffle(keys)
        total = len(items)
        cum = 0
        for k in keys:
            frac = cum / total
            split = (
                "train" if frac < ratios[0] else ("val" if frac < ratios[0] + ratios[1] else "test")
            )
            for c in groups[k]:
                out.append(c.model_copy(update={"split": split}))
            cum += len(groups[k])
    return out


def _shingles(text: str, k: int = 5) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower())
    return {t[i : i + k] for i in range(max(len(t) - k + 1, 1))}


def leakage_check(
    cases: list[EvalCase], threshold: float = 0.8
) -> tuple[list[EvalCase], dict[str, Any]]:
    """Drop val/test cases whose user prompt is a near-duplicate of any train prompt in the
    same category. Shared instruction text is excluded (only the user turn is compared)."""

    def user_text(c: EvalCase) -> str:
        return " ".join(m["content"] for m in c.messages if m["role"] == "user")

    sh = {c.id: _shingles(user_text(c)) for c in cases}
    train = defaultdict(list)
    for c in cases:
        if c.split == "train":
            train[c.category].append(c)
    dropped: list[dict[str, Any]] = []
    max_sim: dict[str, float] = {}
    keep: list[EvalCase] = []
    exact = {
        hashlib.sha256(user_text(c).encode()).hexdigest(): c.id for c in cases if c.split == "train"
    }
    for c in cases:
        if c.split == "train":
            keep.append(c)
            continue
        best, best_id = 0.0, None
        if hashlib.sha256(user_text(c).encode()).hexdigest() in exact:
            best, best_id = 1.0, exact[hashlib.sha256(user_text(c).encode()).hexdigest()]
        else:
            a = sh[c.id]
            for t in train[c.category]:
                b = sh[t.id]
                inter = len(a & b)
                if inter == 0:
                    continue
                j = inter / len(a | b)
                if j > best:
                    best, best_id = j, t.id
        max_sim[c.id] = round(best, 4)
        if best >= threshold:
            dropped.append(
                {"case": c.id, "split": c.split, "similar_train_case": best_id, "jaccard": best}
            )
        else:
            keep.append(c)
    report = {
        "method": "character 5-gram Jaccard between user turns, val/test vs. train within category",
        "threshold": threshold,
        "dropped": dropped,
        "max_similarity_by_split": {
            s: max(
                [v for k, v in max_sim.items() if next(x for x in cases if x.id == k).split == s]
                or [0.0]
            )
            for s in ("val", "test")
        },
    }
    return keep, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--name", default="routeforge-bench-v1")
    args = ap.parse_args()

    n = args.per_category
    builders = [
        build_factual,
        build_extraction,
        build_classification,
        build_summarization,
        build_code,
        build_reasoning,
    ]
    cases: list[EvalCase] = []
    for b in builders:
        part = b(n, args.seed)
        print(f"{b.__name__}: {len(part)} cases")
        cases.extend(part)
    json_cases = [_case(**c) for c in generate_json_cases(n, args.seed)]
    print(f"json: {len(json_cases)} cases")
    cases.extend(json_cases)

    ids = [c.id for c in cases]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        raise SystemExit(f"duplicate case ids: {dupes[:5]}")
    cases = split_by_group(cases, args.seed)
    cases, report = leakage_check(cases)

    out_dir = OUT_ROOT / args.name
    write_cases(out_dir / "cases.jsonl", cases)
    (out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2))
    manifest = {
        "name": args.name,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": file_sha256(out_dir / "cases.jsonl"),
        "n_cases": len(cases),
        "splits": dict(Counter(c.split for c in cases)),
        "categories": dict(Counter(c.category for c in cases)),
        "sources": [{"key": k, **v} for k, v in SOURCES.items()]
        + [
            {
                "key": "json_templates",
                "dataset": "router_training/datasets/json_templates.py",
                "license": "MIT",
                "split": "generated",
            }
        ],
        "split_method": "group-wise 60/20/20 within category; near-duplicate filter (5-gram Jaccard >= 0.8)",
        "notes": (
            f"seed={args.seed}. Public-benchmark answer keys are the datasets' own labels; they were not "
            "re-verified by a human for this project. JSON cases are synthetic and correct by construction "
            "(answers are generated alongside the text). No case was hand-labelled."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n_cases", "splits", "categories")}, indent=2))
    print(
        f"leakage: dropped {len(report['dropped'])} near-duplicates; max sim {report['max_similarity_by_split']}"
    )


if __name__ == "__main__":
    main()
