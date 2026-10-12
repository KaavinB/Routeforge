# Evaluation methodology

## Dataset: `routeforge-bench-v1` (version 1.0.0)

840 cases, 7 task categories × 120, built by `router_training/datasets/build.py`
(seed 20261011). Files: `router_training/datasets/data/routeforge-bench-v1/`
(`cases.jsonl`, `manifest.json` with SHA-256, `leakage_report.json`,
`oracle_validation.json`).

| Category | Source (public samples unless noted) | License | Scorer | Acceptable if |
|---|---|---|---|---|
| factual_qa | google-research-datasets/nq_open (validation) | CC BY-SA 3.0 | `alias_match` v2 | an alias appears as a phrase, or all its content tokens appear, in an answer ≤ 30 words |
| extraction | rajpurkar/squad v1.1 (validation) | CC BY-SA 4.0 | `span_f1` | exact match or token-F1 ≥ 0.6 |
| classification | fancyzhx/ag_news (test) | unspecified (research use) | `label` | exactly one label mentioned and it is the gold label |
| summarization | abisee/cnn_dailymail 3.0.0 (test), articles truncated to 450 words | Apache-2.0 | `summary_judge` | LLM judge: faithfulness ≥ 4 and coverage ≥ 3 (1–5 rubric, reference-guided) |
| code | openai_humaneval (MIT) + mbpp sanitized (CC BY 4.0) | MIT / CC BY 4.0 | `code_tests` | all unit tests pass in the sandbox |
| reasoning | openai/gsm8k main (test) | MIT | `numeric` | final number equals the reference |
| json | **synthetic**, 12 templates (`json_templates.py`) | MIT | `json_fields` | schema-valid and every expected field correct |

Provenance is recorded per case (`public_benchmark` / `synthetic_generated`). **No case
was hand-labelled or human-verified for this project**: public answer keys are the
datasets' own; JSON answer keys are correct by construction (generated together with the
text, with distractor values in the text).

### Splits and leakage control

- Group-wise 60/20/20 split *within each category*: SQuAD grouped by source article
  (≤ 2 questions per passage), JSON grouped by **template** (test templates are unseen
  during training), others by source item. Result: 521 train / 158 val / 161 test.
- Cross-split near-duplicate check: character 5-gram Jaccard between user turns of every
  val/test case and every train case in the same category; threshold 0.8. **0 cases
  dropped**; maximum similarity 0.64 (val) and 0.38 (test).
- Router training uses only `train`; model/feature-set selection and calibration-error
  margins use `val`; every headline metric is computed on `test` (161 cases).

### Oracle validation (are the answer keys and scorers right?)

`router_training/datasets/validate.py` scores the *reference* answer of every
deterministically scored case (gold number, gold span, gold label, expected JSON,
canonical HumanEval/MBPP solution executed in the Docker sandbox): **720 / 720 pass**.
Building it found and fixed: one NQ case whose gold answer was `")"` (filtered: answers
that normalize to empty are dropped), and two sandbox-policy false positives
(`import sys`, `exit()` in reference solutions).

## Scorers

All in `backend/app/evaluation/scorers.py` / `scoring.py`; each returns a score in
[0, 1] and a binary `acceptable` label. The binary label drives router training and
headline quality ("acceptable-answer rate").

**Alias scorer change (v1 → v2).** v1 required a normalized alias to appear as a phrase;
it rejected e.g. "Santa Monica, California" for gold "in Santa Monica". v2 also accepts
when every non-stopword token of an alias appears. Applied identically to every model;
it moved factual-QA rates by 1–3 pp. NQ-open's answer keys remain noisy and
time-sensitive (e.g. "when did X last…"), which caps every model's factual-QA score
(best: Opus 5.5 at 61.7%) and adds label noise to router training.

**Code sandbox.** Generated code passes an AST policy (stdlib allowlist, no `open`,
`eval`, `exec`, dunder attributes …), then runs in a throwaway container: no network,
read-only root, tmpfs, all capabilities dropped, uid 65534, memory/CPU/PID limits; inside
it each program runs in `python -I -S` with an empty environment, rlimits and a 10 s
timeout. Generated code never runs directly on the host.

## LLM-as-judge for summaries

- Judge: `openai/gpt-4.1`, temperature 0, JSON output, rubric version
  `summary-rubric-v1` (`backend/app/evaluation/judge.py`). Reference-guided: the judge
  sees the article, the dataset's editor-written highlights and the candidate.
- GPT-4.1 is **not a routing candidate** in either hosted pool, avoiding self-preference
  for its own outputs (it shares a vendor with some candidates).
- Verdicts are cached by (judge model, rubric, inputs); judging is deterministic, so
  caching never affects latency measurements, and CI replays reuse verdicts at zero cost.

### How reliable is the judge? (`router_training/judge_agreement.py`, Benchmark H)

Second judge: Claude Fable 5.1 (different vendor, not a candidate), same rubric, on 48
summaries (every summary the primary judge rejected plus a seeded random sample of
accepted ones, 12 per model).

| | Value |
|---|---|
| Raw agreement on `acceptable` | 91.7% |
| Cohen's κ | 0.31 (fair) |
| Accept rate: GPT-4.1 vs Fable 5.1 | 97.9% vs 89.6% |
| Agreement on GPT-4.1's rejections | 100% |
| Spearman ρ: coverage / faithfulness | 0.72 / 0.19 |

Interpretation: when GPT-4.1 rejects a summary, the second judge agrees; but GPT-4.1 is
**lenient** — it accepts ~8% of summaries Fable 5.1 rejects, mostly on coverage.
Faithfulness scores sit at the ceiling (≈4.7–4.9/5) for both judges, so their rank
correlation is low. Consequence: summarization contributes almost no routing signal
(every hosted model scores ~99–100% acceptable). The acceptability threshold was fixed
before results were seen and was **not** tightened afterwards.

**Human calibration: not measured.** A labelling kit with the same 48 items and blank
human columns is at
`router_training/data/collections/hosted-2026-10-11/judge_calibration_sample.csv`;
re-running `judge_agreement.py` after filling it in reports human–judge agreement.

## Router training (`router_training/`)

1. `collect.py` — every case × every candidate model, same per-model parameter
   adaptation as the gateway, cache-free; resumable; paid calls checked against a hard
   budget and appended to `benchmarks/results/spend_ledger.jsonl`.
2. `score.py` — labels every output (`labels.jsonl`).
3. `features/table.py` — builds the (case × model) table with
   `app.routing.features.extract_features`, the serving code path.
4. `training/train.py`
   - Task classifier: word 1–2-gram TF-IDF + multinomial logistic regression (train).
     Training rows for quality models use **out-of-fold** task probabilities (GroupKFold)
     so the quality models see realistically imperfect task predictions.
   - Per candidate model: L2 logistic regression (C ∈ {0.05, 0.5}) and histogram
     gradient boosting, each Platt-calibrated with grouped 5-fold CV on train
     (`ensemble=False`); chosen by validation log-loss (boosting only if it beats linear
     by > 0.005).
   - Variants `full`, `no_task`, `task_only`; production `selected` = lowest validation
     log-loss per model.
   - Safety margin per model = validation ECE (capped at 0.15).
   - Output-length table and latency priors from the train split.
   - Artifact: `router_training/artifacts/router-<collection>/` (`router.joblib`, SHA-256,
     `metadata.json` with every metric).

### Router quality (held-out test split)

Task-classifier accuracy is **100%** on val and test. This is a property of the benchmark
— each category uses one fixed instruction — not evidence of robust task detection on
real traffic. Prompts phrased unlike the benchmark are flagged out-of-distribution and
routed conservatively (to stronger models).

Per-model quality classifiers, latest pool (`router-latest-2026-10-11/metadata.json`):

| Model | Selected features | Test AUC | Test Brier | Val ECE (margin) |
|---|---|---|---|---|
| gpt-5.6-luna | no_task | 0.821 | 0.090 | see metadata |
| claude-haiku-5-5 | no_task | 0.831 | 0.093 | see metadata |
| gpt-5.6-terra | full | 0.846 | 0.078 | see metadata |
| claude-sonnet-5-5 | no_task | 0.821 | 0.077 | see metadata |
| claude-opus-5-5 | task_only | 0.895 | 0.056 | see metadata |

**What the ablations show.** On this benchmark most of the predictable signal is *which
task it is*; per-prompt lexical difficulty features add little and sometimes overfit
(the `task_only` variant has equal or better test AUC for several models). The router's
value therefore comes mainly from *learning, per task, which model is good enough*, plus
calibrated cost-aware selection — not from fine-grained difficulty estimation. A larger
and more heterogeneous training set (or embedding features) would be needed to learn
within-task difficulty; this is listed as future work rather than claimed.

## Regression gates (`backend/app/evaluation/regression.py`)

A candidate configuration is compared with a reference on the same prompts using paired
bootstrap (4,000 resamples):

| Gate | Fails / warns when |
|---|---|
| quality_drop | FAIL if the 95% CI upper bound of (candidate − reference) acceptable rate < −2 pp; WARN if only the point estimate is below −2 pp |
| latency_increase | WARN if the CI lower bound of the p50 latency change > +25% |
| cost_per_acceptable_increase | WARN if the CI lower bound > +20% |
| structured_output_validity | FAIL if > 5% of JSON cases are schema-invalid |

Validation of the gate itself is in BENCHMARKS.md (Benchmark G): false alarms on neutral
changes, and a power curve against synthetic regressions of known size.

## Statistics used throughout

- Acceptable-rate intervals: Wilson 95% intervals.
- Policy comparisons: paired bootstrap over prompts (the same prompts answered by both
  policies), reporting point estimate, 95% percentile interval and a bootstrap p-value.
- Cost reduction: `100 × (baseline_cost − candidate_cost) / baseline_cost` over the
  whole test set. Quality retention: `100 × candidate_rate / baseline_rate`.
- Calibration: expected calibration error (10 equal-width bins) and Brier score.
