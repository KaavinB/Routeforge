# Design decisions

Each entry records the decision, the alternatives considered and the evidence. Where a
decision was changed after measurement, the history is kept.

## Architecture

**Modular monolith, not microservices.** Routing decisions take milliseconds and share
state (breakers, latency estimates, artifacts) with execution; splitting them into
services would add a network hop per request and distributed-state problems without any
scaling benefit at this size. Package boundaries (`routing`, `reliability`, `caching`,
`telemetry`, `evaluation`) are the seams along which it would split.

**LiteLLM SDK for connectivity only.** LiteLLM translates one request shape to OpenAI,
Anthropic and Ollama and normalizes usage. Its proxy server, router, retries, caching and
budgets are deliberately *not* used: those are the parts this project is about, and they
need to be measurable and explainable. LiteLLM retries are disabled (`num_retries=0`,
`max_retries=0`).

**Dependency cooldown.** Every dependency is pinned to the newest stable version released
at least 14 days earlier (Python and npm). LiteLLM itself had a PyPI supply-chain
compromise in 2026; a cooldown window is a cheap mitigation against that class of attack.

## Model pools

Two hosted pools are measured, plus a local one:

| Pool | Models (cheap → premium) | Why |
|---|---|---|
| `hosted-2026-10-11` | gpt-4.1-nano, claude-haiku-5-5, gpt-4.1-mini, claude-sonnet-5-5 | first experiment; non-reasoning OpenAI models |
| `latest-2026-10-11` | gpt-5.6-luna, claude-haiku-5-5, gpt-5.6-terra, claude-sonnet-5-5, claude-opus-5-5 | newest generation from both vendors (primary pool) |
| `local-2026-10-11` | llama3.2:1b, qwen2.5:3b, qwen2.5:7b (Ollama) | free, larger quality spread; illustrative cost units |

- GPT-5.6 models reason by default; they run with `reasoning_effort: none` so they behave
  as fast completion models (and accept `temperature=0`, which they reject while
  reasoning). `minimal` effort is not supported by these models (verified).
- Claude Haiku 5.5 runs with thinking disabled; Sonnet 5.5 rejects `disabled` and uses
  `between_tools` (its documented thinking-off mode).
- Claude Opus 5.5's thinking cannot be disabled; it runs at the lowest effort and gets a
  `min_max_tokens: 2048` floor so thinking tokens cannot starve the answer. Thinking tokens
  are billed as output and included in every cost figure.
- Claude 5.5 models reject non-default `temperature`; the gateway drops it and reports
  that in a response header (or rejects, by configuration) rather than hiding it.
- Haiku 5.5 and Sonnet 5.5 recordings are shared between the two hosted pools (identical
  model configuration), saving ~$1.2 of collection spend.

## Routing

**Per-model calibrated classifiers predicting P(acceptable).** One binary model per
candidate LLM. Alternatives: a single multi-output model with model one-hot features
(shares statistical strength but couples models: adding a model means retraining
everything), pairwise "strong vs weak" routers (RouteLLM-style, natural for two models
but awkward for five), and regression on a quality score (the decision only needs
"good enough or not"). Per-model binary classifiers make the decision rule explicit:
*cheapest model whose calibrated P(acceptable) clears the threshold*.

**Calibration first.** The threshold is meaningful only if probabilities are calibrated.
Each classifier is Platt-calibrated on out-of-fold predictions (grouped 5-fold CV on the
train split, `ensemble=False`). The validation split is used only for model selection
and to estimate each model's expected calibration error (ECE), which becomes its
**safety margin**: a model must clear `threshold + margin`. Requests that look
out-of-distribution (input longer than anything in training, or a low-confidence task
prediction) get an extra margin, which escalates them to stronger models.

**Lightweight features, no embeddings.** 26 lexical/structural features (token estimate,
code/math/JSON/extraction/summary markers, question words, requested `max_tokens`, …)
plus task probabilities from a TF-IDF + logistic-regression classifier. A sentence
embedding model would add ~100–500 MB and milliseconds per request; the ablation results
below show the cheap lexical features add little beyond task identity on this benchmark,
so a heavier encoder was not justified by evidence.

**Feature set chosen on validation.** Three variants are trained per model: `full`
(features + task probabilities), `no_task`, `task_only`. The production `learned` policy
uses, per model, whichever has the lowest validation log-loss. Validation differences
were tiny (≤0.005 log-loss), so this choice is close to arbitrary at n=158 — reported, not
hidden. Development note: test-set metrics for the ablation variants were inspected while
building the pipeline; the selection rule itself never consults the test split.

**Estimator family.** Logistic regression (two C values) vs. histogram gradient boosting.
Gradient boosting is selected only if it beats the best linear model by more than
0.005 validation log-loss (noise level), because linear models are simpler, behave better
out of distribution, and compile to a microsecond NumPy fast path.

**Cost model.** `in_tokens × in_price + E[out_tokens | model, task] × out_price`, where
`E[out_tokens]` is the training-split median per (model, task), weighted by predicted task
probabilities and capped by `max_tokens`. Models differ a lot in verbosity, so list price
alone ranks them wrongly.

**Latency model.** Per-model least-squares prior (`ttft + a·in + b·out`) from collection
runs, corrected online by an EWMA of observed/predicted latency (clamped per observation).

**Fallback policy when no model qualifies.** Highest predicted quality within the latency
budget; if none fits the budget, highest predicted quality overall (quality wins over
latency).

**Heuristic baseline.** Task → capability level (classification → economy; QA/extraction
→ low; summarization/JSON → mid; reasoning/code → premium), escalated for long context or
large `max_tokens`. It uses the rule-based task detector, never the learned classifier
(a bug where it silently used the learned classifier was found by profiling and fixed).

## Reliability

**Retry budget, not just per-request attempt limits.** Per-request limits still let
every request retry during an outage, multiplying load and cost. A token-bucket budget
(retries ≤ 20% of successful volume + 1/s floor) sheds retries when failure is
widespread. Worst case per request: 2 attempts × 3 models = 6 upstream calls; measured
maximum in the fault-injection benchmark: 4.

**Full-jitter exponential backoff**, honouring `Retry-After` as a lower bound; if a
provider asks to wait longer than our maximum backoff we fall back instead.

**Circuit breakers per model, per process.** Open on 5 consecutive failures or ≥50%
failures over the last 20 calls (≥10 samples), half-open after 30 s with a single probe.
Open breakers make the model ineligible *at routing time*, so traffic shifts without
paying a timeout per request. Per-process state is acceptable because detection needs
only a handful of calls per replica.

**Explicit model requests never fall back.** A client pinning `model=X` gets X or an
error; silently answering with a different model would invalidate benchmarks.

**Streaming.** Retries and fallbacks are possible until the first token; afterwards the
client has partial output, so the gateway emits an SSE error event and ends the stream.

## Caching

**Exact match only, deterministic requests only (default).** Semantic caching would trade
correctness for hit rate in ways that are hard to evaluate; exact-match keyed on tenant,
resolved model, registry fingerprint, messages and all adapted parameters can never
return an answer generated for a different request. Only `temperature=0` requests are
cached by default (`cache_mode=all` caches sampled answers too). Keys are looked up for
the *selected* model; a fallback answer is stored under the model that produced it.

**Degraded mode.** A Redis error marks the cache unavailable for 5 s so an outage does
not add a socket timeout to every request.

## Observability

- Prometheus labels are limited to registry-derived or fixed-enum values (model,
  strategy, status, cache result, error kind); request ids live in traces and logs only.
- Every request gets an OpenTelemetry trace id (returned as `x-routeforge-trace-id`,
  stored in the request log); export to Jaeger is optional. Langfuse receives a
  generation per request attached to the same trace id, with its own TracerProvider so it
  never hijacks the gateway's.
- The request log is written asynchronously in batches; requests never wait on Postgres.

## Performance work (measured)

| Problem | Evidence | Fix |
|---|---|---|
| Learned routing decision 362 ms under load | profiler: HistGradientBoosting `predict_proba` spawning OpenMP thread pools per single-row call; 5-fold calibrated ensembles; char n-gram TF-IDF | single-thread BLAS/OpenMP (`threadpool_limits(1)`, `OMP_NUM_THREADS=1`), `ensemble=False` calibration, word-only TF-IDF, NumPy fast path for calibrated logistic models (verified equal to sklearn on probes), linear-model tie-break → ~9 ms under heavy load; see Benchmark E for quiet-machine numbers |
| Feature extraction 11 ms on long prompts | per-regex timing: alternation-heavy case-insensitive regexes | lower-case once, keyword sets + `str.count` + literal-prefixed regexes → 4–5× faster |
| Periodic event-loop stalls | 375k objects tracked by the cyclic GC after importing LiteLLM et al.; a full collection took 97 ms | `gc.freeze()` after startup (full collection → 0 ms); GC-pause and event-loop-lag histograms added |
| Throughput collapse at 25+ concurrent clients in Docker | see Benchmark C notes | documented; investigation recorded in BENCHMARKS.md |

## Evaluation

- **Group-wise splits** within each category (SQuAD by article, JSON by template, others by
  item) plus a cross-split near-duplicate filter; zero near-duplicates were found.
- **Oracle validation**: every answer key and test program is checked by scoring the
  reference answer; 720/720 pass (one broken NQ label was found and filtered, two sandbox
  policy false positives were fixed).
- **Alias scorer v2**: v1 rejected e.g. "Santa Monica, California" for gold "in Santa
  Monica"; v2 adds a stopword-insensitive token-subset match. Applied identically to all
  models; changed factual-QA rates by 1–3 pp. NQ-open gold answers remain noisy.
- **Judge**: GPT-4.1 (not a routing candidate), reference-guided rubric, temperature 0,
  cached. Measured agreement with a second judge (Claude Fable 5.1): 91.7% raw, κ=0.31 —
  GPT-4.1 is lenient on summaries. Human calibration labels: not yet collected (kit
  provided).
- **Thresholds are pre-registered**: the summary acceptability rule (faithfulness ≥ 4 and
  coverage ≥ 3) was not tightened after seeing that every model passes ~99% of summaries.
