# Benchmarks

Every number below comes from a recorded run whose machine-readable output is in
`benchmarks/results/<name>/summary.json` (served to the dashboard's Benchmarks page),
usually with per-request CSVs and charts alongside. Each summary's `metadata` records
experiment id, timestamp, git SHA (+ dirty flag), dataset version and hash, registry and
pricing version, router artifact, software versions, hardware and — for runs after it was
added — the machine's load average.

Environment for all runs: MacBook (Apple M4, 10 cores, 16 GB), Docker Desktop
(Compose deployment: one gateway container, 1 uvicorn worker, `cpus: 2.0`, 1 GB),
PostgreSQL 17, Redis 7.4, providers in us regions, 2026-10-11/12.

All paid API usage is in `benchmarks/results/spend_ledger.jsonl`: **$9.36 total**
(collection $4.60, LLM judging $3.42, live benchmarks $1.34) against a $20 cap.

## Model pools

| Pool | Models (cheap → premium), list $ / 1M tokens in/out |
|---|---|
| **latest** (primary) | gpt-5.6-luna 0.20/1.20 · claude-haiku-5-5 0.10/0.50 · gpt-5.6-terra 2.00/12.00 · claude-sonnet-5-5 2.00/10.00 · claude-opus-5-5 4.00/20.00 |
| hosted (earlier) | gpt-4.1-nano 0.10/0.40 · claude-haiku-5-5 · gpt-4.1-mini 0.40/1.60 · claude-sonnet-5-5 |

Costs everywhere are **estimates**: provider-reported tokens × list price (no billing
data). Opus 5.5 always thinks; its thinking tokens are billed as output and included.

---

## A. Routing-policy comparison

Held-out test split: 161 prompts across 7 categories (`routeforge-bench-v1`). Two
methods: **offline replay** (every model answered every prompt during collection; a
policy picks a model per prompt and inherits that recorded, scored outcome — fully
paired, no extra spend) and **live** (every policy answers every prompt end-to-end
through the running gateway, fresh generations, same scorers and judge).

### Latest pool — live (`A_routing_live_latest`)

| Policy | Acceptable (95% CI) | $ / 1k requests | p50 latency (end-to-end) |
|---|---|---|---|
| premium: always claude-opus-5-5 | 90.7% | $2.99 | 1,532 ms |
| **learned router, τ = 0.8** | **89.4%** | **$0.32** | **1,096 ms** |
| random | 90.1% | $1.09 | 1,155 ms |
| heuristic | 87.6% | $1.52 | 1,120 ms |
| economy: always gpt-5.6-luna | 85.7% | $0.085 | 1,173 ms |

Paired vs. premium (bootstrap over prompts):

| Policy | Cost reduction | Quality Δ | Quality retention |
|---|---|---|---|
| **learned** | **89.1% [82.1, 93.9]** | **−1.2 pp [−4.3, +1.9]** | **98.6%** |
| heuristic | 49.2% [43.0, 54.6] | −3.1 pp [−7.5, +0.6] | 96.6% |
| random | 63.4% [56.0, 70.7] | −0.6 pp [−4.3, +3.1] | 99.3% |
| economy | 97.2% [97.0, 97.3] | −5.0 pp [−9.3, −0.6] | 94.5% |

The learned router served 81% of requests with claude-haiku-5-5, 13% with
claude-opus-5-5 and the rest with terra/sonnet/luna.

### Latest pool — offline replay (`A_routing_offline_latest`)

| Policy | Acceptable | $ / 1k req | Notes |
|---|---|---|---|
| fixed opus-5-5 (premium) | 92.5% [87.4, 95.7] | $2.99 | |
| fixed sonnet-5-5 | 89.4% | $1.28 | dominated by learned |
| fixed terra | 88.8% | $0.89 | dominated by learned |
| fixed luna (economy) | 87.0% | $0.085 | dominated by learned τ = 0.5 (87.0%, $0.077) |
| fixed haiku-5-5 | 85.7% | $0.063 | |
| random | 89.4% | $1.07 | dominated |
| heuristic | 87.0% | $1.54 | dominated |
| **learned τ = 0.8** | **89.4%** | **$0.29** | quality-violation rate 6.6% |
| learned, no cost model | 91.3% | $1.07 | ablation |
| oracle (cheapest acceptable model per prompt) | 95.0% | $0.11 | upper bound, not achievable |

Paired, learned (τ = 0.8) vs: **opus** −90.3% cost [85.0, 94.2], −3.1 pp [−6.2, −0.6];
**sonnet** −77.5% cost at +0.0 pp [−3.7, +3.7]; **terra** −67.5% cost at +0.6 pp;
**haiku** +359% cost for **+3.7 pp [+0.6, +7.5]** (significant); **luna** +241% cost for
+2.5 pp [−1.9, +6.8] (not significant).

Offline and live agree within ~2 pp for every policy (live Opus is nondeterministic:
92.5% offline vs 90.7% live).

![Pareto frontier, latest pool](../benchmarks/results/A_routing_offline_latest/charts/pareto.png)

### Earlier GPT-4.1 pool (`A_routing_live_hosted`, `A_routing_offline_hosted`)

Live: learned 87.0% at $0.095/1k vs premium (Sonnet 5.5) 89.4% at $1.27/1k →
**−92.5% cost [90.2, 94.3], −2.5 pp [−6.2, +1.2]**, p50 963 ms vs 1,204 ms. But in this pool
the learned router was **not significantly better than always using claude-haiku-5-5**
(+1.2 pp [−1.9, +5.0] at 51% higher cost): one cheap model captured nearly all of the
value. With the latest pool the router does beat the best cheap model (above), because
there is a stronger premium tier worth escalating to.

### Ablations (latest pool, offline, τ = 0.8)

| Variant | Acceptable | $ / 1k | Reading |
|---|---|---|---|
| learned (validation-selected features) | 89.4% | $0.29 | |
| request features + task probs ("full") | 88.8% | $0.31 | no gain from combining |
| task probabilities only | 88.8% | $0.31 | most signal is task identity |
| no cost model (random among qualifying models) | 91.3% | $1.07 | cost modelling cuts spend 3.7× for −1.9 pp |
| latency budget 1,500 ms vs ignoring it | 88.2% vs 89.4% | $0.12 vs $0.29 | latency model halves budget violations (5.6% vs 10.6%) |

Differences between feature variants are within noise at n = 161 (see EVALUATION.md).

## B. Caching (`B_cache_hosted`)

Real provider (gpt-5.6-luna, temperature 0), 80 unique held-out prompts, concurrency 8.

| Phase | Hit rate | p50 | p95 | Backend calls |
|---|---|---|---|---|
| cache bypassed | — | 1,068 ms | 2,120 ms | 80 |
| cold (flushed) | 0% | 836 ms | 1,229 ms | 80 |
| warm | 100% | **10.9 ms** | 25.6 ms | **0** |
| mixed, 26% repeated prompts | 24.0% | 963 ms | 1,685 ms | 152 / 200 |

Hits cut p50 latency by 99% and avoid the provider call entirely. The server-side
cache lookup averages 1.34 ms (what a miss costs). The mixed workload's repetition rate is
an **assumption**; real hit rates depend entirely on traffic (repeats that arrive in the
same batch as their first occurrence miss, hence 24% < 26%). Only deterministic
(`temperature = 0`) requests are cached by default.

## C. Throughput and latency under load

### C1. Gateway capacity (`C_capacity_replay`)

Replay mode (recorded real outputs, router artifact loaded, **zero provider latency**),
`hey` closed loop, 4,000 requests per level, cache bypassed, request logging on.

| Clients | Pinned model (no routing) | Learned routing | Learned p95 |
|---|---|---|---|
| 1 | 965 req/s, p50 1.0 ms | 615 req/s, p50 1.5 ms | 2.2 ms |
| 10 | 1,236 | 780 | 18.4 ms |
| 25 | **1,334** | **854** | 35.2 ms |
| 50 | 1,235 | 787 | 72.5 ms |
| 100 | 1,193 | 718 | 147 ms |

0 errors at every level. Routing (features, task classifier, five calibrated quality
models, cost/latency estimates, decision record) costs ≈ 0.5 ms of CPU per request, which
lowers a single worker's peak throughput by ~36% — negligible against ~1 s provider
latency, and the gateway scales horizontally (stateless apart from per-process breakers).

**Measurement lesson.** Earlier runs suggested throughput "collapsed" above ~10 clients.
That was wrong three times over: (1) configuration knobs set in the shell never reached
the container (Compose passes only listed variables), so an A/B compared identical
configs; (2) local LLM inference was saturating the CPU; (3) the single-process
Python/httpx load generator was itself the bottleneck. With `hey` the server holds
~1.2–1.4k req/s up to 100 clients, and latency grows linearly with concurrency as
Little's law predicts for a saturated single worker.

### C2. Realistic closed loop (`C_load_mock_realistic_latency`)

Locust, 30 s per level, random held-out prompts, mock providers with simulated
latency (0.2–2 s per request), default heuristic routing.

| Users | Successful req/s | p50 | p95 | Errors |
|---|---|---|---|---|
| 1 | 0.9 | 610 ms | 1,900 ms | 0% |
| 10 | 8.3 | 1,200 ms | 1,900 ms | 0% |
| 25 | 20.6 | 1,000 ms | 1,900 ms | 0% |
| 50 | 35.7 | 620 ms | 2,800 ms | 0% |
| 100 | 49.2 | 1,000 ms | 3,800 ms | 0% |

Throughput scales linearly until the per-model concurrency bulkhead (32 in-flight
calls per model) is reached; beyond that requests wait (≤ 2 s) for a slot or overflow to
the next model, which raises p95 without errors. Simulated providers: this measures
gateway behaviour, not provider throughput.

## D. Failure injection (`D_faults_mock`)

60 requests per scenario (premium policy, concurrency 6) against mock providers with
faults injected in front of them; outcomes reconstructed from the gateway's per-attempt
request log.

| Scenario | Success | Recovered (retry / fallback) | Max upstream calls | Recovery overhead p50 / p95 |
|---|---|---|---|---|
| baseline | 100% | — | 1 | — |
| 20% HTTP 500 (transient) | 100% | 16 / 4 of 20 | 3 | 131 / 248 ms |
| 50% HTTP 429 (Retry-After 0.2 s) | 100% | 1 / 51 of 52 | 3 | 8 / 211 ms |
| 30% malformed responses | 100% | 0 / 58 of 58 | 3 | 8 / 44 ms |
| persistent HTTP 500 | 100% | 60 fallback | 2 | 7 / 102 ms |
| connection outage | 100% | 60 fallback | 2 | 8 / 120 ms |
| hang until 3 s timeout | 100% | 60 fallback | 2 | 7 ms / 3.05 s |
| two models down | 100% | 60 fallback | 4 | 7 / 148 ms |
| Redis stopped | 100% | graceful degradation (cache miss, local rate limiter) | — | — |
| PostgreSQL stopped | 100% | graceful degradation (bootstrap-key auth, logs dropped) | — | — |
| stream aborted after first token | client receives SSE error event | not recoverable by design | — | — |

**Totals:** 370 / 370 fault-affected requests recovered (100%); 353 / 353 that needed
another model were recovered by fallback. Retry amplification stayed ≤ 4 upstream calls
per request (hard bound 6) because circuit breakers opened after 3–12 requests and made
the failing model ineligible at routing time. Recovery here means "the request succeeded
on some model"; a fallback answer may differ in quality from the primary's.

Breaker sensitivity: the consecutive-failure counter (threshold 5) includes retry
attempts, so even a 30% error rate opens the breaker after a short unlucky streak (two
requests failing both attempts back to back). Traffic then moves to a healthy model
instead of paying retries — a deliberate fail-fast choice; raise
`ROUTEFORGE_BREAKER_FAILURE_THRESHOLD` to favour retries on a partially degraded provider.

## E. Router overhead (`E_router_overhead`)

In-process, 161 held-out prompts × 5 repeats, BLAS/OpenMP single-threaded:

| Stage | p50 | p95 |
|---|---|---|
| feature extraction | 0.08 ms | 0.49 ms |
| task classifier (TF-IDF + LR) | 0.22 ms | 0.39 ms |
| 5 quality models (NumPy fast path) | 0.015 ms | 0.018 ms |
| **full learned decision** | **0.32 ms** | **0.88 ms** |
| heuristic decision | 0.30 ms | 0.86 ms |

End-to-end over HTTP in replay mode (sequential, 322 requests each): pinned model p50
1.51 ms, heuristic 2.40 ms, **learned 2.55 ms (+1.04 ms)**.

Optimization history (same artifact family, measured during development under heavy
machine load): 362 ms → 51 ms (single-threaded OpenMP) → ~9 ms (non-ensemble calibration,
word-only TF-IDF, NumPy fast path) → 0.32 ms on a quiet machine.

## F. Quality–cost Pareto frontier

The threshold sweep (τ ∈ 0.3 … 0.99) is in `A_routing_offline_*` (`pareto` key, chart
above). Latest pool: the learned frontier dominates sonnet-5-5, terra, random and the
heuristic and luna; it peaks at 91.9% at τ = 0.97 ($2.04/1k) and never matches Opus's 92.5%.
Above τ ≈ 0.8 cost rises ~7× for ~2.5 pp — the classifiers cannot pinpoint which
individual prompts only Opus answers correctly. Earlier pool: quality plateaus at
τ ≈ 0.8 (~87%) while cost keeps rising.

## G. Regression-gate validation (`G_regression_gate_validation_latest`)

Replay-based (deterministic, $0). Reference: learned router τ = 0.8 on the test split.

- **Neutral configuration changes** (identical re-run, τ = 0.85, feature-set ablation,
  no-cost ablation): 0 / 4 blocked; the cost gate warned on the no-cost variant (correct:
  it is 3.7× more expensive).
- **Power curve with synthetic regressions** (flip exactly *k* % of acceptable answers;
  100 trials per size). Pure degradation: 0% false blocks at 0 pp, 100% blocked at ≥ 6 pp.
  With realistic churn (4% of answers additionally swap in both directions):

  | True drop | 0 pp | 2 pp | 3–6 pp | 8 pp | 10 pp |
  |---|---|---|---|---|---|
  | Blocked (FAIL) | 0% | 0% | 0% | 81% | 100% |
  | Flagged (FAIL or WARN) | 0% | 0% | 100% | 100% | 100% |

  The gate is conservative by design: at n = 161 it only blocks when the 95% CI of the
  paired drop lies entirely below −2 pp, so drops of 3–6 pp are reported as warnings
  rather than blocking a merge.

## H. LLM-judge agreement (`H_judge_agreement`)

See EVALUATION.md: 91.7% raw agreement, κ = 0.31 between GPT-4.1 and Claude Fable 5.1;
GPT-4.1 is the more lenient judge.

## Reproducing

```bash
make dataset validate-dataset                 # dataset + oracle checks
make collect score train COLLECTION=latest-2026-10-11 REGISTRY=config/models/hosted-latest.yaml   # paid
make replay                                   # A + F offline (free)
make regression-suite                         # G (free)
# gateway benchmarks (start the stack in the right mode first; see README)
make live-benchmark                           # A live (paid, budget-capped)
uv run python -m benchmarks.cache_bench --model gpt-5.6-luna      # B (cents)
uv run python -m benchmarks.load_tests.capacity                   # C1 (replay mode)
make load-test                                                    # C2 (mock mode)
make fault-test                                                   # D (mock + fault injection)
make overhead BASE_URL=http://localhost:8000                      # E (replay mode)
uv run python -m router_training.judge_agreement                  # H (paid, ~$1.6)
```

## Limitations

- **Multi-replica behaviour** (shared Redis rate limits, per-process breakers) is
  covered by integration tests, not load-tested.
- **Capacity vs. real providers:** throughput was measured with zero-latency replay and
  mock providers; at high concurrency, real providers' own rate limits dominate.
- Single dataset version, single region, one week of provider behaviour.
