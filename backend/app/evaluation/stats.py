"""Uncertainty estimates for evaluation and benchmark metrics.

All comparative metrics use *paired* resampling: when two policies answer the same
prompts, we resample prompt indices and recompute both policies' statistics on the same
resample, which removes between-prompt variance from the comparison.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

DEFAULT_RESAMPLES = 5000


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(resamples, len(arr)))
    stats = np.array([stat(arr[i]) for i in idx])
    return (
        float(stat(arr)),
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - alpha / 2)),
    )


def paired_bootstrap(
    a: Mapping[str, Sequence[float]],
    b: Mapping[str, Sequence[float]],
    metric: Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Generic paired bootstrap. ``a`` and ``b`` map column name -> per-prompt values,
    aligned by prompt. ``metric(a_cols, b_cols)`` computes the comparison statistic
    (e.g. relative cost reduction). Returns point estimate, CI and a two-sided
    bootstrap p-value for the hypothesis metric == 0."""
    a_np = {k: np.asarray(v, dtype=float) for k, v in a.items()}
    b_np = {k: np.asarray(v, dtype=float) for k, v in b.items()}
    n = len(next(iter(a_np.values())))
    if n == 0:
        return {
            "estimate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_value": math.nan,
            "n": 0,
        }
    point = metric(a_np, b_np)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    for s in range(resamples):
        idx = rng.integers(0, n, size=n)
        samples[s] = metric(
            {k: v[idx] for k, v in a_np.items()}, {k: v[idx] for k, v in b_np.items()}
        )
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {
            "estimate": point,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "p_value": math.nan,
            "n": n,
        }
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {
        "estimate": float(point),
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
        "p_value": float(min(1.0, p)),
        "n": n,
    }


def cost_reduction_pct(base: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> float:
    total = base["cost"].sum()
    return float(100 * (total - cand["cost"].sum()) / total) if total > 0 else math.nan


def quality_retention_pct(base: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> float:
    q = base["acceptable"].mean()
    return float(100 * cand["acceptable"].mean() / q) if q > 0 else math.nan


def quality_diff_pp(base: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> float:
    return float(100 * (cand["acceptable"].mean() - base["acceptable"].mean()))


def latency_change_pct(
    base: dict[str, np.ndarray], cand: dict[str, np.ndarray], q: float = 50
) -> float:
    b = np.percentile(base["latency_ms"], q)
    return float(100 * (b - np.percentile(cand["latency_ms"], q)) / b) if b > 0 else math.nan


def ece(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return math.nan
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else math.nan
