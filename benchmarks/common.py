"""Shared benchmark utilities: experiment metadata, result writing, chart style.

Every benchmark writes ``benchmarks/results/<name>/summary.json`` (machine-readable,
served to the dashboard), optional CSV files with per-item rows, and PNG charts.
``metadata`` always records what is needed to reproduce or audit the run.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"

PACKAGES = (
    "fastapi",
    "litellm",
    "scikit-learn",
    "numpy",
    "redis",
    "sqlalchemy",
    "uvicorn",
    "locust",
)


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "git_sha": run("rev-parse", "HEAD") or None,
        "git_dirty": bool(run("status", "--porcelain")),
    }


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        info["cpu"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        info["memory_gb"] = round(int(mem) / 2**30, 1) if mem else None
        info["cpu_count"] = os.cpu_count()
        # Recorded so readers can judge contention: the dev laptop also ran local LLM
        # inference and other apps during some benchmarks.
        info["load_average_1m_5m_15m"] = [round(x, 2) for x in os.getloadavg()]
    except (OSError, subprocess.SubprocessError, ValueError):
        info["cpu_count"] = os.cpu_count()
    return info


def experiment_metadata(benchmark: str, **extra: Any) -> dict[str, Any]:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "benchmark": benchmark,
        "experiment_id": f"{benchmark}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now(UTC).isoformat(),
        **git_info(),
        "hardware": hardware(),
        "software": versions,
        **extra,
    }


def write_summary(name: str, summary: dict[str, Any]) -> Path:
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    return path


def write_csv(name: str, filename: str, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (json.dumps(v) if isinstance(v, dict | list) else v) for k, v in r.items()}
            )
    return path


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------- chart style
# Reference categorical palette (fixed order, validated for adjacent-pair CVD safety).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9893"


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "lines.linewidth": 2,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt
