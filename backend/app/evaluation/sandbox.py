"""Isolated execution of model-generated code for unit-test scoring.

Defence in depth — model output is treated as untrusted:

1. Static policy (``check_policy``): an AST pass rejects imports outside a small stdlib
   allowlist and references to dangerous builtins / dunder attributes. This is a filter,
   not a security boundary.
2. Process isolation: each program runs in a fresh ``python -I -S`` interpreter with an
   empty environment (no credentials), a temporary working directory, CPU/file-size/
   open-file rlimits and a wall-clock timeout.
3. Container isolation (``backend="docker"``, the default for CLI scoring on a developer
   machine): the whole batch runs inside a throwaway container with no network,
   read-only root filesystem, dropped capabilities, ``no-new-privileges``, an
   unprivileged user, and memory/CPU/PID limits. Generated code therefore never executes
   directly on the host.

Inside the API container the ``subprocess`` backend is used (the container is already the
isolation boundary); docs/SECURITY.md records the residual risk.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "re",
        "string",
        "itertools",
        "functools",
        "collections",
        "heapq",
        "bisect",
        "typing",
        "operator",
        "statistics",
        "fractions",
        "decimal",
        "datetime",
        "copy",
        "dataclasses",
        "enum",
        "random",
        "array",
        "cmath",
        "numbers",
        "hashlib",
        "sys",
    }
)
BANNED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)

SANDBOX_IMAGE = "python:3.12-slim"

# Executed inside the sandbox (container or local interpreter). Self-contained on purpose.
_RUNNER = r"""
import json, os, resource, subprocess, sys, tempfile
jobs = json.load(sys.stdin)
def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
    except (ValueError, OSError):
        pass  # not enforceable on macOS
out = []
for job in jobs:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as fh:
            fh.write(job["program"])
        try:
            r = subprocess.run([sys.executable, "-I", "-S", path], cwd=d, env={},
                               capture_output=True,
                               text=True, timeout=job["timeout"], preexec_fn=limits)
            out.append({"id": job["id"], "passed": r.returncode == 0, "detail": r.stderr[-400:]})
        except subprocess.TimeoutExpired:
            out.append({"id": job["id"], "passed": False, "detail": "timeout"})
json.dump(out, sys.stdout)
"""


@dataclass
class CodeJob:
    id: str
    program: str
    timeout_s: float = 10.0


@dataclass
class CodeResult:
    id: str
    passed: bool
    detail: str


def check_policy(code: str) -> str | None:
    """Return a violation message, or None if the code passes the static policy."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import not allowed: {name}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"name not allowed: {node.id}"
        elif (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr not in ("__name__", "__init__")
        ):
            return f"attribute not allowed: {node.attr}"
    return None


def docker_available() -> bool:
    return shutil.which("docker") is not None


async def run_jobs(
    jobs: list[CodeJob], backend: str = "docker", batch_timeout_s: float = 300.0
) -> list[CodeResult]:
    if not jobs:
        return []
    payload = json.dumps([{"id": j.id, "program": j.program, "timeout": j.timeout_s} for j in jobs])
    if backend == "docker":
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("docker sandbox requested but the docker CLI is not on PATH")
        # The docker *client* needs HOME/PATH to find its context; the container itself
        # receives no host environment variables.
        client_env = {
            k: v
            for k, v in os.environ.items()
            if k in ("HOME", "PATH", "DOCKER_HOST", "DOCKER_CONTEXT")
        }
        cmd = [
            docker,
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",  # noqa: S108 - tmpfs inside the sandbox container
            "--memory",
            "768m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            SANDBOX_IMAGE,
            "python",
            "-I",
            "-c",
            _RUNNER,
        ]
    elif backend == "subprocess":
        cmd = [sys.executable, "-I", "-c", _RUNNER]
        client_env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
    else:
        raise ValueError(f"unknown sandbox backend {backend}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=client_env,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(payload.encode()), timeout=batch_timeout_s
        )
    except TimeoutError:
        proc.kill()
        return [CodeResult(j.id, False, "sandbox batch timeout") for j in jobs]
    if proc.returncode != 0:
        detail = err.decode(errors="replace")[-400:]
        return [CodeResult(j.id, False, f"sandbox error: {detail}") for j in jobs]
    rows: list[dict[str, Any]] = json.loads(out)
    return [CodeResult(r["id"], bool(r["passed"]), r.get("detail", "")) for r in rows]
