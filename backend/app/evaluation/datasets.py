"""Versioned evaluation datasets (JSONL + manifest)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Split = Literal["train", "val", "test"]


class EvalCase(BaseModel):
    id: str
    category: str
    source: str
    license: str
    provenance: Literal["public_benchmark", "synthetic_generated", "hand_written"]
    group_id: str
    split: Split
    messages: list[dict[str, str]]
    max_tokens: int = 256
    response_format: dict[str, Any] | None = None
    scorer: str
    reference: dict[str, Any] = Field(default_factory=dict)

    def request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"max_tokens": self.max_tokens, "temperature": 0}
        if self.response_format:
            params["response_format"] = self.response_format
        return params


class DatasetManifest(BaseModel):
    name: str
    version: str
    created_at: str
    sha256: str
    n_cases: int
    splits: dict[str, int]
    categories: dict[str, int]
    sources: list[dict[str, Any]]
    split_method: str
    notes: str = ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> list[EvalCase]:
    with path.open() as fh:
        return [EvalCase.model_validate_json(line) for line in fh if line.strip()]


def write_cases(path: Path, cases: list[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in cases:
            fh.write(c.model_dump_json(exclude_none=True) + "\n")


class Dataset:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = DatasetManifest.model_validate_json(
            (directory / "manifest.json").read_text()
        )
        cases_path = directory / "cases.jsonl"
        digest = file_sha256(cases_path)
        if digest != self.manifest.sha256:
            raise ValueError(
                f"dataset {directory} does not match its manifest checksum "
                "(edited without re-versioning?)"
            )
        self.cases = load_cases(cases_path)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    def select(
        self,
        split: str | None = None,
        categories: list[str] | None = None,
        limit: int | None = None,
    ) -> list[EvalCase]:
        out = [
            c
            for c in self.cases
            if (split is None or c.split == split) and (not categories or c.category in categories)
        ]
        return out[:limit] if limit else out


def resolve_dataset(datasets_dir: Path, name: str) -> Dataset:
    directory = (datasets_dir / name).resolve()
    if datasets_dir.resolve() not in directory.parents:
        raise ValueError("invalid dataset name")
    return Dataset(directory)


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)
