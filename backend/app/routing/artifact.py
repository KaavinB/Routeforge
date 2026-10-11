"""Loading and querying trained router artifacts.

An artifact directory (produced by ``router_training.training.train``) contains:

* ``router.joblib`` — task classifier, per-model calibrated quality classifiers (one set
  per feature variant), output-token table, latency priors, safety margins.
* ``router.joblib.sha256`` — integrity digest, verified before unpickling.
* ``metadata.json`` — human-readable training config, dataset version and metrics.

joblib uses pickle, so artifacts must come from a trusted build; the digest check guards
against accidental corruption or substitution, not a malicious build pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.routing.features import FEATURE_NAMES, TASK_CATEGORIES, RequestFeatures
from app.routing.latency import LatencyPrior

ARTIFACT_FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    pass


class FastCalibratedLogistic:
    """Pure-NumPy inference for ``CalibratedClassifierCV(Pipeline(StandardScaler,
    LogisticRegression), method="sigmoid", ensemble=False)``:

        z = ((x - mean) / scale) . w + b ;  p = 1 / (1 + exp(A * z + B))

    sklearn's generic predict_proba path costs ~1-5 ms per call (input validation,
    pipeline dispatch, thread-pool setup); this is a few microseconds. Built only after
    verifying it reproduces sklearn's probabilities on probe inputs.
    """

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        a: float,
        b: float,
    ) -> None:
        self.mean, self.scale, self.coef = mean, scale, coef
        self.intercept, self.a, self.b = intercept, a, b

    @classmethod
    def try_compile(cls, est: Any, n_features: int) -> FastCalibratedLogistic | None:
        try:
            if len(est.calibrated_classifiers_) != 1:
                return None
            cc = est.calibrated_classifiers_[0]
            pipe = cc.estimator
            scaler, lr = pipe.named_steps["scale"], pipe.named_steps["lr"]
            cal = cc.calibrators[0]
            fast = cls(
                np.asarray(scaler.mean_, dtype=float),
                np.asarray(scaler.scale_, dtype=float),
                np.asarray(lr.coef_[0], dtype=float),
                float(lr.intercept_[0]),
                float(cal.a_),
                float(cal.b_),
            )
        except (AttributeError, KeyError, IndexError, TypeError):
            return None
        rng = np.random.default_rng(0)
        probe = fast.mean + rng.normal(size=(16, n_features)) * fast.scale
        expected = est.predict_proba(probe)[:, 1]
        got = np.array([fast.predict_one(row) for row in probe])
        return fast if np.allclose(expected, got, atol=1e-9) else None

    def predict_one(self, x: np.ndarray) -> float:
        z = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(self.a * z + self.b)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_bundle(bundle: dict[str, Any], directory: Path, metadata: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "router.joblib"
    joblib.dump(bundle, path, compress=3)
    digest = sha256_file(path)
    (directory / "router.joblib.sha256").write_text(digest + "\n")
    metadata = {**metadata, "artifact_sha256": digest}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return path


class RouterArtifact:
    def __init__(self, bundle: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        if bundle.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ArtifactError("unsupported router artifact format")
        if tuple(bundle["feature_names"]) != FEATURE_NAMES:
            raise ArtifactError(
                "artifact feature set does not match this code version; retrain the router"
            )
        self.bundle = bundle
        self.metadata = metadata or {}
        self.version: str = bundle["router_version"]
        self.models: list[str] = list(bundle["models"])
        self.task_classifier = bundle["task_classifier"]
        self.quality_models: dict[str, dict[str, Any]] = bundle["quality_models"]
        self.margins: dict[str, dict[str, float]] = bundle["margins"]
        self.output_tokens: dict[str, dict[str, float]] = bundle["output_tokens"]
        self.latency_priors = {
            m: LatencyPrior(**p) for m, p in bundle.get("latency_priors", {}).items()
        }
        self.selected_feature_sets: dict[str, str] = dict(bundle.get("selected_feature_sets", {}))
        # Compile linear models to the NumPy fast path where possible.
        self._fast: dict[tuple[str, str], FastCalibratedLogistic] = {}
        n_feat = {
            "full": len(FEATURE_NAMES) + len(TASK_CATEGORIES),
            "no_task": len(FEATURE_NAMES),
            "task_only": len(TASK_CATEGORIES),
        }
        for variant, models in self.quality_models.items():
            for model_id, est in models.items():
                fs = (
                    self.selected_feature_sets.get(model_id, "full")
                    if variant == "selected"
                    else variant
                )
                fast = FastCalibratedLogistic.try_compile(est, n_feat.get(fs, 0))
                if fast is not None:
                    self._fast[(variant, model_id)] = fast
        self.default_variant: str = bundle.get("default_variant", "full")
        self.max_train_input_tokens: float = float(bundle.get("max_train_input_tokens", 1e9))
        self.ood_task_confidence: float = float(bundle.get("ood_task_confidence", 0.4))

    @classmethod
    def load(cls, directory: Path) -> RouterArtifact:
        path = directory / "router.joblib"
        digest_file = directory / "router.joblib.sha256"
        if not path.exists() or not digest_file.exists():
            raise ArtifactError(f"router artifact missing in {directory}")
        expected = digest_file.read_text().strip()
        if sha256_file(path) != expected:
            raise ArtifactError("router artifact checksum mismatch; refusing to load")
        bundle = joblib.load(path)  # trusted, checksum-verified artifact
        meta_path = directory / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(bundle, metadata)

    @property
    def variants(self) -> list[str]:
        return list(self.quality_models.keys())

    def task_probs(self, text: str) -> dict[str, float]:
        probs = self.task_classifier.predict_proba([text])[0]
        classes = list(self.task_classifier.classes_)
        out = dict.fromkeys(TASK_CATEGORIES, 0.0)
        for cls_name, p in zip(classes, probs, strict=True):
            out[str(cls_name)] = float(p)
        return out

    @staticmethod
    def design_row(
        features: RequestFeatures, task_probs: dict[str, float], variant: str
    ) -> np.ndarray:
        x = features.vector()
        if variant == "no_task":
            return x.reshape(1, -1)
        tp = np.array([task_probs[t] for t in TASK_CATEGORIES], dtype=np.float64)
        if variant == "task_only":
            return tp.reshape(1, -1)
        return np.concatenate([x, tp]).reshape(1, -1)

    def feature_set(self, variant: str, model_id: str) -> str:
        """Feature set used by ``variant`` for ``model_id``. The ``selected`` variant picks
        a feature set per model (chosen on validation data at training time)."""
        if variant == "selected":
            return self.selected_feature_sets.get(model_id, "full")
        return variant

    def quality_probs(
        self, features: RequestFeatures, task_probs: dict[str, float], variant: str = "full"
    ) -> dict[str, float]:
        rows: dict[str, np.ndarray] = {}
        out: dict[str, float] = {}
        for m, est in self.quality_models[variant].items():
            fs = self.feature_set(variant, m)
            if fs not in rows:
                rows[fs] = self.design_row(features, task_probs, fs)
            fast = self._fast.get((variant, m))
            out[m] = (
                fast.predict_one(rows[fs][0])
                if fast is not None
                else float(est.predict_proba(rows[fs])[0, 1])
            )
        return out

    @property
    def compiled_models(self) -> int:
        return len(self._fast)

    def margin(self, variant: str, model_id: str) -> float:
        return float(self.margins.get(variant, {}).get(model_id, 0.0))
