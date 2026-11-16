"""
tracking.py
===========
Thin MLflow wrapper used by the training pipeline and the retraining workflow.

* Default backend: a local SQLite file (``sqlite:///mlflow.db``) plus a local
  artifact folder; no server and no credentials. ``MLFLOW_TRACKING_URI`` points it
  at a remote tracking server instead.
* If tracking is disabled in config, or MLflow is not installed, every call is a
  no-op and a WARNING is logged once. Nothing fails silently.
* Model registration in the MLflow Model Registry is attempted only for
  database-backed URIs (sqlite/postgres/mysql/http). The serving decision itself
  lives in ``registry.py``; MLflow aliases mirror it for visibility.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .mlops_settings import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


def git_commit() -> str:
    """Current git commit (or ``$GIT_COMMIT``, or 'unknown')."""
    env = os.environ.get("GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _normalise_uri(uri: str) -> str:
    """Make relative sqlite / file URIs relative to the project root."""
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        rel = uri[len("sqlite:///"):]
        if not Path(rel).is_absolute():
            return "sqlite:///" + str(resolve_path(rel))
    if "://" not in uri:
        return resolve_path(uri).as_uri()
    return uri


class Tracker:
    """Experiment tracker. Use :meth:`run` as a context manager."""

    def __init__(self, cfg: dict[str, Any]):
        mcfg = cfg.get("mlflow", {})
        self.enabled = bool(mcfg.get("enabled", True))
        self.experiment = str(mcfg.get("experiment", "experiment"))
        self.register_models = bool(mcfg.get("register_models", False))
        self.model_name = str(cfg.get("registry", {}).get("model_name", self.experiment))
        self.log_weights = bool(mcfg.get("log_weights", False))
        self.uri = _normalise_uri(str(mcfg.get("tracking_uri", "sqlite:///mlflow.db")))
        self.artifact_location = resolve_path(mcfg.get("artifact_location", "mlartifacts"))
        self._mlflow = None
        self.run_id: str | None = None
        if not self.enabled:
            logger.warning("MLflow tracking is DISABLED by configuration; runs will not be recorded.")
            return
        try:
            import mlflow  # noqa: PLC0415
        except ImportError:
            logger.warning("MLflow is not installed; runs will not be recorded. "
                           "Install requirements.txt to enable tracking.")
            self.enabled = False
            return
        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        if mlflow.get_experiment_by_name(self.experiment) is None:
            loc = None
            if self.uri.startswith("sqlite:"):
                self.artifact_location.mkdir(parents=True, exist_ok=True)
                loc = self.artifact_location.as_uri()
            mlflow.create_experiment(self.experiment, artifact_location=loc)
        mlflow.set_experiment(self.experiment)
        logger.info("MLflow tracking -> %s (experiment %s)", self.uri, self.experiment)

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[Tracker]:
        if not self.enabled:
            yield self
            return
        mlflow = self._mlflow
        all_tags = {"git_commit": git_commit()}
        all_tags.update(tags or {})
        with mlflow.start_run(run_name=run_name, tags=all_tags) as active:
            self.run_id = active.info.run_id
            yield self
        logger.info("MLflow run %s finished (%s)", self.run_id, run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, directory: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(directory).exists():
            self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def register_model(self, version_tag: str) -> str | None:
        """Register this run's ``model`` artifact in the MLflow Model Registry.

        Returns the MLflow model version, or None when registration is disabled
        or not supported by the backend (a WARNING explains why).
        """
        if not (self.enabled and self.register_models and self.run_id):
            return None
        if self.uri.startswith("file:"):
            logger.warning("MLflow Model Registry needs a database-backed tracking URI; "
                           "skipping registration for %s", self.uri)
            return None
        # MLflow 3's register_model() requires an MLflow "logged model"; our artifact
        # is a plain folder (numpy arrays + JSON), so the version is created directly
        # from the run's artifact URI.
        client = self._mlflow.MlflowClient()
        try:
            client.get_registered_model(self.model_name)
        except self._mlflow.exceptions.MlflowException:
            client.create_registered_model(self.model_name, description="German SME transaction classifier")
        # Weights (~420 MB) are logged to MLflow only when mlflow.log_weights is true;
        # otherwise the version points at the run's metadata and the weights stay in
        # the file registry under the same registry_version id.
        sub = "model" if self.log_weights else "run"
        source = f"{self._mlflow.get_run(self.run_id).info.artifact_uri}/{sub}"
        mv = client.create_model_version(self.model_name, source=source, run_id=self.run_id,
                                         tags={"registry_version": version_tag,
                                               "weights_in_mlflow": str(self.log_weights).lower()})
        logger.info("MLflow registry: %s version %s registered", self.model_name, mv.version)
        return str(mv.version)

    def set_alias(self, alias: str, mlflow_version: str | None) -> None:
        """Point an MLflow alias (e.g. 'champion') at a registered model version."""
        if not (self.enabled and mlflow_version):
            return
        client = self._mlflow.MlflowClient()
        client.set_registered_model_alias(self.model_name, alias, mlflow_version)
        logger.info("MLflow registry: alias '%s' -> %s version %s", alias,
                    self.model_name, mlflow_version)


def flatten_metrics(block: dict, prefix: str) -> dict[str, float]:
    """Numeric headline metrics from a train_model.py metrics block (group + row level)."""
    out: dict[str, float] = {}
    for level in ("group_level", "row_level"):
        m = block.get(level, {})
        tag = "group" if level == "group_level" else "row"
        for k in ("accuracy", "macro_f1", "weighted_f1", "top_2_accuracy"):
            if isinstance(m.get(k), (int, float)):
                out[f"{prefix}{tag}_{k}"] = float(m[k])
        cw = m.get("confidence_workflow", {})
        for k in ("pct_auto", "accuracy_auto", "pct_manual_review"):
            if isinstance(cw.get(k), (int, float)):
                out[f"{prefix}{tag}_{k}"] = float(cw[k])
        cs = m.get("confidence_statistics", {})
        if isinstance(cs.get("mean"), (int, float)):
            out[f"{prefix}{tag}_mean_confidence"] = float(cs["mean"])
    return out


def log_training_run(tracker: Tracker, run_dir: Path, version_id: str | None = None) -> None:
    """Log a train_model.py / candidate run directory to the active MLflow run."""
    import json
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
    tr = cfg.get("training", {})
    data = meta.get("data", {})
    aug = data.get("augmentation", {}) or {}
    tracker.log_params({
        "base_checkpoint": meta.get("model", {}).get("hf_name"), "seed": tr.get("seed"),
        "epochs": tr.get("epochs"), "learning_rate": tr.get("learning_rate"), "batch_size": tr.get("batch_size"),
        "max_length": tr.get("max_length"), "loss": tr.get("loss"), "focal_gamma": tr.get("focal_gamma"),
        "model_selection": tr.get("model_selection"), "use_amp": tr.get("use_amp"),
        "confidence_threshold": meta.get("confidence_threshold"),
        "augmentation_version": aug.get("augmentation_version"), "n_train_rows": data.get("n_train_rows"),
        "device": meta.get("device")})
    tracker.set_tags({"frozen_split_sha256": data.get("frozen_split_sha256"),
                      "augmentation_corpus_sha256": aug.get("augmentation_corpus_sha256"),
                      "registry_version": version_id or "unregistered"})
    for name, prefix in (("validation_metrics.json", "val_"), ("test_metrics.json", "test_")):
        p = run_dir / name
        if p.exists():
            tracker.log_metrics(flatten_metrics(json.loads(p.read_text(encoding="utf-8")), prefix))
    for epoch in meta.get("training", {}).get("history", []):
        tracker.log_metrics({k: v for k, v in epoch.items() if isinstance(v, (int, float)) and k != "epoch"},
                            step=int(epoch.get("epoch", 0)))
    for f in ("config.json", "model_meta.json", "validation_metrics.json", "test_metrics.json",
              "label_mapping.json", "test_predictions.csv", "confusion_matrix_test.csv",
              "classification_report_test.txt", "train.log"):
        tracker.log_artifact(run_dir / f, artifact_path="run")
    if tracker.log_weights:
        tracker.log_artifacts(run_dir / "model", artifact_path="model")
