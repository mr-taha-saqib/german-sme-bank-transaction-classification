"""Configuration for the MLOps layer (API, registry, MLflow, monitoring, retraining).

Loaded from ``configs/mlops.yaml`` (or ``$TX_CONFIG``) with a few documented
environment overrides. Paths are resolved against the repository root.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mlops.yaml"

_ENV: dict[str, tuple[str, str, Any]] = {
    "TX_REGISTRY_DIR": ("registry", "dir", str),
    "TX_MONITORING_DIR": ("monitoring", "log_dir", str),
    "TX_RETRAINING_REPORT_DIR": ("retraining", "report_dir", str),
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri", str),
    "TX_MLFLOW_ARTIFACTS": ("mlflow", "artifact_location", str),
    "TX_MLFLOW_ENABLED": ("mlflow", "enabled", lambda s: s.strip().lower() in {"1", "true", "yes"}),
    "TX_MAX_BATCH_SIZE": ("api", "max_batch_size", int),
}


class ConfigError(ValueError):
    pass


def load_mlops_config(path: str | Path | None = None, apply_env: bool = True) -> dict[str, Any]:
    p = Path(path or os.environ.get("TX_CONFIG", DEFAULT_CONFIG))
    if not p.exists():
        raise ConfigError(f"MLOps config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping")
    cfg = copy.deepcopy(data)
    if apply_env:
        for env, (sec, key, cast) in _ENV.items():
            raw = os.environ.get(env)
            if raw:
                try:
                    cfg.setdefault(sec, {})[key] = cast(raw)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"Invalid {env}={raw!r}: {exc}") from exc
    thr = float(cfg.get("model", {}).get("confidence_threshold", 0.8))
    if not 0.0 < thr < 1.0:
        raise ConfigError("model.confidence_threshold must be in (0, 1)")
    return cfg


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
