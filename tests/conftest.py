"""Shared fixtures for the MLOps tests.

A TINY BERT (1 layer, hidden 32) is built offline from the training vocabulary and
trained for a few epochs on the frozen TRAIN split. It exercises the real inference,
packaging, API, monitoring and retraining code paths without downloading the
420 MB GermanFinBERT checkpoint. It is a test fixture, not the production model.

Nothing here writes to results/, model_registry/, mlflow.db or monitoring/.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TX_API_NO_AUTOCREATE", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def tiny_experiment_config():
    from src.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.training.max_length = 32
    cfg.training.epochs = 4
    cfg.training.batch_size = 32
    cfg.training.learning_rate = 5e-3
    cfg.evaluation.make_plots = False
    return cfg


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """(base_model_dir, run_dir) for a tiny model trained once per test session."""
    from src.candidate_training import build_tiny_base_model, train_run
    root = tmp_path_factory.mktemp("tiny")
    base = build_tiny_base_model(root / "base", ROOT / "results" / "dataset")
    train_run(str(base), root / "run", tiny_experiment_config(), ROOT / "results" / "dataset",
              augmentation_version=None, short_name="tinybert")
    return base, root / "run"


@pytest.fixture
def mlops_cfg(tmp_path):
    from src.mlops_settings import load_mlops_config
    cfg = copy.deepcopy(load_mlops_config(apply_env=False))
    cfg["registry"]["dir"] = str(tmp_path / "registry")
    cfg["monitoring"]["log_dir"] = str(tmp_path / "monitoring")
    cfg["monitoring"]["min_requests_for_drift"] = 20
    cfg["monitoring"]["min_labels_for_performance"] = 5
    cfg["retraining"]["report_dir"] = str(tmp_path / "reports")
    cfg["mlflow"]["enabled"] = False
    return cfg


@pytest.fixture
def approved_registry(tiny_run, mlops_cfg):
    from src.model_artifact import package_run
    from src.registry import ModelRegistry
    reg = ModelRegistry(mlops_cfg["registry"]["dir"])
    vid = package_run(tiny_run[1], reg, mlops_cfg, ROOT / "results" / "dataset")
    reg.approve(vid, {"reason": "test fixture"})
    return reg, vid
