"""Registry, artifact packaging, monitoring and settings."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.mlops_settings import ConfigError, load_mlops_config
from src.model_artifact import ArtifactError, package_run, verify_version
from src.monitoring import TransactionMonitor, drift_status, psi, text_features
from src.registry import ModelRegistry, RegistryError


def test_package_contents_and_metadata(approved_registry):
    reg, vid = approved_registry
    d = reg.version_dir(vid)
    for f in ("model", "label_mapping.json", "config.json", "model_meta.json", "validation_metrics.json",
              "category_parents.json", "train_vocabulary.json", "reference_profile.json", "metadata.json"):
        assert (d / f).exists(), f
    meta = verify_version(d)
    assert meta["n_classes"] == 37 and meta["confidence_threshold"] == 0.8
    assert meta["data"]["frozen_split_sha256"].startswith("f715f040d8a5daa6")
    assert not list((d / "model").glob("*.bin"))          # safetensors only
    parents = json.loads((d / "category_parents.json").read_text(encoding="utf-8"))
    assert len(parents) == 37


def test_package_rejects_incomplete_run(tmp_path, mlops_cfg):
    (tmp_path / "run").mkdir()
    with pytest.raises(ArtifactError, match="missing"):
        package_run(tmp_path / "run", ModelRegistry(tmp_path / "r"), mlops_cfg)


def test_packaging_is_idempotent(tiny_run, approved_registry, mlops_cfg):
    reg, vid = approved_registry
    assert package_run(tiny_run[1], reg, mlops_cfg) == vid
    assert reg.list_versions() == [vid]


def test_registry_rollback_and_rejection(tmp_path):
    reg = ModelRegistry(tmp_path)
    for v in ("a", "b"):
        src = tmp_path / f"src-{v}"
        src.mkdir()
        (src / "w").write_text(v)
        reg.register(src, v, {"content_sha256": v})
    reg.approve("a", {"reason": "first"})
    reg.approve("b", {"reason": "second"})
    assert reg.rollback("bad") == "a" and reg.exists("b")
    reg.record_rejection("b", {"reason": "gate"})
    assert reg.approved_version() == "a" and reg.metadata("b")["status"] == "rejected"
    with pytest.raises(RegistryError):
        reg.approve("zzz", {"reason": "x"})


def test_text_features_are_numeric_only():
    f = text_features("Miete Büro 2026-01", "miete büro", {"miete"})
    assert set(f) == {"char_bucket", "token_bucket", "unseen_bucket", "digit_bucket", "empty_after_cleaning"}
    assert all(isinstance(v, int) for v in f.values())


def test_psi_and_status():
    assert psi([0.5, 0.5], [10, 10]) == pytest.approx(0.0, abs=1e-12)
    assert psi([0.9, 0.1], [10, 90]) > 0.25
    assert drift_status(0.3, 10, 50, 0.1, 0.25) == "insufficient_data"
    assert drift_status(0.3, 100, 50, 0.1, 0.25) == "alert"


def _monitor(reference, n_classes=3, **cfg):
    base = {"min_requests_for_drift": 20, "min_labels_for_performance": 5}
    base.update(cfg)
    return TransactionMonitor(reference, base, "v", [f"c{i}" for i in range(n_classes)], 0.8)


def test_controlled_input_drift_detected(approved_registry):
    reg, _ = approved_registry
    ref = json.loads((reg.approved_dir() / "reference_profile.json").read_text(encoding="utf-8"))
    mon = TransactionMonitor(ref, {"min_requests_for_drift": 20}, "v", [f"c{i}" for i in range(37)], 0.8)
    long_unseen = {"char_bucket": 5, "token_bucket": 5, "unseen_bucket": 4, "digit_bucket": 0, "empty_after_cleaning": 0}
    for i in range(50):
        mon.record_prediction(f"p{i}", long_unseen, 0, 0.3, "manual_review")
    s = mon.summary()
    assert s["input_drift"]["unseen_bucket"]["status"] == "alert"
    assert s["prediction_drift"]["category"]["status"] == "alert"
    assert s["observed_performance"]["status"] == "insufficient_labels"   # drift is not performance


def test_delayed_labels_produce_performance():
    mon = _monitor({}, 2)
    feats = {"char_bucket": 1, "token_bucket": 1, "unseen_bucket": 0, "digit_bucket": 0, "empty_after_cleaning": 0}
    for i in range(6):
        mon.record_prediction(f"p{i}", feats, 0, 0.9 if i < 4 else 0.5, "automatic" if i < 4 else "manual_review")
    for i in range(6):
        assert mon.record_label(f"p{i}", "c0" if i % 2 == 0 else "c1") == "matched"
    perf = mon.summary()["observed_performance"]
    assert perf["status"] == "available" and perf["accuracy"] == pytest.approx(0.5)
    assert perf["n_labelled_automatic"] == 4
    assert mon.record_label("nope", "c0") == "unknown_prediction"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("TX_REGISTRY_DIR", "/tmp/x-reg")
    assert load_mlops_config()["registry"]["dir"] == "/tmp/x-reg"
    with pytest.raises(ConfigError):
        load_mlops_config("/nonexistent.yaml")


def test_reference_profile_matches_validation(approved_registry):
    reg, _ = approved_registry
    ref = json.loads((reg.approved_dir() / "reference_profile.json").read_text(encoding="utf-8"))
    assert ref["n"] == 228
    assert np.isclose(sum(ref["category_distribution"]), 1.0)
