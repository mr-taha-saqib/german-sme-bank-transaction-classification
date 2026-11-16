"""Retraining: data validation, leakage gate, promotion gate, promote/reject/rollback."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from src import retrain as rt
from src.config import PreprocessingConfig
from src.dataset_io import load_frozen_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic_feedback import generate  # noqa: E402

APPR = {"row_macro_f1": 0.50, "row_accuracy": 0.60, "row_pct_auto": 40.0, "row_accuracy_auto": 0.90}
RULES = {"primary_metric": "row_macro_f1", "min_improvement": 0.0, "max_accuracy_drop": 0.01,
         "max_auto_accuracy_drop": 0.02, "max_auto_coverage_drop": 5.0}


@pytest.fixture(scope="module")
def frozen():
    return load_frozen_split(ROOT / "results" / "dataset")[0]


class TestGate:
    def test_pass(self):
        g = rt.promotion_gate({**APPR, "row_macro_f1": 0.52}, APPR, RULES)
        assert g.passed

    @pytest.mark.parametrize("cand,failed", [
        ({**APPR, "row_macro_f1": 0.49}, "row_macro_f1_improves"),
        ({**APPR, "row_accuracy": 0.58}, "row_accuracy_not_dropped"),
        ({**APPR, "row_pct_auto": 30.0}, "automation_coverage_not_dropped"),
        ({**APPR, "row_accuracy_auto": 0.80}, "automatic_decision_accuracy_not_dropped"),
        ({**APPR, "row_accuracy_auto": None}, "automatic_decision_accuracy_not_dropped"),
        ({**APPR, "row_macro_f1": None}, "candidate_metrics_present"),
    ])
    def test_each_criterion_blocks(self, cand, failed):
        g = rt.promotion_gate(cand, APPR, RULES)
        assert not g.passed and failed in [c["criterion"] for c in g.criteria if not c["passed"]]


def test_feedback_validation_and_leakage_gate(frozen, mlops_cfg):
    classes = sorted(frozen["categoryName"].unique())
    fb = generate(["train_variants", "heldout_leak", "invalid"], 150, seed=3)
    kept, rep = rt.validate_feedback(fb, frozen, classes, PreprocessingConfig(), mlops_cfg)
    assert rep["dropped_unknown_label"] == 1 and rep["dropped_missing_values"] == 2 and rep["dropped_empty_text"] == 1
    assert rep["leakage_gate"]["n_dropped_collision_with_heldout"] > 0
    heldout = set(frozen.loc[frozen["split"] != "train", "clean_description"])
    assert not set(kept["clean_description"]) & heldout
    assert kept["categoryName"].isin(classes).all()


def test_missing_columns_rejected(frozen, mlops_cfg):
    with pytest.raises(rt.RetrainingError, match="missing required columns"):
        rt.validate_feedback(pd.DataFrame({"text": ["a"]}), frozen, [], PreprocessingConfig(), mlops_cfg)


def test_conflicting_labels_dropped(frozen, mlops_cfg):
    classes = sorted(frozen["categoryName"].unique())
    fb = pd.DataFrame({"description": ["Neue Lieferung Zubehör A", "Neue Lieferung Zubehör A 2026-01"],
                       "categoryName": [classes[0], classes[1]]})
    _, rep = rt.validate_feedback(fb, frozen, classes, PreprocessingConfig(), mlops_cfg)
    assert rep["dropped_conflicting_labels"] == 2


def test_insufficient_data_rejected_without_training(approved_registry, mlops_cfg):
    reg, vid = approved_registry
    rep = rt.run_retraining(mlops_cfg, generate(["invalid"], 5))
    assert rep["decision"] == "rejected_data_requirements" and reg.approved_version() == vid
    assert rep["approved_after"] == vid


def test_validate_only(approved_registry, mlops_cfg):
    reg, vid = approved_registry
    rep = rt.run_retraining(mlops_cfg, generate(["train_variants"], 200), validate_only=True)
    assert rep["decision"] == "validated_only" and reg.list_versions() == [vid]


def test_live_retraining_promotes_or_rejects_and_keeps_previous(tiny_run, approved_registry, mlops_cfg, monkeypatch):
    reg, vid = approved_registry
    used = {}
    import src.candidate_training as ct
    real = ct.train_run

    def spy(*a, **k):
        used["evaluate_test"] = k.get("evaluate_test")
        return real(*a, **k)

    monkeypatch.setattr(ct, "train_run", spy)
    rep = rt.run_retraining(mlops_cfg, generate(["train_variants", "heldout_leak"], 200),
                            base_model=str(tiny_run[0]), overrides={"epochs": 4, "max_length": 32})
    assert used["evaluate_test"] is False and rep["test_split_used"] is False
    assert rep["decision"] in ("promoted", "rejected_by_gate")
    cand = rep["candidate_version"]
    assert reg.exists(cand) and reg.exists(vid)
    if rep["promoted"]:
        assert reg.approved_version() == cand
        assert reg.rollback("test") == vid
    else:
        assert reg.approved_version() == vid and reg.metadata(cand)["status"] == "rejected"
    assert list(Path(mlops_cfg["retraining"]["report_dir"]).glob("retraining-*.md"))
    # idempotent: the same feedback against the same approved model is not retrained again
    if not rep["promoted"]:
        again = rt.run_retraining(mlops_cfg, generate(["train_variants", "heldout_leak"], 200),
                                  base_model=str(tiny_run[0]), overrides={"epochs": 4, "max_length": 32})
        assert again["candidate_version"] == cand


def test_dry_run_changes_nothing(tiny_run, approved_registry, mlops_cfg):
    reg, vid = approved_registry
    before = (reg.approved_version(), reg.list_versions())
    rep = rt.run_retraining(mlops_cfg, generate(["train_variants"], 200, seed=5), dry_run=True,
                            base_model=str(tiny_run[0]), overrides={"epochs": 1, "max_length": 32})
    assert rep["decision"].startswith("dry_run") and (reg.approved_version(), reg.list_versions()) == before


def test_no_approved_model(mlops_cfg):
    with pytest.raises(rt.RetrainingError, match="No approved model"):
        rt.run_retraining(mlops_cfg, generate(["train_variants"], 50))


def test_mlflow_tracking_of_candidate(tiny_run, approved_registry, mlops_cfg, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlops_cfg["mlflow"].update({"enabled": True, "tracking_uri": f"sqlite:///{tmp_path}/m.db",
                                "artifact_location": str(tmp_path / "art")})
    rep = rt.run_retraining(mlops_cfg, generate(["train_variants"], 200, seed=8), dry_run=True,
                            base_model=str(tiny_run[0]), overrides={"epochs": 1, "max_length": 32})
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/m.db")
    run = mlflow.get_run(rep["mlflow_run_id"])
    assert run.data.tags["stage"] == "retraining" and "val_row_accuracy" in run.data.metrics
    assert "approved_val_row_accuracy" in run.data.metrics and run.data.params["max_length"] == "32"


def test_back_to_back_runs_keep_separate_reports(approved_registry, mlops_cfg):
    from pathlib import Path as _P
    for _ in range(3):
        rt.run_retraining(mlops_cfg, generate(["train_variants"], 200), validate_only=True)
    assert len(list(_P(mlops_cfg["retraining"]["report_dir"]).glob("retraining-*.md"))) == 3


def test_rollback_cli_without_history_exits_cleanly(approved_registry, mlops_cfg, monkeypatch):
    monkeypatch.setenv("TX_REGISTRY_DIR", mlops_cfg["registry"]["dir"])
    assert rt.main(["--rollback", "--reason", "test"]) == 1
