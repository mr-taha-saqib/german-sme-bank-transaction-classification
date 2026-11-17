"""Controlled retraining with an explicit promotion gate.

    python -m src.retrain --feedback new_labelled.csv --validate-only   # data checks only
    python -m src.retrain --feedback new_labelled.csv --dry-run         # train + evaluate, change nothing
    python -m src.retrain --feedback new_labelled.csv                   # train, gate, promote if passed
    python -m src.retrain --rollback --reason "..."                     # re-approve the previous model

Interface: a CSV of NEW, human-confirmed labelled transactions with columns
``description,categoryName`` (UTF-8). Labels must belong to the existing 37-class
taxonomy. The API does not store descriptions, so this file comes from the
client's own bookkeeping (e.g. confirmed manual-review decisions).

Workflow
--------
 1. Load the APPROVED version; verify the frozen split checksum matches it.
 2. Validate rows: required columns, missing/empty/over-long text, label taxonomy,
    duplicates, empty after the research preprocessing.
 3. Leakage protection: drop rows that equal or near-duplicate (token-Jaccard
    >= 0.60, the research grouping threshold) a VALIDATION or TEST pattern, using
    the research gates B and C.
 4. Check minimum rows and class coverage.
 5. Train a candidate with the approved run's configuration on the frozen train
    split + corpus v2 + the accepted rows.
 6. Evaluate the candidate on the frozen VALIDATION split (test is never read) and
    compare with the approved model's validation metrics (same split, same code).
 7. Apply the promotion criteria (configs/mlops.yaml ``retraining.promotion``).
 8. Register the candidate, record the MLflow run and the decision.
 9. Promote only if every criterion passes (and not --dry-run). The previous model
    stays in the registry for rollback.
10. Write a JSON + Markdown report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig, PreprocessingConfig
from .dataset_io import load_frozen_split
from .mlops_settings import get, load_mlops_config, resolve_path
from .model_artifact import git_commit, package_run
from .preprocessing import clean_description
from .registry import ModelRegistry, RegistryError
from .splitting import filter_synthetic_for_train

logger = logging.getLogger("retrain")


class RetrainingError(RuntimeError):
    pass


@dataclass
class GateResult:
    passed: bool
    criteria: list[dict[str, Any]] = field(default_factory=list)


def headline(block: dict) -> dict[str, float | None]:
    """Row-level validation metrics used by the gate (+ group level for information)."""
    row, grp = block.get("row_level", {}), block.get("group_level", {})
    cw = row.get("confidence_workflow", {})
    return {"row_accuracy": row.get("accuracy"), "row_macro_f1": row.get("macro_f1"),
            "row_weighted_f1": row.get("weighted_f1"), "row_pct_auto": cw.get("pct_auto"),
            "row_accuracy_auto": cw.get("accuracy_auto"),
            "group_accuracy": grp.get("accuracy"), "group_macro_f1": grp.get("macro_f1"),
            "n_validation_rows": row.get("n"), "n_validation_groups": grp.get("n_independent_groups")}


def promotion_gate(cand: dict[str, Any], appr: dict[str, Any], rules: dict[str, Any]) -> GateResult:
    """Pure function over VALIDATION metrics; the test split is not an input."""
    primary = str(rules.get("primary_metric", "row_macro_f1"))
    crit: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        crit.append({"criterion": name, "passed": bool(ok), "detail": detail})

    def num(x: Any) -> bool:
        return isinstance(x, (int, float))

    add("candidate_metrics_present", all(num(cand.get(k)) for k in (primary, "row_accuracy", "row_pct_auto")),
        "candidate validation metrics were computed")
    imp = float(rules.get("min_improvement", 0.0))
    add(f"{primary}_improves", num(cand.get(primary)) and cand[primary] >= appr[primary] + imp,
        f"candidate {cand.get(primary)} >= approved {appr.get(primary)} + {imp}")
    tol = float(rules.get("max_accuracy_drop", 0.01))
    add("row_accuracy_not_dropped", num(cand.get("row_accuracy")) and cand["row_accuracy"] >= appr["row_accuracy"] - tol,
        f"candidate {cand.get('row_accuracy')} >= approved {appr.get('row_accuracy')} - {tol}")
    cov_tol = float(rules.get("max_auto_coverage_drop", 5.0))
    add("automation_coverage_not_dropped",
        num(cand.get("row_pct_auto")) and cand["row_pct_auto"] >= appr["row_pct_auto"] - cov_tol,
        f"candidate {cand.get('row_pct_auto')}% >= approved {appr.get('row_pct_auto')}% - {cov_tol} pp")
    a_tol = float(rules.get("max_auto_accuracy_drop", 0.02))
    if num(appr.get("row_accuracy_auto")):
        ok = num(cand.get("row_accuracy_auto")) and cand["row_accuracy_auto"] >= appr["row_accuracy_auto"] - a_tol
        add("automatic_decision_accuracy_not_dropped", ok,
            f"candidate {cand.get('row_accuracy_auto')} >= approved {appr.get('row_accuracy_auto')} - {a_tol}")
    return GateResult(passed=all(c["passed"] for c in crit), criteria=crit)


def validate_feedback(df: pd.DataFrame, frozen: pd.DataFrame, class_names: list[str],
                      pre_cfg: PreprocessingConfig, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Schema/quality checks + research leakage gates. Returns (accepted rows, report)."""
    rcfg = cfg.get("retraining", {})
    rep: dict[str, Any] = {"n_input_rows": int(len(df))}
    missing = [c for c in ("description", "categoryName") if c not in df.columns]
    if missing:
        raise RetrainingError(f"feedback is missing required columns {missing}")
    out = df[["description", "categoryName"]].copy()
    bad_missing = out.isna().any(axis=1) | ~out["description"].map(lambda x: isinstance(x, str))
    out = out[~bad_missing]
    out["description"] = out["description"].astype(str)
    out["categoryName"] = out["categoryName"].astype(str).str.strip()
    empty = out["description"].str.strip() == ""
    too_long = out["description"].str.len() > int(rcfg.get("max_text_length", 512))
    unknown_label = ~out["categoryName"].isin(class_names)
    rep.update({"dropped_missing_values": int(bad_missing.sum()), "dropped_empty_text": int(empty.sum()),
                "dropped_too_long": int(too_long.sum()), "dropped_unknown_label": int(unknown_label.sum()),
                "unknown_labels_seen": sorted(out.loc[unknown_label, "categoryName"].unique().tolist())[:20]})
    out = out[~(empty | too_long | unknown_label)].copy()
    out["clean_description"] = [clean_description(t, pre_cfg) for t in out["description"]]
    empty_clean = out["clean_description"].isin(["", pre_cfg.empty_placeholder])
    rep["dropped_empty_after_cleaning"] = int(empty_clean.sum())
    out = out[~empty_clean]
    dup = out.duplicated(["clean_description", "categoryName"])
    rep["dropped_duplicates"] = int(dup.sum())
    out = out[~dup]
    conflicting = out.groupby("clean_description")["categoryName"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    rep["dropped_conflicting_labels"] = int(out["clean_description"].isin(conflicting).sum())
    out = out[~out["clean_description"].isin(conflicting)]
    # Leakage gates B (exact/group-key collision) and C (near-duplicate) from the research code.
    kept, gate = filter_synthetic_for_train(
        out, frozen, text_col="clean_description", label_col="categoryName",
        require_source_in_train=False, drop_collisions_with_heldout=True,
        near_duplicate_threshold=float(rcfg.get("near_duplicate_threshold", 0.60)))
    rep["leakage_gate"] = {k: v for k, v in gate.items() if k != "near_duplicate_examples_dropped"}
    rep["n_accepted_rows"] = int(len(kept))
    rep["n_classes_covered"] = int(kept["categoryName"].nunique())
    rep["classes_covered"] = sorted(kept["categoryName"].unique().tolist())
    rep["class_counts"] = {str(k): int(v) for k, v in kept["categoryName"].value_counts().items()}
    problems = []
    if rep["n_accepted_rows"] < int(rcfg.get("min_new_rows", 30)):
        problems.append(f"{rep['n_accepted_rows']} accepted rows < minimum {rcfg.get('min_new_rows', 30)}")
    if rep["n_classes_covered"] < int(rcfg.get("min_classes", 3)):
        problems.append(f"{rep['n_classes_covered']} classes covered < minimum {rcfg.get('min_classes', 3)}")
    rep["requirements_passed"] = not problems
    rep["requirement_failures"] = problems
    return kept.reset_index(drop=True), rep


def run_retraining(cfg: dict[str, Any], feedback: pd.DataFrame, dry_run: bool = False, validate_only: bool = False,
                   base_model: str | None = None, overrides: dict[str, Any] | None = None,
                   dataset_dir: str | Path = "results/dataset", force: bool = False) -> dict[str, Any]:
    from .tracking import Tracker, log_training_run

    reg = ModelRegistry(resolve_path(get(cfg, "registry.dir", "model_registry")))
    try:
        approved_id = reg.approved_version()
        appr_dir = reg.approved_dir()
    except RegistryError as exc:
        raise RetrainingError(f"No approved model to compare against: {exc}") from exc
    appr_meta = reg.metadata(approved_id)
    report_dir = resolve_path(get(cfg, "retraining.report_dir", "reports/retraining"))
    # microseconds: two runs in the same second must not overwrite each other's report
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dataset_dir = resolve_path(dataset_dir)
    frozen, manifest = load_frozen_split(dataset_dir, require_checksum=True)
    split_sha = manifest.get("sha256_row_uid_group_split")
    if appr_meta.get("data", {}).get("frozen_split_sha256") not in (None, split_sha):
        raise RetrainingError("Frozen split checksum differs from the one the approved model was trained on")
    exp_cfg = ExperimentConfig.load(appr_dir / "config.json")
    class_names = json.loads((appr_dir / "label_mapping.json").read_text(encoding="utf-8"))["class_names"]
    accepted, data_rep = validate_feedback(feedback, frozen, class_names, exp_cfg.preprocessing, cfg)
    overrides = dict(overrides or {})
    fb_sha = hashlib.sha256(accepted[["clean_description", "categoryName"]].sort_values(
        ["clean_description", "categoryName"]).to_csv(index=False).encode()).hexdigest()
    key = hashlib.sha256(f"{approved_id}|{fb_sha}|{json.dumps(overrides, sort_keys=True)}|{base_model}".encode()).hexdigest()[:12]
    report: dict[str, Any] = {"timestamp": stamp, "approved_version": approved_id, "dry_run": dry_run,
                              "validate_only": validate_only, "feedback_sha256": fb_sha, "candidate_key": key,
                              "frozen_split_sha256": split_sha, "data": data_rep, "overrides": overrides,
                              "test_split_used": False, "git_commit": git_commit()}
    marker = report_dir / f"done-{key}.json"
    if marker.exists() and not (force or dry_run or validate_only):
        logger.info("This feedback was already processed against %s; returning the earlier report", approved_id)
        return json.loads(marker.read_text(encoding="utf-8"))
    if validate_only or not data_rep["requirements_passed"]:
        report["decision"] = "validated_only" if (validate_only and data_rep["requirements_passed"]) \
            else "rejected_data_requirements"
        report["promoted"] = False
        report["approved_after"] = reg.approved_version()
        _write_reports(report, report_dir, stamp)
        return report

    for k, v in overrides.items():
        if not hasattr(exp_cfg.training, k):
            raise RetrainingError(f"unknown training override {k}")
        setattr(exp_cfg.training, k, v)
    base = base_model or str(get(cfg, "model.base_local_dir") or "") or appr_meta.get("base_checkpoint") \
        or get(cfg, "model.base_checkpoint")
    aug_version = appr_meta.get("data", {}).get("augmentation_version")
    cand_dir = resolve_path("artifacts/candidates") / f"candidate-{stamp}"
    from .candidate_training import train_run
    tracker = Tracker(cfg)
    with tracker.run(run_name=f"retrain-{stamp}", tags={"stage": "retraining", "dry_run": str(dry_run),
                                                       "approved_version": approved_id, "feedback_sha256": fb_sha}):
        result = train_run(base, cand_dir, exp_cfg, dataset_dir, aug_version, extra_rows=accepted,
                            evaluate_test=False,
                            short_name=json.loads((appr_dir / "model_meta.json").read_text(encoding="utf-8"))
                            .get("model", {}).get("short_name", "candidate"))
        cand = headline(result["validation"])
        appr = headline(json.loads((appr_dir / "validation_metrics.json").read_text(encoding="utf-8")))
        gate = promotion_gate(cand, appr, get(cfg, "retraining.promotion", {}))
        log_training_run(tracker, cand_dir)
        tracker.log_params({"n_feedback_rows_accepted": len(accepted), "approved_version": approved_id,
                            "base_model": base, "overrides": json.dumps(overrides)})
        tracker.log_metrics({f"approved_val_{k}": v for k, v in appr.items() if isinstance(v, (int, float))})
        tracker.log_metrics({"gate_passed": float(gate.passed)})
        report.update({"base_model": base, "candidate": {"validation": cand, "run_dir": str(cand_dir),
                                                         "seconds": round(result["seconds"], 1)},
                       "approved": {"validation": appr}, "gate": asdict(gate)})
        if dry_run:
            report["decision"] = "dry_run_gate_passed" if gate.passed else "dry_run_gate_failed"
        else:
            vid = package_run(cand_dir, reg, cfg, dataset_dir, version_prefix="candidate", extra_metadata={
                "parent_version": approved_id, "feedback_sha256": fb_sha, "n_feedback_rows": len(accepted),
                "training_overrides": overrides, "promotion_gate": asdict(gate), "mlflow_run_id": tracker.run_id,
                "metrics_note": "candidate: test split not evaluated during retraining"})
            tracker.set_tags({"registry_version": vid})
            mv = tracker.register_model(vid)
            decision = {"reason": "promotion gate passed" if gate.passed else "promotion gate failed",
                        "type": "automated_gate", "criteria": gate.criteria, "candidate_validation": cand,
                        "approved_validation": appr, "mlflow_run_id": tracker.run_id}
            if gate.passed:
                reg.approve(vid, decision)
                tracker.set_alias("champion", mv)
                report["decision"] = "promoted"
            else:
                reg.record_rejection(vid, decision)
                report["decision"] = "rejected_by_gate"
            report["candidate_version"] = vid
            report["mlflow_model_version"] = mv
        tracker.set_tags({"decision": report["decision"]})
    # The registry keeps its own copy of the weights; drop the scratch copy (~420 MB).
    shutil.rmtree(cand_dir / "model", ignore_errors=True)
    report["promoted"] = report["decision"] == "promoted"
    report["mlflow_run_id"] = tracker.run_id
    report["approved_after"] = reg.approved_version()
    _write_reports(report, report_dir, stamp)
    if not dry_run:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Retraining decision: %s (approved now: %s)", report["decision"], report["approved_after"])
    return report


def _write_reports(report: dict[str, Any], report_dir: Path, stamp: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"retraining-{stamp}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    d = report["data"]
    lines = [f"# Retraining report {stamp}", "",
             f"* Approved model at start: `{report['approved_version']}`",
             f"* Mode: {'validate-only' if report.get('validate_only') else 'dry-run' if report['dry_run'] else 'live'}",
             f"* Decision: **{report.get('decision')}**",
             f"* Candidate version: `{report.get('candidate_version', 'n/a')}`",
             f"* Approved model after run: `{report.get('approved_after')}`",
             f"* MLflow run: `{report.get('mlflow_run_id')}`",
             f"* Training overrides: `{report.get('overrides') or 'none'}`",
             "* Test split used: **no**", "", "## Feedback data", "",
             f"* Input rows: {d['n_input_rows']} | accepted: {d['n_accepted_rows']} | classes covered: {d['n_classes_covered']}",
             f"* Dropped: missing {d['dropped_missing_values']}, empty {d['dropped_empty_text']}, too long "
             f"{d['dropped_too_long']}, unknown label {d['dropped_unknown_label']}, empty after cleaning "
             f"{d['dropped_empty_after_cleaning']}, duplicates {d['dropped_duplicates']}, conflicting labels "
             f"{d['dropped_conflicting_labels']}",
             f"* Leakage gate: exact/group-key collisions with validation/test "
             f"{d['leakage_gate'].get('n_dropped_collision_with_heldout', 0)}, near-duplicates (Jaccard >= "
             f"{d['leakage_gate'].get('near_duplicate_threshold')}) {d['leakage_gate'].get('n_dropped_near_duplicate_of_heldout', 0)}",
             f"* Requirement failures: {d['requirement_failures'] or 'none'}", ""]
    if "gate" in report:
        lines += ["## Validation comparison (frozen validation split)", "", "| metric | approved | candidate |", "|---|---|---|"]
        for k, v in report["candidate"]["validation"].items():
            lines.append(f"| {k} | {report['approved']['validation'].get(k)} | {v} |")
        lines += ["", "## Promotion criteria", "", "| criterion | passed | detail |", "|---|---|---|"]
        lines += [f"| {c['criterion']} | {c['passed']} | {c['detail']} |" for c in report["gate"]["criteria"]]
    (report_dir / f"retraining-{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from .logging_utils import setup_logging
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feedback", help="CSV with description,categoryName")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base-model", help="base checkpoint (HF id or local dir); default: approved model's base")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-length", type=int, help="tokenizer length; safe to lower only if no text is truncated")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--reason", default="manual rollback")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    setup_logging("INFO", fmt="text")
    cfg = load_mlops_config()
    if args.no_mlflow:
        cfg["mlflow"]["enabled"] = False
    if args.rollback:
        try:
            target = ModelRegistry(resolve_path(get(cfg, "registry.dir"))).rollback(args.reason)
        except RegistryError as exc:
            logger.error("Rollback not possible: %s", exc)
            return 1
        from .tracking import Tracker
        Tracker(cfg).mirror_alias("champion", target)    # keep the MLflow alias in step with the registry
        print(json.dumps({"rolled_back_to": target}))
        return 0
    if not args.feedback or not Path(args.feedback).exists():
        ap.error("--feedback must point to an existing CSV")
    overrides = {k: v for k, v in (("epochs", args.epochs), ("max_length", args.max_length)) if v is not None}
    try:
        rep = run_retraining(cfg, pd.read_csv(args.feedback, encoding="utf-8-sig"), dry_run=args.dry_run,
                             validate_only=args.validate_only, base_model=args.base_model, overrides=overrides,
                             force=args.force)
    except RetrainingError as exc:
        logger.error("Retraining failed: %s", exc)
        return 1
    print(json.dumps({k: rep.get(k) for k in ("decision", "promoted", "approved_version", "candidate_version",
                                               "approved_after", "mlflow_run_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
