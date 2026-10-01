#!/usr/bin/env python
"""STEP 5 - temperature scaling fitted on validation logits only.

    python calibrate_models.py
    python calibrate_models.py --runs-dir results/step4_comparison

For every (model, seed) run produced in Step 4:

  1. fit a single scalar temperature T by minimising NLL on the **validation**
     logits and labels saved in `logits.npz`;
  2. apply that T, unchanged, to the corresponding test logits;
  3. recompute test metrics, calibration metrics and the 0.8 confidence workflow.

The test labels are never used to fit or to select a temperature. No temperature is
chosen by comparing test performance. Each seed is calibrated independently.

Scalar temperature scaling is monotone in the logits, so it cannot reorder
predictions: accuracy, macro-F1, weighted-F1 and top-2 are mathematically identical
before and after. This script asserts that rather than assuming it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import TemperatureScaler, calibration_metrics, softmax
from src.evaluation import (
    classification_metrics,
    confidence_workflow,
    threshold_sweep,
    to_group_level,
)
from src.utils import save_json, section, setup_logging

THRESHOLD_SWEEP = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def evaluate(probs: np.ndarray, labels: np.ndarray, group_ids: np.ndarray,
             class_names: list[str], threshold: float, ece_bins: int) -> dict:
    out: dict = {}
    for level, (p, y, n_groups) in {
        "row_level": (probs, labels, None),
        "group_level": (*to_group_level(probs, labels, group_ids)[:2], None),
    }.items():
        block = classification_metrics(p, y, class_names, 2)
        conf = p.max(axis=1)
        block["confidence_statistics"] = {
            "mean": round(float(conf.mean()), 5),
            "median": round(float(np.median(conf)), 5),
            "std": round(float(conf.std()), 5),
            "min": round(float(conf.min()), 5),
            "max": round(float(conf.max()), 5),
        }
        block["calibration"] = calibration_metrics(p, y, ece_bins)
        block["confidence_workflow"] = confidence_workflow(p, y, threshold)
        block["threshold_sweep"] = threshold_sweep(p, y, THRESHOLD_SWEEP).to_dict(
            orient="records")
        block.pop("classification_report_text", None)
        out[level] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="results/step4_comparison")
    ap.add_argument("--out", default="results/step5_calibration")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--lbfgs-lr", type=float, default=0.05)
    ap.add_argument("--lbfgs-max-iter", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "calibrate.log")

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted(d for d in runs_dir.glob("*_seed*") if (d / "logits.npz").exists())
    if not run_dirs:
        log.error("No runs with logits.npz found in %s", runs_dir)
        return 2

    section("step 5 - temperature scaling (fitted on validation only)")
    log.info("Runs        : %d", len(run_dirs))
    log.info("Threshold   : %.2f", args.threshold)
    log.info("ECE bins    : %d", args.ece_bins)
    log.info("Fitting on  : VALIDATION logits and labels. Test labels are never used "
             "to fit or select T.")

    rows: list[dict] = []
    per_run: dict = {}

    for d in run_dirs:
        model, _, seed = d.name.rpartition("_seed")
        z = np.load(d / "logits.npz", allow_pickle=True)
        class_names = [str(c) for c in z["class_names"]]

        v_logits, v_labels = z["validation_logits"], z["validation_labels"]
        v_groups = z["validation_group_ids"]
        t_logits, t_labels = z["test_logits"], z["test_labels"]
        t_groups = z["test_group_ids"]

        # ---- fit on validation ONLY ------------------------------------
        scaler = TemperatureScaler().fit(v_logits, v_labels,
                                         args.lbfgs_lr, args.lbfgs_max_iter)
        T = scaler.temperature

        # sensitivity: the same fit on one averaged point per validation group
        gl, gy, _ = to_group_level(v_logits, v_labels, v_groups)
        T_groups = TemperatureScaler().fit(gl, gy, args.lbfgs_lr,
                                           args.lbfgs_max_iter).temperature

        # ---- apply UNCHANGED to test -----------------------------------
        probs_uncal = softmax(t_logits)
        probs_cal = scaler.predict_proba(t_logits)

        # Monotone transform: the argmax must not move.
        assert np.array_equal(probs_uncal.argmax(1), probs_cal.argmax(1)), (
            f"{d.name}: temperature scaling changed a prediction, which is "
            f"mathematically impossible for a scalar T")

        uncal = evaluate(probs_uncal, t_labels, t_groups, class_names,
                         args.threshold, args.ece_bins)
        cal = evaluate(probs_cal, t_labels, t_groups, class_names,
                       args.threshold, args.ece_bins)
        val_uncal = calibration_metrics(softmax(v_logits), v_labels, args.ece_bins)
        val_cal = calibration_metrics(scaler.predict_proba(v_logits), v_labels,
                                      args.ece_bins)

        log.info(
            "%-14s seed %s | T=%.4f (groups-fit %.4f) | val NLL %.4f->%.4f  "
            "val ECE %.4f->%.4f | TEST group ECE %.4f->%.4f  >=%.2f %.1f%%->%.1f%%",
            model, seed, T, T_groups,
            scaler.fit_info["nll_before"], scaler.fit_info["nll_after"],
            val_uncal["ece"], val_cal["ece"],
            uncal["group_level"]["calibration"]["ece"],
            cal["group_level"]["calibration"]["ece"], args.threshold,
            uncal["group_level"]["confidence_workflow"]["pct_auto"],
            cal["group_level"]["confidence_workflow"]["pct_auto"],
        )

        per_run[d.name] = {
            "model": model, "seed": int(seed),
            "temperature": round(T, 6),
            "temperature_fit_on_groups": round(T_groups, 6),
            "fit_info": scaler.fit_info,
            "n_validation_rows": int(len(v_labels)),
            "n_validation_groups": int(len(gy)),
            "validation_calibration_uncalibrated": val_uncal,
            "validation_calibration_calibrated": val_cal,
            "test_uncalibrated": uncal,
            "test_calibrated": cal,
        }
        save_json(per_run[d.name], out_dir / f"{d.name}_calibration.json")
        np.savez_compressed(out_dir / f"{d.name}_calibrated_probs.npz",
                            test_probabilities_calibrated=probs_cal,
                            test_labels=t_labels, test_group_ids=t_groups,
                            temperature=np.array([T]))

        for level in ("group_level", "row_level"):
            for variant, blk in (("uncalibrated", uncal), ("calibrated", cal)):
                m = blk[level]
                cw = m["confidence_workflow"]
                cb = m["calibration"]
                rows.append({
                    "Model": model, "Seed": int(seed), "Level": level.replace("_level", ""),
                    "Probabilities": variant, "T": round(T, 4), "N": m["n"],
                    "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"],
                    "Weighted F1": m["weighted_f1"], "Top-2": m["top_2_accuracy"],
                    "ECE": cb["ece"], "MCE": cb["mce"], "NLL": cb["nll"],
                    "Brier": cb["brier"], "Mean conf": cb["mean_confidence"],
                    "Max conf": m["confidence_statistics"]["max"],
                    ">=0.8 %": cw["pct_auto"], "Acc >=0.8": cw["accuracy_auto"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_seed_calibration.csv", index=False, encoding="utf-8-sig")

    # ---- mean +/- sd ----------------------------------------------------
    num = ["T", "Accuracy", "Macro F1", "Weighted F1", "Top-2", "ECE", "MCE", "NLL",
           "Brier", "Mean conf", "Max conf", ">=0.8 %", "Acc >=0.8"]
    summ_rows = []
    for (model, level, variant), g in df.groupby(["Model", "Level", "Probabilities"]):
        row = {"Model": model, "Level": level, "Probabilities": variant, "Seeds": len(g)}
        for c in num:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            row[c] = (f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
                      if len(vals) > 1 else (f"{vals.mean():.4f}" if len(vals) else "-"))
        summ_rows.append(row)
    summ = pd.DataFrame(summ_rows).sort_values(["Level", "Model", "Probabilities"])
    summ.to_csv(out_dir / "summary_calibration.csv", index=False, encoding="utf-8-sig")

    for level in ("group", "row"):
        section(f"{level}-level test results, mean +/- sd over seeds")
        sub = summ[summ.Level == level][
            ["Model", "Probabilities", "T", "Accuracy", "Macro F1", "Top-2",
             "ECE", "NLL", "Brier", "Mean conf", ">=0.8 %", "Acc >=0.8"]]
        for line in sub.to_string(index=False).splitlines():
            log.info("  %s", line)

    section("temperatures")
    for (model,), g in df[df.Level == "group"].groupby(["Model"]):
        t = g.drop_duplicates("Seed")["T"]
        log.info("  %-14s T = %.4f +/- %.4f   per seed %s",
                 model, t.mean(), t.std(ddof=1), [round(x, 4) for x in t.tolist()])

    save_json({"per_run": per_run,
               "per_seed_table": df.to_dict(orient="records"),
               "summary": summ.to_dict(orient="records")},
              out_dir / "calibration_results.json")
    section("done")
    log.info("Artefacts in %s", out_dir.resolve())
    log.info("Uncalibrated Step-4 results were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
