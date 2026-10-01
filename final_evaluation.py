#!/usr/bin/env python
"""STEP 6 - final evaluation, complete model comparison and traceability audit.

    python final_evaluation.py

Aggregates Steps 2-5 into one defensible comparison. Retrains nothing, modifies
nothing, and recomputes every reported metric from the saved per-example artefacts
so that no number in the final tables rests on a previously written summary.

Outputs
-------
    results/final_results.csv      one row per (model, seed, calibration, level)
    results/final_results.json     same plus integrity, traceability and error analysis
    results/final_comparison.md    the written comparison
    results/final_summary.xlsx     multi-sheet workbook
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import calibration_metrics, softmax
from src.dataset_io import load_frozen_split, split_frame
from src.evaluation import (
    classification_metrics,
    confidence_workflow,
    to_group_level,
)
from src.utils import save_json, section, setup_logging

RUNS_DIR = Path("results/step4_comparison")
CAL_DIR = Path("results/step5_calibration")
OUT = Path("results")
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["germanfinbert", "finbert"]
THRESHOLD = 0.80
ECE_BINS = 15

INTEGRITY: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    INTEGRITY.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def integrity_audit(frozen: pd.DataFrame) -> dict:
    section("final data integrity")
    split_manifest = json.loads(
        Path("results/dataset/split_manifest.json").read_text(encoding="utf-8"))
    aug_manifest = json.loads(
        Path("results/dataset/augmentation_manifest_v2.json").read_text(encoding="utf-8"))

    metas = {}
    for m in MODELS:
        for s in SEEDS:
            p = RUNS_DIR / f"{m}_seed{s}" / "model_meta.json"
            metas[(m, s)] = json.loads(p.read_text(encoding="utf-8"))

    splits = {v["data"]["frozen_split_sha256"] for v in metas.values()}
    check("all 10 runs used one frozen split", len(splits) == 1, str(splits))
    check("that split matches the Step-2 manifest",
          splits == {split_manifest["sha256_row_uid_group_split"]},
          f"{splits} vs {split_manifest['sha256_row_uid_group_split']}")

    corpora = {v["data"]["augmentation"].get("augmentation_corpus_sha256")
               for v in metas.values()}
    check("all 10 runs used one augmentation corpus", len(corpora) == 1, str(corpora))
    check("that corpus is v2 and matches the Step-3A manifest",
          corpora == {aug_manifest["sha256_text_label"]},
          f"{corpora} vs {aug_manifest['sha256_text_label']}")
    gates = {tuple(v["data"]["augmentation"].get("augmentation_gates_applied") or [])
             for v in metas.values()}
    check("corpus carries all three leakage gates",
          gates == {("A_provenance", "B_collision", "C_near_duplicate")}, str(gates))

    maps = set()
    for m in MODELS:
        for s in SEEDS:
            lm = json.loads((RUNS_DIR / f"{m}_seed{s}" / "label_mapping.json")
                            .read_text(encoding="utf-8"))
            maps.add(json.dumps(lm["class_names"], ensure_ascii=False))
    check("all 10 runs used one identical label mapping", len(maps) == 1,
          f"{len(maps)} distinct mappings")

    groups = set()
    for m in MODELS:
        for s in SEEDS:
            z = np.load(RUNS_DIR / f"{m}_seed{s}" / "logits.npz", allow_pickle=True)
            groups.add((tuple(z["test_group_ids"].tolist()),
                        tuple(z["test_labels"].tolist())))
    check("all 10 runs evaluated identical test groups, in identical order",
          len(groups) == 1, f"{len(groups)} distinct test orderings")

    thr = {v["confidence_threshold"] for v in metas.values()}
    check("all 10 runs used the same confidence threshold", thr == {THRESHOLD}, str(thr))

    keys = ["epochs", "learning_rate", "batch_size", "max_length", "loss",
            "focal_gamma", "use_class_weights", "warmup_mode", "warmup_ratio",
            "model_selection", "gradient_accumulation_steps", "weight_decay"]
    diffs = [k for k in keys
             if len({v["training"]["config"][k] for v in metas.values()}) != 1]
    check("identical evaluation/training protocol across models", not diffs,
          f"differs on: {diffs}")

    seeds_ok = all(metas[(m, s)]["training"]["config"]["seed"] == s
                   for m in MODELS for s in SEEDS)
    check("each run used its declared seed", seeds_ok)

    cal_leak = []
    for m in MODELS:
        for s in SEEDS:
            c = json.loads((CAL_DIR / f"{m}_seed{s}_calibration.json")
                           .read_text(encoding="utf-8"))
            if c["fit_info"]["n_calibration_rows"] != c["n_validation_rows"]:
                cal_leak.append((m, s))
    n_val = int((frozen["split"] == "validation").sum())
    n_test = int((frozen["split"] == "test").sum())
    check("temperature was fitted on validation rows only, never on test",
          not cal_leak and n_val != n_test,
          f"validation={n_val} rows, test={n_test} rows; offenders={cal_leak}")

    return {
        "frozen_split_sha256": split_manifest["sha256_row_uid_group_split"],
        "augmentation_v2_sha256": aug_manifest["sha256_text_label"],
        "augmentation_gates": aug_manifest["gates_applied"],
        "n_train_rows": split_manifest["splits"]["train"]["n_rows"],
        "n_validation_rows": n_val,
        "n_validation_groups": split_manifest["splits"]["validation"]["n_groups"],
        "n_test_rows": n_test,
        "n_test_groups": split_manifest["splits"]["test"]["n_groups"],
        "n_classes": len(json.loads(list(maps)[0])),
        "seeds": SEEDS,
        "threshold": THRESHOLD,
    }


# --------------------------------------------------------------------------- #
def recompute_all(class_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Recompute every metric from raw logits. Nothing is read from a summary."""
    section("recomputing all metrics from saved logits (traceability)")
    rows = []
    trace = {"matches": 0, "mismatches": []}

    for m in MODELS:
        for s in SEEDS:
            d = RUNS_DIR / f"{m}_seed{s}"
            z = np.load(d / "logits.npz", allow_pickle=True)
            t_logits, t_labels = z["test_logits"], z["test_labels"]
            t_groups = z["test_group_ids"]

            cal = json.loads((CAL_DIR / f"{m}_seed{s}_calibration.json")
                             .read_text(encoding="utf-8"))
            T = cal["temperature"]

            for variant, probs in (("uncalibrated", softmax(t_logits)),
                                   ("calibrated", softmax(t_logits / T))):
                for level in ("group", "row"):
                    if level == "group":
                        p, y, uniq = to_group_level(probs, t_labels, t_groups)
                    else:
                        p, y, uniq = probs, t_labels, np.unique(t_groups)
                    cm = classification_metrics(p, y, class_names, 2)
                    cb = calibration_metrics(p, y, ECE_BINS)
                    cw = confidence_workflow(p, y, THRESHOLD)
                    conf = p.max(axis=1)
                    rows.append({
                        "Model": m, "Seed": s, "Calibration": variant, "Level": level,
                        "Temperature": round(T, 6), "N": cm["n"],
                        "Accuracy": cm["accuracy"], "Macro F1": cm["macro_f1"],
                        "Weighted F1": cm["weighted_f1"], "Top-2": cm["top_2_accuracy"],
                        "ECE": cb["ece"], "MCE": cb["mce"], "NLL": cb["nll"],
                        "Brier": cb["brier"],
                        "Mean conf": round(float(conf.mean()), 5),
                        "Max conf": round(float(conf.max()), 5),
                        "Auto coverage %": cw["pct_auto"],
                        "Manual review %": cw["pct_manual_review"],
                        "Acc >=0.8": cw["accuracy_auto"],
                        "Errors auto-accepted": cw["n_errors_in_auto"],
                    })

                    # cross-check against what Step 4 / Step 5 recorded
                    src = (json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
                           if variant == "uncalibrated" else cal["test_calibrated"])
                    ref = src[f"{level}_level"]
                    for k, got in (("accuracy", cm["accuracy"]),
                                   ("macro_f1", cm["macro_f1"]),
                                   ("top_2_accuracy", cm["top_2_accuracy"])):
                        if abs(ref[k] - got) < 1e-6:
                            trace["matches"] += 1
                        else:
                            trace["mismatches"].append(
                                f"{m} s{s} {variant} {level} {k}: {ref[k]} vs {got}")

    df = pd.DataFrame(rows)
    check(f"every recomputed metric matches its saved value "
          f"({trace['matches']} comparisons)", not trace["mismatches"],
          "; ".join(trace["mismatches"][:5]))
    return df, trace


# --------------------------------------------------------------------------- #
def error_analysis(frozen: pd.DataFrame, class_names: list[str]) -> dict:
    section("error analysis")
    test = split_frame(frozen, "test")
    first = test.drop_duplicates("group_id").set_index("group_id")
    parent = frozen.drop_duplicates("categoryName").set_index("categoryName")[
        "parentCategoryName"].to_dict()

    per_group: dict[tuple[str, int], np.ndarray] = {}
    gids = None
    for m in MODELS:
        for s in SEEDS:
            z = np.load(RUNS_DIR / f"{m}_seed{s}" / "logits.npz", allow_pickle=True)
            p, y, uniq = to_group_level(softmax(z["test_logits"]), z["test_labels"],
                                        z["test_group_ids"])
            gids = uniq
            per_group[(m, s)] = p.argmax(1)
            per_group[("_truth", 0)] = y

    truth = per_group[("_truth", 0)]
    recs = []
    for i, gid in enumerate(gids):
        row = {"group_id": int(gid),
               "text": str(first.loc[gid, "clean_description"]),
               "true_category": class_names[truth[i]],
               "true_parent": parent.get(class_names[truth[i]]),
               "n_test_rows": int((test["group_id"] == gid).sum())}
        for m in MODELS:
            correct = [int(per_group[(m, s)][i] == truth[i]) for s in SEEDS]
            preds = [class_names[per_group[(m, s)][i]] for s in SEEDS]
            row[f"{m}_n_correct"] = sum(correct)
            row[f"{m}_modal_prediction"] = max(set(preds), key=preds.count)
        recs.append(row)
    g = pd.DataFrame(recs)

    g["always_wrong_both"] = (g.germanfinbert_n_correct == 0) & (g.finbert_n_correct == 0)
    g["always_right_both"] = (g.germanfinbert_n_correct == 5) & (g.finbert_n_correct == 5)
    g["german_better"] = g.germanfinbert_n_correct > g.finbert_n_correct
    g["finbert_better"] = g.finbert_n_correct > g.germanfinbert_n_correct
    g.to_csv(OUT / "final_error_analysis.csv", index=False, encoding="utf-8-sig")

    confusions = []
    for r in g[g.germanfinbert_n_correct < 5].itertuples():
        confusions.append({
            "text": r.text, "true": r.true_category,
            "predicted": r.germanfinbert_modal_prediction,
            "seeds_correct": r.germanfinbert_n_correct,
            "true_parent": r.true_parent,
            "pred_parent": parent.get(r.germanfinbert_modal_prediction),
            "same_parent": parent.get(r.true_category) == parent.get(
                r.germanfinbert_modal_prediction),
        })

    n_train = frozen[frozen.split == "train"].groupby("categoryName").size().to_dict()
    for c in confusions:
        c["train_rows_for_true_class"] = int(n_train.get(c["true"], 0))

    out = {
        "n_test_groups": int(len(g)),
        "always_wrong_for_both_models": g[g.always_wrong_both][
            ["text", "true_category"]].to_dict(orient="records"),
        "always_right_for_both_models": int(g.always_right_both.sum()),
        "germanfinbert_better_on_n_groups": int(g.german_better.sum()),
        "finbert_better_on_n_groups": int(g.finbert_better.sum()),
        "tied_on_n_groups": int((~g.german_better & ~g.finbert_better).sum()),
        "germanfinbert_confusions": sorted(confusions, key=lambda c: c["seeds_correct"]),
        "cross_parent_error_rate": round(
            float(np.mean([not c["same_parent"] for c in confusions])), 4) if confusions else None,
    }
    print(f"  test groups: {out['n_test_groups']}")
    print(f"  both models always correct : {out['always_right_for_both_models']}")
    print(f"  both models always wrong   : {len(out['always_wrong_for_both_models'])}")
    print(f"  GermanFinBERT better on    : {out['germanfinbert_better_on_n_groups']} groups")
    print(f"  FinBERT better on          : {out['finbert_better_on_n_groups']} groups")
    print(f"  tied                       : {out['tied_on_n_groups']} groups")
    print(f"  errors crossing a parent category: {out['cross_parent_error_rate']}")
    return out


# --------------------------------------------------------------------------- #
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    num = ["Accuracy", "Macro F1", "Weighted F1", "Top-2", "ECE", "NLL", "Brier",
           "Mean conf", "Auto coverage %", "Manual review %", "Acc >=0.8", "Temperature"]
    out = []
    for (m, c, lv), gg in df.groupby(["Model", "Calibration", "Level"]):
        row = {"Model": m, "Calibration": c, "Level": lv, "Seeds": len(gg)}
        for k in num:
            v = pd.to_numeric(gg[k], errors="coerce").dropna()
            row[k] = (f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1
                      else (f"{v.mean():.4f}" if len(v) else "n/a"))
        out.append(row)
    return pd.DataFrame(out).sort_values(["Level", "Model", "Calibration"])


def main() -> int:
    OUT.mkdir(exist_ok=True)
    log = setup_logging(OUT / "final_evaluation.log")
    section("step 6 - final evaluation")

    frozen, _ = load_frozen_split("results/dataset", require_checksum=True)
    class_names = json.loads(
        (RUNS_DIR / "germanfinbert_seed42" / "label_mapping.json")
        .read_text(encoding="utf-8"))["class_names"]

    integrity = integrity_audit(frozen)
    df, trace = recompute_all(class_names)
    summ = summarise(df)
    errors = error_analysis(frozen, class_names)

    df.to_csv(OUT / "final_results.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(OUT / "final_summary.csv", index=False, encoding="utf-8-sig")

    for level in ("group", "row"):
        section(f"{level}-level final comparison (mean +/- sd over 5 seeds)")
        cols = ["Model", "Calibration", "Accuracy", "Macro F1", "Weighted F1", "Top-2",
                "ECE", "NLL", "Auto coverage %", "Acc >=0.8", "Manual review %"]
        for line in summ[summ.Level == level][cols].to_string(index=False).splitlines():
            log.info("  %s", line)

    save_json({
        "integrity": integrity,
        "integrity_checks": [{"check": n, "passed": bool(o), "detail": d}
                             for n, o, d in INTEGRITY],
        "all_checks_passed": all(o for _, o, _ in INTEGRITY),
        "traceability": {"comparisons": trace["matches"],
                         "mismatches": trace["mismatches"]},
        "per_seed": df.to_dict(orient="records"),
        "summary": summ.to_dict(orient="records"),
        "error_analysis": errors,
    }, OUT / "final_results.json")

    try:
        with pd.ExcelWriter(OUT / "final_summary.xlsx", engine="openpyxl") as w:
            summ.to_excel(w, sheet_name="Summary", index=False)
            df.to_excel(w, sheet_name="PerSeed", index=False)
            pd.DataFrame([{"check": n, "passed": o, "detail": d}
                          for n, o, d in INTEGRITY]).to_excel(
                w, sheet_name="Integrity", index=False)
            pd.DataFrame(errors["germanfinbert_confusions"]).to_excel(
                w, sheet_name="Errors", index=False)
            pd.DataFrame([integrity]).T.to_excel(w, sheet_name="Provenance")
        log.info("Wrote %s", OUT / "final_summary.xlsx")
    except Exception as exc:
        log.warning("XLSX export failed: %s", exc)

    section("integrity summary")
    n_ok = sum(1 for _, o, _ in INTEGRITY if o)
    log.info("%d/%d integrity checks passed", n_ok, len(INTEGRITY))
    log.info("%d metric recomputations matched their saved values", trace["matches"])
    return 0 if n_ok == len(INTEGRITY) and not trace["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
