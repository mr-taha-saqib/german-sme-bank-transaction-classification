#!/usr/bin/env python
"""STEP 2 - dataset preparation, preprocessing, grouping and leakage-safe splitting.

Runs the data pipeline end to end and **freezes** the resulting split to disk, so
that every downstream model in Step 3 trains and evaluates on byte-identical data.

    python prepare_dataset.py
    python prepare_dataset.py --grouping-mode token_set --seed 43
    python prepare_dataset.py --out results/dataset_alt

No model is downloaded, loaded or trained by this script.

Stages
------
  1  inspect      structure, missingness, labels, identifiers, duplicates
  2  deduplicate  remove records the source system flags as duplicates
  3  preprocess   German-aware conservative cleaning (T0-T13)
  4  group        label-free canonical key + near-duplicate merge
  5  split        group-disjoint train / validation / test
  6  validate     the five required leakage and distribution checks
  7  freeze       write the canonical split artefact + checksum
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig
from src.data_preparation import (
    dataset_summary,
    load_raw_transactions,
    remove_duplicate_records,
)
from src.grouping import (
    assign_groups,
    grouping_report,
    near_duplicate_census,
    token_set_key,
)
from src.preprocessing import clean_series, sample_transformations
from src.splitting import group_based_split, validate_split
from src.utils import (
    environment_report,
    get_logger,
    save_json,
    section,
    set_seed,
    setup_logging,
)

SPLIT_COLUMNS = [
    "row_uid", "description", "clean_description", "group_key", "group_id",
    "categoryName", "parentCategoryName", "split",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/dataset", help="output directory")
    p.add_argument("--config", help="ExperimentConfig JSON to start from")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grouping-mode", choices=["exact", "token_set", "near_dup"], default=None)
    p.add_argument("--near-dup-threshold", type=float, default=None)
    p.add_argument("--keep-duplicates", action="store_true",
                   help="do NOT remove source-flagged duplicate records")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def inspect_source_workbook(path: str, out_dir: Path) -> dict:
    """Stage 1: document the actual structure of the primary workbook."""
    log = get_logger()
    raw = pd.read_excel(path, dtype=str)
    if raw.shape[1] == 1:
        header = list(raw.columns)[0].split(";")
        raw = raw.iloc[:, 0].astype(str).str.split(";", expand=True)
        raw.columns = header

    rows = []
    for c in raw.columns:
        s = raw[c].astype(str).str.strip()
        nonempty = int((s != "").sum())
        rows.append({
            "column": c,
            "non_empty": nonempty,
            "missing": int(len(raw) - nonempty),
            "missing_pct": round(100 * (len(raw) - nonempty) / max(len(raw), 1), 2),
            "n_unique": int(s[s != ""].nunique()),
            "example": (s[s != ""].iloc[0][:60] if nonempty else ""),
        })
    census = pd.DataFrame(rows).sort_values("n_unique", ascending=False)
    census.to_csv(out_dir / "column_census.csv", index=False, encoding="utf-8-sig")

    informative = census[census.n_unique > 1]
    constant = census[census.n_unique <= 1]["column"].tolist()

    # Identifiers that could link related transactions -- candidate leakage units.
    id_cols = [c for c in ["vendorName", "vendorGroupId", "vendorGroupName", "vendorId",
                           "counterpartIban", "counterpartyNameRaw", "duplicateGroupId",
                           "fingerprint", "bankAccountId", "counterpartCreditorId",
                           "counterpartMandateReference"] if c in raw.columns]
    identifiers = {}
    for c in id_cols:
        s = raw[c].astype(str).str.strip()
        filled = s[s != ""]
        identifiers[c] = {
            "n_unique": int(filled.nunique()),
            "coverage_pct": round(100 * len(filled) / max(len(raw), 1), 2),
            "categories_per_value_max": (
                int(raw[s != ""].groupby(c)["categoryName"].nunique().max())
                if len(filled) and "categoryName" in raw.columns else None
            ),
            "values_per_category_min": (
                int(raw[s != ""].groupby("categoryName")[c].nunique().min())
                if len(filled) and "categoryName" in raw.columns else None
            ),
        }

    rep = {
        "path": str(path),
        "n_rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "n_informative_columns": int(len(informative)),
        "constant_columns": constant,
        "columns_with_missing_values": informative[informative.missing > 0][
            ["column", "missing", "missing_pct"]].to_dict(orient="records"),
        "candidate_related_transaction_identifiers": identifiers,
    }
    log.info("Workbook: %d rows x %d columns | %d informative, %d constant",
             rep["n_rows"], rep["n_columns"], rep["n_informative_columns"], len(constant))
    log.info("Columns with missing values: %d",
             len(rep["columns_with_missing_values"]))
    return rep


def freeze_split(df: pd.DataFrame, out_dir: Path) -> dict:
    """Stage 7: write the canonical split artefact plus a checksum.

    Step 3 loads this file rather than recomputing the split, which is what
    guarantees that GermanFinBERT and FinBERT see byte-identical data.
    """
    log = get_logger()
    frozen = df.copy()
    # Stable row identity that survives reordering.
    frozen["row_uid"] = (
        frozen["id"] if "id" in frozen.columns
        else pd.Series(range(len(frozen))).astype(str)
    )
    frozen = frozen[[c for c in SPLIT_COLUMNS if c in frozen.columns]]
    frozen = frozen.sort_values(["group_id", "row_uid"]).reset_index(drop=True)

    path = out_dir / "split_assignment.csv"
    frozen.to_csv(path, index=False, encoding="utf-8-sig")

    # Fixed CRLF so the recorded checksums (computed on Windows) match on every OS.
    payload = frozen[["row_uid", "group_id", "split"]].to_csv(index=False, lineterminator="\r\n")
    checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest = {
        "split_file": path.name,
        "n_rows": int(len(frozen)),
        "n_groups": int(frozen["group_id"].nunique()),
        "splits": {
            s: {"n_rows": int((frozen["split"] == s).sum()),
                "n_groups": int(frozen.loc[frozen["split"] == s, "group_id"].nunique())}
            for s in sorted(frozen["split"].unique())
        },
        "group_to_split": {
            str(g): str(s) for g, s in
            frozen.drop_duplicates("group_id").set_index("group_id")["split"].items()
        },
        "sha256_row_uid_group_split": checksum,
        "usage": (
            "Step 3 must load this file and assert the checksum matches, so that both "
            "models are trained and evaluated on identical data."
        ),
    }
    save_json(manifest, out_dir / "split_manifest.json")
    log.info("Frozen split written to %s (sha256 %s)", path, checksum[:16])
    return manifest


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.seed is not None:
        cfg.split.seed = args.seed
    if args.grouping_mode:
        cfg.grouping.mode = args.grouping_mode
    if args.near_dup_threshold is not None:
        cfg.grouping.near_duplicate_threshold = args.near_dup_threshold
    if args.keep_duplicates:
        cfg.data.drop_source_flagged_duplicates = False
        cfg.data.drop_fingerprint_duplicates = False

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "prepare_dataset.log")

    section("step 2 - dataset preparation, preprocessing, grouping, splitting")
    log.info("Output       : %s", out_dir.resolve())
    log.info("Seed         : %d", cfg.split.seed)
    log.info("Grouping     : %s (near-dup threshold %.2f)",
             cfg.grouping.mode, cfg.grouping.near_duplicate_threshold)
    log.info("Dedup        : source-flagged=%s fingerprint=%s",
             cfg.data.drop_source_flagged_duplicates, cfg.data.drop_fingerprint_duplicates)
    set_seed(cfg.split.seed)

    summary: dict = {
        "config": cfg.to_dict(),
        "environment": environment_report(),
    }

    # -- 1. inspect --------------------------------------------------------
    section("stage 1  inspect the primary dataset")
    summary["source_inspection"] = inspect_source_workbook(cfg.data.raw_xlsx, out_dir)

    real = load_raw_transactions(cfg.data.raw_xlsx, cfg.data)
    summary["raw_label_summary"] = dataset_summary(real)
    log.info("Classes: %d | largest %s | smallest %s | imbalance %.1fx",
             summary["raw_label_summary"]["n_classes"],
             summary["raw_label_summary"]["largest_class"],
             summary["raw_label_summary"]["smallest_class"],
             summary["raw_label_summary"]["imbalance_ratio"])

    dup_raw = int(real["description"].duplicated().sum())
    log.info("Exact duplicate raw descriptions: %d of %d rows (%d distinct texts)",
             dup_raw, len(real), real["description"].nunique())
    summary["raw_duplicate_descriptions"] = {
        "n_rows": int(len(real)),
        "n_distinct_raw_descriptions": int(real["description"].nunique()),
        "n_rows_whose_raw_description_repeats": dup_raw,
    }

    # -- 2. deduplicate ----------------------------------------------------
    section("stage 2  remove source-flagged duplicate records")
    real, dup_report = remove_duplicate_records(real, cfg.data)
    summary["deduplication"] = dup_report
    log.info("Rows after de-duplication: %d (removed %d)",
             dup_report["n_output"], dup_report["n_removed_total"])

    # -- 3. preprocess -----------------------------------------------------
    section("stage 3  preprocessing")
    cleaned, pre_audit = clean_series(real[cfg.data.text_column], cfg.preprocessing)
    real["clean_description"] = cleaned
    summary["preprocessing"] = pre_audit
    sample_transformations(real[cfg.data.text_column], cfg.preprocessing, 40).to_csv(
        out_dir / "preprocessing_examples.csv", index=False, encoding="utf-8-sig")
    log.info("Cleaning: %d distinct raw -> %d distinct cleaned (collapse %.1fx); "
             "%d empty -> placeholder; vocab %d -> %d",
             pre_audit["unique_raw"], pre_audit["unique_cleaned"],
             pre_audit["collapse_ratio"], pre_audit["empty_after_clean"],
             pre_audit["vocab_raw"], pre_audit["vocab_cleaned"])

    if cfg.data.drop_empty_after_clean:
        n0 = len(real)
        real = real[real["clean_description"] != cfg.preprocessing.empty_placeholder]
        real = real.reset_index(drop=True)
        if len(real) != n0:
            log.warning("Dropped %d rows that cleaned to the empty placeholder", n0 - len(real))

    # -- 4. group ----------------------------------------------------------
    section("stage 4  grouping")
    keys = sorted(real["clean_description"].map(token_set_key).unique())
    key_label = (real.assign(_k=real["clean_description"].map(token_set_key))
                     .drop_duplicates("_k").set_index("_k")[cfg.data.label_column].to_dict())
    census = near_duplicate_census(keys, cfg.grouping.census_thresholds, key_label)
    summary["near_duplicate_census"] = census
    save_json(census, out_dir / "near_duplicate_census.json")
    log.info("Near-duplicate census over %d canonical keys:", len(keys))
    for row in census:
        log.info("   Jaccard >= %.2f : %3d pairs (%3d same-category, %d cross-category)",
                 row["threshold"], row["n_pairs"], row["n_same_label"], row["n_cross_label"])

    real = assign_groups(real, "clean_description", cfg.data.label_column, cfg.grouping)
    grp = grouping_report(real, cfg.data.label_column)
    summary["grouping"] = grp
    log.info("%s", grp["effective_sample_size_note"])
    log.info("Groups/category: median %d | %d categories with 1 group | %d with <3 | %d with <4",
             grp["median_groups_per_category"], grp["categories_with_1_group"],
             grp["categories_with_lt3_groups"], grp["categories_with_lt4_groups"])
    if grp["n_label_impure_groups"]:
        log.warning("%d groups span more than one category (see grouping report)",
                    grp["n_label_impure_groups"])
    else:
        log.info("All groups are label-pure (verified after grouping, not used to group)")

    # -- 5. split ----------------------------------------------------------
    section("stage 5  leakage-safe group-based split")
    real = group_based_split(real, cfg.data.label_column, cfg.split)

    # -- 6. validate -------------------------------------------------------
    section("stage 6  validation checks")
    split_report = validate_split(
        real, "clean_description", cfg.data.label_column,
        cfg.grouping.audit_jaccard_threshold, strict=True,
    )
    summary["split_validation"] = split_report

    log.info("CHECK 1  no group in more than one split      : %s",
             "PASS" if all(v == 0 for v in split_report["checks"]["group_overlap"].values()) else "FAIL")
    log.info("CHECK 2  no exact duplicate text across splits: %s",
             "PASS" if all(v == 0 for v in split_report["checks"]["exact_text_overlap"].values()) else "FAIL")
    nd_same = sum(v["n_same_label"] for v in split_report["checks"]["near_duplicate"].values())
    nd_cross = sum(v["n_cross_label"] for v in split_report["checks"]["near_duplicate"].values())
    log.info("CHECK 3  near-duplicate leakage (Jaccard>=%.2f): %d same-category pairs, "
             "%d cross-category pairs", cfg.grouping.audit_jaccard_threshold, nd_same, nd_cross)

    dist_rows = []
    for s in ("train", "validation", "test"):
        st = split_report["split_stats"].get(s)
        if not st:
            continue
        for cat, n in sorted(st["rows_per_class"].items()):
            dist_rows.append({"split": s, "categoryName": cat, "n_rows": n,
                              "n_groups": st["groups_per_class"].get(cat, 0)})
    dist = pd.DataFrame(dist_rows)
    wide = dist.pivot_table(index="categoryName", columns="split", values="n_rows",
                            fill_value=0, aggfunc="sum")
    wideg = dist.pivot_table(index="categoryName", columns="split", values="n_groups",
                             fill_value=0, aggfunc="sum")
    wide.columns = [f"rows_{c}" for c in wide.columns]
    wideg.columns = [f"groups_{c}" for c in wideg.columns]
    class_dist = wide.join(wideg).sort_values("rows_train", ascending=False)
    class_dist.to_csv(out_dir / "class_distribution.csv", encoding="utf-8-sig")
    log.info("CHECK 4  class distributions written to class_distribution.csv (%d categories)",
             len(class_dist))

    log.info("CHECK 5  rows and groups per split:")
    for s in ("train", "validation", "test"):
        st = split_report["split_stats"].get(s)
        if st:
            log.info("           %-11s %5d rows | %3d groups | %2d categories | %3d unique texts",
                     s, st["n_rows"], st["n_groups"], st["n_classes"], st["n_unique_texts"])

    if split_report.get("classes_absent_from_test"):
        log.warning("%d categories cannot appear in test (too few groups): %s",
                    len(split_report["classes_absent_from_test"]),
                    split_report["classes_absent_from_test"])
    if split_report.get("test_classes_with_single_group"):
        log.warning("%d test categories hold a SINGLE group -- their recall can only be "
                    "0.0 or 1.0", len(split_report["test_classes_with_single_group"]))

    # -- 7. freeze ---------------------------------------------------------
    section("stage 7  freeze the split")
    summary["frozen_split"] = freeze_split(real, out_dir)

    save_json(summary, out_dir / "dataset_summary.json")
    save_json(grp, out_dir / "grouping_report.json")
    save_json(split_report, out_dir / "split_report.json")
    save_json(pre_audit, out_dir / "preprocessing_audit.json")
    cfg.save(out_dir / "config.json")

    section("step 2 complete")
    log.info("Artefacts in %s", out_dir.resolve())
    log.info("Leakage validation passed: %s", split_report["passed"])
    log.info("No model was trained. Step 3 should load %s/split_assignment.csv",
             out_dir)
    return 0 if split_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
