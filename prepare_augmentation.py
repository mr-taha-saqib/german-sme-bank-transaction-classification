#!/usr/bin/env python
"""STEP 3A - build and freeze the leakage-corrected synthetic augmentation corpus.

    python prepare_augmentation.py
    python prepare_augmentation.py --version v2 --out results/dataset

Applies three independent gates to the shipped LLM corpus and writes the survivors as
a versioned, checksummed artefact. Training then LOADS that artefact instead of
rebuilding it, so the guarantee that GermanFinBERT and FinBERT saw the same corpus is
auditable rather than merely reliable.

Gates
-----
A  provenance     -- the row's `source_text` must resolve only to TRAINING groups
B  collision      -- the row's text must not equal, nor share a canonical group key
                     with, any validation/test pattern
C  near-duplicate -- the row's text must not reach the project's grouping threshold
                     (Jaccard >= 0.60) against any validation/test pattern

Gate C is the Step-3A correction. Gates A and B test only exact equality, while Step-2
grouping declares two texts to be the SAME leakage unit at Jaccard >= 0.60; without
gate C a synthetic row can enter training even though grouping would have placed it
inside a held-out group.

The original `data/final_synthetic_clean_7033.csv` is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig
from src.data_preparation import attach_parent_categories, load_synthetic
from src.dataset_io import load_frozen_split, split_frame
from src.grouping import jaccard, token_set_key
from src.preprocessing import clean_series
from src.splitting import build_provenance_index, filter_synthetic_for_train
from src.utils import save_json, section, set_seed, setup_logging

CORPUS_COLUMNS = ["clean_description", "categoryName", "parentCategoryName",
                  "source_text", "source_file"]


def corpus_checksum(df: pd.DataFrame) -> str:
    # Fixed CRLF so the recorded checksums (computed on Windows) match on every OS.
    payload = df[["clean_description", "categoryName"]].to_csv(index=False, lineterminator="\r\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="results/dataset")
    ap.add_argument("--out", default="results/dataset")
    ap.add_argument("--version", default="v2",
                    help="v1 = gates A+B only (the pre-correction corpus); "
                         "v2 = gates A+B+C (the Step-3A correction)")
    ap.add_argument("--config", help="ExperimentConfig JSON")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "prepare_augmentation.log")
    set_seed(cfg.split.seed)

    apply_gate_c = args.version != "v1"
    thr = cfg.grouping.near_duplicate_threshold

    section(f"step 3a - build augmentation corpus {args.version}")
    log.info("Source corpus     : %s", cfg.data.synthetic_csvs)
    log.info("Gates             : A provenance, B collision%s",
             f", C near-duplicate (Jaccard >= {thr:.2f})" if apply_gate_c else " (gate C DISABLED)")
    log.info("Grouping threshold: %.2f  (same definition as Step 2 -- not a new knob)", thr)

    # -- inputs ------------------------------------------------------------
    frozen, split_manifest = load_frozen_split(args.dataset_dir, require_checksum=True)
    train = split_frame(frozen, "train")
    heldout = frozen[frozen["split"] != "train"]

    raw = load_synthetic(cfg.data.synthetic_csvs, require_source_text=True)
    if not len(raw):
        log.error("No synthetic corpus could be loaded.")
        return 2

    cleaned, _ = clean_series(raw["description"], cfg.preprocessing)
    raw["clean_description"] = cleaned
    raw = attach_parent_categories(raw, frozen)
    known = set(frozen[cfg.data.label_column].unique())
    n_before_label = len(raw)
    raw = raw[raw[cfg.data.label_column].isin(known)]
    if len(raw) != n_before_label:
        log.warning("Dropped %d rows with an unknown category", n_before_label - len(raw))

    # -- gates -------------------------------------------------------------
    section("gates")
    kept, gate = filter_synthetic_for_train(
        raw, frozen, "clean_description", cfg.data.label_column,
        require_source_in_train=True,
        drop_collisions_with_heldout=True,
        provenance_index=build_provenance_index(frozen, cfg.data.text_column),
        near_duplicate_threshold=thr if apply_gate_c else None,
    )

    # -- independent post-condition ---------------------------------------
    section("post-condition: zero rows may violate the Jaccard rule")
    heldout_keys = {k: set(k.split()) for k in heldout["group_key"].unique()}
    inverted: dict[str, list[str]] = {}
    for k, toks in heldout_keys.items():
        for t in toks:
            inverted.setdefault(t, []).append(k)

    def max_sim(text: str) -> float:
        toks = set(token_set_key(str(text)).split())
        cands = {c for t in toks for c in inverted.get(t, ())}
        return max((jaccard(toks, heldout_keys[c]) for c in cands), default=0.0)

    sims = kept["clean_description"].astype(str).map(max_sim)
    n_violating = int((sims >= thr).sum())
    log.info("Max similarity of any kept row to a held-out pattern: %.4f", float(sims.max()))
    log.info("Rows violating Jaccard >= %.2f: %d", thr, n_violating)
    if apply_gate_c and n_violating:
        log.error("POST-CONDITION FAILED: %d rows still violate the rule", n_violating)
        return 1

    # -- freeze ------------------------------------------------------------
    section("freeze")
    corpus = kept[[c for c in CORPUS_COLUMNS if c in kept.columns]].copy()
    corpus = corpus.sort_values(["categoryName", "clean_description"]).reset_index(drop=True)
    corpus_path = out_dir / f"augmentation_corpus_{args.version}.csv"
    corpus.to_csv(corpus_path, index=False, encoding="utf-8-sig")
    checksum = corpus_checksum(corpus)

    per_class = corpus[cfg.data.label_column].value_counts()
    train_counts = train[cfg.data.label_column].value_counts()
    combined = train_counts.add(per_class, fill_value=0).astype(int)
    all_classes = sorted(frozen[cfg.data.label_column].unique())

    manifest = {
        "version": args.version,
        "corpus_file": corpus_path.name,
        "sha256_text_label": checksum,
        "source_files": cfg.data.synthetic_csvs,
        "frozen_split_sha256": split_manifest.get("sha256_row_uid_group_split"),
        "gates_applied": ["A_provenance", "B_collision"] + (["C_near_duplicate"] if apply_gate_c else []),
        "near_duplicate_threshold": float(thr) if apply_gate_c else None,
        "gate_report": gate,
        "max_similarity_to_heldout": round(float(sims.max()), 4),
        "n_rows_violating_jaccard_rule": n_violating,
        "counts": {
            "raw_rows": int(gate["n_input"]),
            "kept_rows": int(len(corpus)),
            "categories_covered": int(per_class.size),
            "categories_total": len(all_classes),
            "categories_without_synthetic": sorted(set(all_classes) - set(per_class.index)),
            "n_source_patterns_used": int(corpus["source_text"].nunique())
            if "source_text" in corpus.columns else None,
        },
        "class_distribution_synthetic": {str(k): int(v) for k, v in per_class.items()},
        "class_distribution_combined": {str(k): int(v) for k, v in combined.items()},
        "combined_rows": int(combined.sum()),
        "combined_imbalance": round(float(combined.max() / combined.min()), 2),
        "usage": ("Training must LOAD this file and assert the checksum, so that every "
                  "model in the comparison provably sees the same augmentation corpus."),
    }
    save_json(manifest, out_dir / f"augmentation_manifest_{args.version}.json")

    log.info("Corpus written to %s", corpus_path)
    log.info("sha256 %s", checksum[:24])
    log.info("Rows: %d raw -> %d kept (%d removed)",
             gate["n_input"], len(corpus), gate["n_input"] - len(corpus))
    log.info("  gate A (source held out)   : %d", gate.get("n_dropped_source_not_in_train", 0))
    log.info("  gate B (exact collision)   : %d", gate.get("n_dropped_collision_with_heldout", 0))
    log.info("  gate C (near-duplicate)    : %d", gate.get("n_dropped_near_duplicate_of_heldout", 0))
    log.info("  duplicates within corpus   : %d", gate.get("n_dropped_duplicates", 0))
    log.info("Categories covered: %d of %d", per_class.size, len(all_classes))
    log.info("Combined training corpus: %d rows, imbalance %.1fx",
             manifest["combined_rows"], manifest["combined_imbalance"])
    section("done")
    log.info("Original %s was NOT modified.", cfg.data.synthetic_csvs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
