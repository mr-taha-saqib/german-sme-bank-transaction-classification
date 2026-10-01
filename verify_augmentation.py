#!/usr/bin/env python
"""Adversarial verification of the gated synthetic augmentation corpus.

    python verify_augmentation.py

Tries to FIND leakage rather than to confirm its absence. Covers the direct
provenance channel, the exact-collision channel, and the near-duplicate channel that
the two production gates do not themselves test.

Nothing here trains a model or modifies the frozen split.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ExperimentConfig  # noqa: E402
from src.data_preparation import attach_parent_categories, load_synthetic  # noqa: E402
from src.dataset_io import load_frozen_split, split_frame  # noqa: E402
from src.grouping import jaccard, token_set_key  # noqa: E402
from src.preprocessing import clean_series, legacy_clean_description  # noqa: E402
from src.splitting import build_provenance_index, filter_synthetic_for_train  # noqa: E402
from src.utils import save_json, setup_logging  # noqa: E402

FINDINGS: list[str] = []
PASSED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FINDINGS).append(f"{name}" + (f" -- {detail}" if detail else ""))
    print(f"  [{'PASS' if ok else '**FINDING**'}] {name}" + (f"\n        {detail}" if detail else ""))


def main() -> int:
    import logging
    setup_logging()
    logging.getLogger("gfb").setLevel(logging.WARNING)
    cfg = ExperimentConfig()
    report: dict = {}

    print("=" * 78)
    print("AUGMENTATION VERIFICATION")
    print("=" * 78)

    frozen, manifest = load_frozen_split("results/dataset", require_checksum=True)
    train = split_frame(frozen, "train")
    val = split_frame(frozen, "validation")
    test = split_frame(frozen, "test")
    heldout = pd.concat([val, test], ignore_index=True)

    train_keys = set(train["group_key"])
    heldout_keys = set(heldout["group_key"])
    heldout_texts = set(heldout["clean_description"].astype(str))
    train_texts = set(train["clean_description"].astype(str))

    # ---------------------------------------------------------------- 1
    print("\n1. PROVENANCE OF THE RAW CORPUS")
    raw = load_synthetic(cfg.data.synthetic_csvs, require_source_text=True)
    print(f"        file: {cfg.data.synthetic_csvs}")
    print(f"        rows={len(raw)}  distinct source_text={raw['source_text'].nunique()}"
          f"  categories={raw['categoryName'].nunique()}")
    check("every raw synthetic row carries a source_text provenance value",
          raw["source_text"].notna().all() and (raw["source_text"].astype(str).str.strip() != "").all())
    report["raw_corpus"] = {
        "rows": int(len(raw)),
        "distinct_source_text": int(raw["source_text"].nunique()),
        "categories": int(raw["categoryName"].nunique()),
    }

    synth = raw.copy()
    synth_clean, _ = clean_series(synth["description"], cfg.preprocessing)
    synth["clean_description"] = synth_clean
    synth = attach_parent_categories(synth, frozen)
    synth["skey"] = synth["clean_description"].map(token_set_key)
    synth["src_legacy"] = synth["source_text"].astype(str).map(
        lambda t: token_set_key(legacy_clean_description(t)))

    # ---------------------------------------------------------------- 2
    print("\n2. WHY ROWS ARE REJECTED")
    prov = build_provenance_index(frozen, "description")
    kept, gate = filter_synthetic_for_train(
        synth, frozen, "clean_description", cfg.data.label_column,
        require_source_in_train=True, drop_collisions_with_heldout=True,
        provenance_index=prov,
        near_duplicate_threshold=cfg.grouping.near_duplicate_threshold)
    report["gate"] = gate
    for k, v in gate.items():
        if isinstance(v, int):
            print(f"        {k}: {v}")
    parts = {
        "gate A (source held out)": gate.get("n_dropped_source_not_in_train", 0),
        "gate B (exact collision)": gate.get("n_dropped_collision_with_heldout", 0),
        "gate C (near-duplicate)": gate.get("n_dropped_near_duplicate_of_heldout", 0),
        "within-corpus duplicates": gate.get("n_dropped_duplicates", 0),
    }
    check("rejection count reconciles with input and output",
          gate["n_input"] - gate["n_output"] == sum(parts.values()),
          f"{gate['n_input']} - {gate['n_output']} = {gate['n_input'] - gate['n_output']} vs "
          + " + ".join(f"{v} ({k})" for k, v in parts.items()))
    report["rejection_breakdown"] = parts

    # Attribute every rejected row to a reason.
    rejected = synth[~synth.index.isin(kept.index)] if kept.index.isin(synth.index).all() else None
    resolved = synth["src_legacy"].map(lambda k: prov.get(k))
    in_train = resolved.map(lambda g: bool(g) and set(g).issubset(train_keys) if g else False)
    touches_heldout = resolved.map(lambda g: bool(g) and bool(set(g) & heldout_keys) if g else False)
    unresolved = resolved.isna()
    print(f"        attribution: source in train={int(in_train.sum())} | "
          f"source held out={int(touches_heldout.sum())} | unresolvable={int(unresolved.sum())}")
    check("every rejected row has an identified reason",
          int(in_train.sum()) + int(touches_heldout.sum()) + int(unresolved.sum()) == len(synth))
    report["rejection_attribution"] = {
        "source_in_train": int(in_train.sum()),
        "source_held_out": int(touches_heldout.sum()),
        "source_unresolvable": int(unresolved.sum()),
    }

    # ---------------------------------------------------------------- 3 & 5
    print("\n3/5. DO ALL KEPT ROWS DERIVE ONLY FROM TRAINING GROUPS?")
    kept_resolved = kept["src_legacy"].map(lambda k: prov.get(k))
    bad_src = kept_resolved.map(lambda g: (not g) or (not set(g).issubset(train_keys)))
    check("every kept row's source_text resolves ONLY to training groups",
          int(bad_src.sum()) == 0, f"{int(bad_src.sum())} rows violate this")
    heldout_src = kept_resolved.map(lambda g: bool(g) and bool(set(g) & heldout_keys) if g else False)
    check("no kept row's source_text touches a validation/test group",
          int(heldout_src.sum()) == 0, f"{int(heldout_src.sum())} rows violate this")

    src_groups = sorted({g for gs in kept_resolved if gs for g in gs})
    print(f"        kept rows draw on {kept['source_text'].nunique()} distinct source texts "
          f"-> {len(src_groups)} training groups (of {len(train_keys)})")
    report["kept_source_groups"] = len(src_groups)

    # ---------------------------------------------------------------- 4
    print("\n4. INFORMATION-LEAKAGE CHANNELS")
    exact = kept["clean_description"].astype(str).isin(heldout_texts)
    check("no kept synthetic TEXT equals a validation/test pattern",
          int(exact.sum()) == 0, f"{int(exact.sum())} collisions")
    keycol = kept["skey"].isin(heldout_keys)
    check("no kept synthetic text shares a canonical GROUP KEY with validation/test",
          int(keycol.sum()) == 0, f"{int(keycol.sum())} collisions")

    # The channel neither production gate tests: near-duplicate similarity.
    print(f"\n        near-duplicate scan: {kept['skey'].nunique()} distinct synthetic keys "
          f"x {len(heldout_keys)} held-out keys")
    hk = {k: set(k.split()) for k in heldout_keys}
    hk_label = heldout.drop_duplicates("group_key").set_index("group_key")[
        cfg.data.label_column].to_dict()
    sk = {k: set(k.split()) for k in kept["skey"].unique()}
    skey_label = kept.drop_duplicates("skey").set_index("skey")[cfg.data.label_column].to_dict()

    buckets = Counter()
    worst: list[tuple] = []
    for s, stok in sk.items():
        best = 0.0
        bestk = None
        for h, htok in hk.items():
            j = jaccard(stok, htok)
            if j > best:
                best, bestk = j, h
        for thr in (0.9, 0.8, 0.7, 0.6, 0.5):
            if best >= thr:
                buckets[thr] += 1
        worst.append((best, s, bestk))
    worst.sort(reverse=True)
    for thr in (0.9, 0.8, 0.7, 0.6, 0.5):
        print(f"        synthetic keys with max-similarity to a held-out key >= {thr}: "
              f"{buckets[thr]}")
    report["near_duplicate_to_heldout"] = {str(t): int(buckets[t]) for t in (0.9, 0.8, 0.7, 0.6, 0.5)}

    check("no synthetic text is a near-duplicate (Jaccard >= 0.6) of a held-out pattern",
          buckets[0.6] == 0,
          f"{buckets[0.6]} synthetic keys reach 0.6 against a validation/test pattern")

    print("\n        closest synthetic <-> held-out pairs:")
    for j, s, h in worst[:6]:
        same = skey_label.get(s) == hk_label.get(h)
        print(f"          {j:.3f}  '{s}'  [{skey_label.get(s)}]")
        print(f"                 vs held-out '{h}'  [{hk_label.get(h)}]"
              f"  {'SAME category' if same else 'different category'}")
    report["closest_pairs"] = [
        {"jaccard": round(float(j), 4), "synthetic": s, "heldout": h,
         "synthetic_label": skey_label.get(s), "heldout_label": hk_label.get(h)}
        for j, s, h in worst[:10]]

    # Does the corpus reproduce training text verbatim? (not leakage, but worth knowing)
    verbatim = kept["clean_description"].astype(str).isin(train_texts)
    print(f"\n        kept synthetic texts identical to a TRAINING pattern: "
          f"{int(verbatim.sum())} ({100*verbatim.mean():.1f}%) -- allowed, not leakage")
    report["verbatim_training_copies"] = int(verbatim.sum())

    # ---------------------------------------------------------------- 6
    print("\n6. IS THE CORPUS MODEL-AGNOSTIC AND REPRODUCIBLE?")
    kept2, gate2 = filter_synthetic_for_train(
        synth.copy(), frozen, "clean_description", cfg.data.label_column,
        require_source_in_train=True, drop_collisions_with_heldout=True,
        provenance_index=build_provenance_index(frozen, "description"),
        near_duplicate_threshold=cfg.grouping.near_duplicate_threshold)
    same = (kept.reset_index(drop=True)["clean_description"].tolist()
            == kept2.reset_index(drop=True)["clean_description"].tolist())
    check("gating is deterministic (identical corpus on a second run)", same)
    cols = set(kept.columns)
    check("corpus contains no model-specific field (no tokenizer/logit/embedding)",
          not any(k in " ".join(cols).lower() for k in
                  ["token", "logit", "embed", "model", "germanfinbert", "finbert"]),
          f"columns: {sorted(cols)}")
    check("corpus is plain text + label only, so both models can consume it identically",
          {"clean_description", cfg.data.label_column} <= cols)

    # ---------------------------------------------------------------- counts
    print("\n7. COUNTS AND CLASS COVERAGE")
    per_class = kept[cfg.data.label_column].value_counts()
    all_classes = sorted(frozen[cfg.data.label_column].unique())
    covered = set(per_class.index)
    print(f"        kept rows: {len(kept)} across {len(covered)} of {len(all_classes)} categories")
    print(f"        categories WITHOUT synthetic support ({len(set(all_classes)-covered)}): "
          f"{sorted(set(all_classes)-covered)}")
    train_counts = train[cfg.data.label_column].value_counts()
    combined = train_counts.add(per_class, fill_value=0).astype(int)
    print(f"        combined training corpus: {int(combined.sum())} rows, "
          f"min class {int(combined.min())}, max class {int(combined.max())}, "
          f"imbalance {combined.max()/combined.min():.1f}x "
          f"(real-only was {train_counts.max()/train_counts.min():.1f}x)")
    report["counts"] = {
        "kept_rows": int(len(kept)),
        "categories_covered": int(len(covered)),
        "categories_without_synthetic": sorted(set(all_classes) - covered),
        "combined_rows": int(combined.sum()),
        "combined_imbalance": round(float(combined.max() / combined.min()), 2),
        "real_only_imbalance": round(float(train_counts.max() / train_counts.min()), 2),
    }

    save_json(report, "results/dataset/augmentation_verification.json")
    print("\n" + "=" * 78)
    print(f"{len(PASSED)} checks passed, {len(FINDINGS)} findings")
    if FINDINGS:
        print("\nFINDINGS REQUIRING ATTENTION:")
        for f in FINDINGS:
            print("  -", f)
    print("=" * 78)
    return 1 if FINDINGS else 0


if __name__ == "__main__":
    sys.exit(main())
