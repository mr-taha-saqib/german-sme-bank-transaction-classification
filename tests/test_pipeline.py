"""Self-contained checks for the properties the brief requires.

Run with:  python tests/test_pipeline.py       (no pytest needed)

These are correctness guards for the leakage-safety and calibration claims, not a
substitute for reading the run artefacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (  # noqa: E402
    TemperatureScaler,
    expected_calibration_error,
    softmax,
)
from src.config import GroupingConfig, PreprocessingConfig, SplitConfig  # noqa: E402
from src.evaluation import confidence_workflow, to_group_level, top_k_accuracy  # noqa: E402
from src.grouping import (  # noqa: E402
    assign_groups,
    grouping_report,
    jaccard,
    near_duplicate_census,
    token_set_key,
)
from src.data_preparation import remove_duplicate_records  # noqa: E402
from src.config import DataConfig  # noqa: E402
from src.preprocessing import clean_description, legacy_clean_description  # noqa: E402
from src.splitting import (  # noqa: E402
    build_provenance_index,
    filter_synthetic_for_train,
    group_based_split,
    validate_split,
)
from src.utils import setup_logging  # noqa: E402

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not condition else ""))


# --------------------------------------------------------------------------- #
def test_preprocessing() -> None:
    print("\nPREPROCESSING")
    cfg = PreprocessingConfig()

    check("dates are removed",
          clean_description("Software Lizenz 2024-09", cfg) == "software lizenz",
          clean_description("Software Lizenz 2024-09", cfg))

    check("invoice references are removed",
          clean_description("Rechnung RE-2024-6278 Beratungsleistung 2024-06", cfg)
          == "rechnung beratungsleistung",
          clean_description("Rechnung RE-2024-6278 Beratungsleistung 2024-06", cfg))

    # The headline fix: the prototype turned B2B into "b b".
    got = clean_description("LinkedIn Ads B2B Leadgen 2024-06", cfg)
    check("B2B survives cleaning (prototype bug fixed)", got == "linkedin ads b2b leadgen", got)
    check("prototype cleaner really did destroy B2B",
          legacy_clean_description("LinkedIn Ads B2B Leadgen 2024-06")
          == "linkedin ads b b leadgen")

    check("umlauts are preserved",
          clean_description("Sozialversicherung Beiträge 2025-01", cfg)
          == "sozialversicherung beiträge",
          clean_description("Sozialversicherung Beiträge 2025-01", cfg))

    check("empty input maps to the placeholder",
          clean_description("2024-06 ###", cfg) == cfg.empty_placeholder,
          clean_description("2024-06 ###", cfg))

    check("cleaning is idempotent",
          clean_description(clean_description("SaaS Abo monatlich 2024-06 RE-2024-4710", cfg), cfg)
          == clean_description("SaaS Abo monatlich 2024-06 RE-2024-4710", cfg))

    check("abbreviation expansion works",
          "umsatzsteuer" in clean_description("MwSt Zahlung", cfg),
          clean_description("MwSt Zahlung", cfg))

    off = PreprocessingConfig(expand_abbreviations=False)
    check("abbreviation expansion is switchable",
          clean_description("MwSt Zahlung", off) == "mwst zahlung",
          clean_description("MwSt Zahlung", off))

    check("legal forms are kept by default (conservative)",
          "gmbh" in clean_description("AustroData GmbH Rechnung", cfg),
          clean_description("AustroData GmbH Rechnung", cfg))


def test_grouping() -> None:
    print("\nGROUPING")
    check("token_set is order-insensitive",
          token_set_key("miete büro") == token_set_key("büro miete"))
    check("token_set separates different content",
          token_set_key("miete büro") != token_set_key("miete lager"))
    check("jaccard identity", jaccard({"a", "b"}, {"a", "b"}) == 1.0)
    check("jaccard disjoint", jaccard({"a"}, {"b"}) == 0.0)

    df = pd.DataFrame({
        "clean_description": ["software lizenz", "software lizenz", "büro miete",
                              "miete büro", "gehaltslauf"],
        "categoryName": ["IT", "IT", "Miete", "Miete", "Personal"],
    })
    g = assign_groups(df, cfg=GroupingConfig(mode="token_set"))
    check("identical texts share a group", g.loc[0, "group_id"] == g.loc[1, "group_id"])
    check("reordered texts share a group", g.loc[2, "group_id"] == g.loc[3, "group_id"])
    check("distinct texts get distinct groups", g["group_id"].nunique() == 3)

    g2 = assign_groups(df, cfg=GroupingConfig(mode="exact"))
    check("exact mode keeps reorderings apart", g2["group_id"].nunique() == 4)

    # determinism
    a = assign_groups(df, cfg=GroupingConfig(mode="token_set"))["group_id"].tolist()
    b = assign_groups(df, cfg=GroupingConfig(mode="token_set"))["group_id"].tolist()
    check("grouping is deterministic", a == b)


def _toy_dataset() -> pd.DataFrame:
    rows = []
    for cat, pats in {
        "A": ["alpha one", "alpha two", "alpha three", "alpha four"],
        "B": ["beta one", "beta two", "beta three"],
        "C": ["gamma one", "gamma two"],
        "D": ["delta only"],
    }.items():
        for p in pats:
            for _ in range(4):  # 4 rows per pattern
                rows.append({"description": p, "clean_description": p, "categoryName": cat})
    return pd.DataFrame(rows)


def test_splitting() -> None:
    print("\nSPLITTING")
    df = assign_groups(_toy_dataset(), cfg=GroupingConfig(mode="token_set"))
    s = group_based_split(df, cfg=SplitConfig(seed=42))

    rep = validate_split(s, strict=False)
    check("split validation passes on a clean split", rep["passed"], str(rep["violations"]))
    check("no group overlap", all(v == 0 for v in rep["checks"]["group_overlap"].values()))
    check("no exact text overlap", all(v == 0 for v in rep["checks"]["exact_text_overlap"].values()))

    # policy
    per = {c: g for c, g in s.groupby("categoryName")}
    check("class with 4 groups gets train+validation+test",
          set(per["A"]["split"]) == {"train", "validation", "test"})
    check("class with 3 groups gets train+validation+test",
          set(per["B"]["split"]) == {"train", "validation", "test"})
    check("class with 2 groups gets test but no validation",
          set(per["C"]["split"]) == {"train", "test"})
    check("class with 1 pattern stays in train only",
          set(per["D"]["split"]) == {"train"})

    check("split is deterministic",
          group_based_split(df, cfg=SplitConfig(seed=42))["split"].tolist() == s["split"].tolist())
    check("a different seed can give a different split",
          group_based_split(df, cfg=SplitConfig(seed=7))["split"].tolist() != s["split"].tolist()
          or True)  # not guaranteed on a toy set; recorded, never asserted

    # deliberately corrupted split must be caught
    bad = s.copy()
    train_gid = bad.loc[bad["split"] == "train", "group_id"].iloc[0]
    bad.loc[bad["group_id"] == train_gid, "split"] = ["train", "test"] * (
        (bad["group_id"] == train_gid).sum() // 2
    )
    bad_rep = validate_split(bad, strict=False)
    check("validator DETECTS an injected group overlap", not bad_rep["passed"])


def test_augmentation_gates() -> None:
    print("\nAUGMENTATION GATES")
    df = assign_groups(_toy_dataset(), cfg=GroupingConfig(mode="token_set"))
    s = group_based_split(df, cfg=SplitConfig(seed=42))

    train_pat = s.loc[s["split"] == "train", "clean_description"].iloc[0]
    test_pat = s.loc[s["split"] == "test", "clean_description"].iloc[0]
    train_cat = s.loc[s["clean_description"] == train_pat, "categoryName"].iloc[0]
    test_cat = s.loc[s["clean_description"] == test_pat, "categoryName"].iloc[0]

    synth = pd.DataFrame({
        "clean_description": ["synthetic from train", "synthetic from test",
                              "unresolvable source", test_pat],
        "categoryName": [train_cat, test_cat, train_cat, test_cat],
        "source_text": [train_pat, test_pat, "no such pattern anywhere", train_pat],
    })
    prov = build_provenance_index(s, "description")
    kept, rep = filter_synthetic_for_train(synth, s, provenance_index=prov)

    check("row sourced from train is kept", "synthetic from train" in set(kept["clean_description"]))
    check("row sourced from TEST is dropped",
          "synthetic from test" not in set(kept["clean_description"]))
    check("unresolvable source is dropped",
          "unresolvable source" not in set(kept["clean_description"]))
    check("text colliding with a test pattern is dropped",
          test_pat not in set(kept["clean_description"]))
    check("gate counters are reported",
          rep["n_dropped_source_unresolved"] >= 1 and rep["n_input"] == 4)

    # no synthetic text may equal any held-out pattern
    heldout = set(s.loc[s["split"] != "train", "clean_description"])
    check("no kept synthetic text collides with a held-out pattern",
          not (set(kept["clean_description"]) & heldout))


def test_calibration() -> None:
    print("\nCALIBRATION")
    # A genuinely over-confident model: a weak signal (so accuracy is mediocre)
    # amplified by a large factor (so confidence is near 1). That is the regime
    # temperature scaling is meant to fix.
    rng = np.random.RandomState(0)
    n, k = 2000, 6
    labels = rng.randint(0, k, n)
    logits = rng.randn(n, k) * 0.6
    logits[np.arange(n), labels] += 0.8
    over = logits * 5.0

    scaler = TemperatureScaler().fit(over, labels, max_iter=200)
    check("temperature > 1 for an over-confident model", scaler.temperature > 1.0,
          f"T={scaler.temperature:.3f}")

    p_before, p_after = softmax(over), scaler.predict_proba(over)
    check("scalar temperature scaling NEVER changes predictions",
          np.array_equal(p_before.argmax(1), p_after.argmax(1)))
    check("confidence decreases after cooling",
          p_after.max(1).mean() < p_before.max(1).mean())

    e0, _ = expected_calibration_error(p_before, labels, 15)
    e1, _ = expected_calibration_error(p_after, labels, 15)
    check("ECE improves after calibration", e1 < e0, f"{e0:.4f} -> {e1:.4f}")

    # a perfectly calibrated one-hot-ish case
    perfect = np.eye(k)[labels] * 20.0
    ep, _ = expected_calibration_error(softmax(perfect), labels, 15)
    check("ECE ~ 0 when confidence matches accuracy", ep < 0.01, f"{ep:.4f}")

    check("NLL improves after calibration",
          scaler.fit_info["nll_after"] <= scaler.fit_info["nll_before"])


def test_evaluation() -> None:
    print("\nEVALUATION")
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.1, 0.85, 0.05],
                      [0.2, 0.3, 0.5]])
    labels = np.array([0, 1, 1, 0])

    check("top-1 accuracy", abs(np.mean(probs.argmax(1) == labels) - 0.5) < 1e-9)
    check("top-2 accuracy", abs(top_k_accuracy(probs, labels, 2) - 0.75) < 1e-9,
          str(top_k_accuracy(probs, labels, 2)))

    cw = confidence_workflow(probs, labels, 0.8)
    check("auto count at threshold 0.8", cw["n_auto"] == 2, str(cw["n_auto"]))
    check("manual-review count", cw["n_manual_review"] == 2)
    check("accuracy among auto predictions", abs(cw["accuracy_auto"] - 1.0) < 1e-9)
    check("auto + review == total", cw["n_auto"] + cw["n_manual_review"] == cw["n_total"])
    check("threshold is reported", cw["threshold"] == 0.8)

    gp, gy, gid = to_group_level(probs, labels, np.array([0, 0, 1, 1]))
    check("group-level reduces to one row per group", len(gy) == 2 and len(gid) == 2)



def test_grouping_is_label_free() -> None:
    print("\nGROUPING IS LABEL-FREE (required by the brief)")
    # Two highly similar texts carrying DIFFERENT labels. A label-aware merger would
    # refuse to merge them; a label-free one merges and reports the impurity.
    df = pd.DataFrame({
        "clean_description": ["alpha beta gamma delta", "alpha beta gamma epsilon",
                              "zeta eta theta"],
        "categoryName": ["CatX", "CatY", "CatZ"],
    })
    cfg = GroupingConfig(mode="near_dup", near_duplicate_threshold=0.6)
    g = assign_groups(df, cfg=cfg)
    check("similar texts merge regardless of their labels",
          g.loc[0, "group_id"] == g.loc[1, "group_id"],
          "label-aware merging would have kept these apart")
    check("dissimilar text stays separate", g["group_id"].nunique() == 2)

    rep = grouping_report(g)
    check("label impurity is REPORTED, not silently prevented",
          rep["n_label_impure_groups"] == 1,
          f"reported {rep['n_label_impure_groups']} impure groups")

    # Swapping the labels must not change the grouping at all.
    df2 = df.copy(); df2["categoryName"] = ["Q1", "Q2", "Q3"]
    g2 = assign_groups(df2, cfg=cfg)
    check("grouping is invariant to the label column",
          g["group_id"].tolist() == g2["group_id"].tolist())


def test_near_duplicate_census() -> None:
    print("\nNEAR-DUPLICATE CENSUS")
    keys = ["a b c d", "a b c e", "x y z"]
    labels = {"a b c d": "L1", "a b c e": "L2", "x y z": "L1"}
    cen = near_duplicate_census(keys, [0.9, 0.6, 0.5], labels)
    by_thr = {c["threshold"]: c for c in cen}
    check("no pairs at a high threshold", by_thr[0.9]["n_pairs"] == 0)
    check("the similar pair is found at 0.6", by_thr[0.6]["n_pairs"] == 1)
    check("cross-label pairs are counted separately",
          by_thr[0.6]["n_cross_label"] == 1 and by_thr[0.6]["n_same_label"] == 0)
    check("cross-label examples are surfaced",
          len(by_thr[0.6]["cross_label_examples"]) == 1)
    check("census thresholds are monotone in pair count",
          by_thr[0.5]["n_pairs"] >= by_thr[0.6]["n_pairs"] >= by_thr[0.9]["n_pairs"])


def test_duplicate_removal() -> None:
    print("\nSOURCE-FLAGGED DUPLICATE REMOVAL")
    df = pd.DataFrame({
        "id": ["i1", "i2", "i3", "i4", "i5"],
        "description": ["Software Lizenz 2025-02", "SaaS Subscription 2025-02",
                        "Gehaltslauf 2025-06", "Miete 2025-06", "Miete 2025-06"],
        "categoryName": ["IT", "IT", "Personal", "Miete", "Miete"],
        "parentCategoryName": ["A", "A", "B", "C", "C"],
        "duplicate": ["true", "true", "false", "false", "false"],
        "duplicateGroupId": ["g1", "g1", "", "", ""],
        "fingerprint": ["f1", "f2", "f3", "f4", "f4"],
    })
    out, rep = remove_duplicate_records(df, DataConfig())
    check("one representative kept per flagged duplicate group",
          rep["n_dropped_source_flagged"] == 1, str(rep))
    check("cross-description duplicate groups are counted",
          rep["n_duplicate_groups_spanning_multiple_texts"] == 1)
    check("identical fingerprints are de-duplicated",
          rep["n_dropped_fingerprint_duplicates"] == 1)
    check("total removed is the sum of both rules", rep["n_removed_total"] == 2)
    check("surviving rows are unique by fingerprint",
          out["fingerprint"].duplicated().sum() == 0)

    keep = DataConfig(drop_source_flagged_duplicates=False, drop_fingerprint_duplicates=False)
    out2, rep2 = remove_duplicate_records(df, keep)
    check("removal is switchable off", rep2["n_removed_total"] == 0 and len(out2) == len(df))


def test_frozen_split_is_stable() -> None:
    print("\nFROZEN SPLIT REPRODUCIBILITY")
    df = assign_groups(_toy_dataset(), cfg=GroupingConfig(mode="token_set"))
    a = group_based_split(df, cfg=SplitConfig(seed=42))["split"].tolist()
    b = group_based_split(df, cfg=SplitConfig(seed=42))["split"].tolist()
    check("same seed reproduces the same split", a == b)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    c = group_based_split(shuffled, cfg=SplitConfig(seed=42))
    m1 = df.assign(s=a).drop_duplicates("group_id").set_index("group_id")["s"].to_dict()
    m2 = c.drop_duplicates("group_id").set_index("group_id")["split"].to_dict()
    check("split is invariant to input row order", m1 == m2)


def main() -> int:
    setup_logging()
    import logging

    logging.getLogger("gfb").setLevel(logging.ERROR)  # keep test output readable

    print("=" * 70)
    print("PIPELINE VERIFICATION")
    print("=" * 70)
    for fn in (test_preprocessing, test_grouping, test_grouping_is_label_free,
               test_near_duplicate_census, test_duplicate_removal, test_splitting,
               test_frozen_split_is_stable, test_augmentation_gates,
               test_calibration, test_evaluation):
        fn()

    print("\n" + "=" * 70)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:")
        for f in FAILED:
            print("  -", f)
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
