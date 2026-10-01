"""Leakage-safe, group-based train / validation / test splitting.

Policy (per category, on that category's list of distinct groups, shuffled with a
fixed seed):

    >= 3 groups ->  1 test group, 1 validation group, remainder to train
       2 groups ->  1 test group, 0 validation,       1 group to train
       1 group  ->  0 test,       0 validation,       1 group to train

Consequences, stated openly rather than hidden:
  * Categories with a single pattern cannot be tested without leakage, so they are
    train-only and simply do not appear in the test set.
  * Categories with two patterns contribute exactly one test item, so their per-class
    recall can only be 0.0 or 1.0.
Both facts are recorded in the split report.

The validation split exists so that model-facing decisions and, in Step 3,
temperature scaling are fitted on data used for nothing else -- never on test.
A 4-way train/validation/calibration/test split is not affordable here: only 13 of
37 categories have >= 4 groups, so validation doubles as the validation set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitConfig
from .grouping import find_near_duplicate_pairs, jaccard, token_set_key
from .utils import get_logger

SPLITS = ("train", "validation", "test")


def group_based_split(
    df: pd.DataFrame,
    label_col: str = "categoryName",
    cfg: SplitConfig | None = None,
) -> pd.DataFrame:
    """Assign every row a ``split`` label. Groups never cross splits."""
    cfg = cfg or SplitConfig()
    log = get_logger()
    rng = np.random.RandomState(cfg.seed)

    # One row per (category, group) so allocation is over groups, not rows.
    cat_groups = (
        df[[label_col, "group_id"]].drop_duplicates().sort_values([label_col, "group_id"])
    )

    assignment: dict[int, str] = {}
    for category in sorted(cat_groups[label_col].unique()):
        gids = cat_groups.loc[cat_groups[label_col] == category, "group_id"].tolist()
        gids = sorted(gids)          # deterministic starting order
        rng.shuffle(gids)            # then a seeded shuffle
        n = len(gids)

        if n >= cfg.min_groups_for_validation:
            assignment[gids[0]] = "test"
            assignment[gids[1]] = "validation"
            for g in gids[2:]:
                assignment[g] = "train"
        elif n >= cfg.min_groups_for_test:
            assignment[gids[0]] = "test"
            for g in gids[1:]:
                assignment[g] = "train"
        else:
            for g in gids:
                assignment[g] = "train"

    out = df.copy()
    out["split"] = out["group_id"].map(assignment)
    if out["split"].isna().any():
        raise RuntimeError("Internal error: some rows were not assigned to a split")

    log.info(
        "Group split (seed %d): train %d rows / %d groups | validation %d / %d | test %d / %d",
        cfg.seed,
        *(v for s in SPLITS for v in (
            int((out["split"] == s).sum()),
            int(out.loc[out["split"] == s, "group_id"].nunique()),
        )),
    )
    return out


def provenance_aligned_split(
    df: pd.DataFrame,
    synthetic_source_keys: set[str],
    text_col: str = "description",
    label_col: str = "categoryName",
    cfg: SplitConfig | None = None,
) -> pd.DataFrame:
    """Hold out exactly the patterns the LLM generator never saw.

    The shipped synthetic corpus was generated from a fixed set of *training-side*
    patterns. Any pattern absent from that set is, by construction, one the generator
    never observed -- which makes it a valid unseen-pattern test item and keeps the
    entire augmentation corpus usable.

    The test set is therefore inherited from the prior work, not chosen by looking at
    model performance. Validation groups are then carved from the remaining
    (generator-seen) patterns; synthetic rows sourced from those are dropped by the
    normal provenance gate.
    """
    from .grouping import token_set_key
    from .preprocessing import legacy_clean_description

    cfg = cfg or SplitConfig()
    log = get_logger()
    rng = np.random.RandomState(cfg.seed)

    # Which of the current groups did the generator see?
    legacy_of_group: dict[int, set[str]] = {}
    for raw, gid in zip(df[text_col].astype(str), df["group_id"]):
        legacy_of_group.setdefault(gid, set()).add(
            token_set_key(legacy_clean_description(raw))
        )
    seen = {
        gid for gid, keys in legacy_of_group.items()
        if keys & synthetic_source_keys
    }

    assignment: dict[int, str] = {}
    cat_groups = df[[label_col, "group_id"]].drop_duplicates()
    n_rescued = 0

    for category in sorted(cat_groups[label_col].unique()):
        gids = sorted(cat_groups.loc[cat_groups[label_col] == category, "group_id"].tolist())
        unseen = [g for g in gids if g not in seen]
        seen_g = [g for g in gids if g in seen]

        # A category must keep at least one training group.
        if not seen_g and unseen:
            sizes = df[df["group_id"].isin(unseen)].groupby("group_id").size()
            keep = int(sizes.idxmax())
            unseen = [g for g in unseen if g != keep]
            seen_g = [keep]
            n_rescued += 1

        for g in unseen:
            assignment[g] = "test"

        rng.shuffle(seen_g)
        if len(seen_g) >= 2:
            assignment[seen_g[0]] = "validation"
            for g in seen_g[1:]:
                assignment[g] = "train"
        else:
            for g in seen_g:
                assignment[g] = "train"

    out = df.copy()
    out["split"] = out["group_id"].map(assignment)
    if out["split"].isna().any():
        raise RuntimeError("Internal error: some rows were not assigned to a split")

    if n_rescued:
        log.warning(
            "%d categories had no generator-seen pattern; their largest group was "
            "moved into training so the class remains learnable", n_rescued
        )
    log.info(
        "Provenance-aligned split (seed %d): train %d rows / %d groups | "
        "validation %d / %d | test %d / %d",
        cfg.seed,
        *(v for s in SPLITS for v in (
            int((out["split"] == s).sum()),
            int(out.loc[out["split"] == s, "group_id"].nunique()),
        )),
    )
    return out


def leaky_random_split(
    df: pd.DataFrame,
    label_col: str = "categoryName",
    cfg: SplitConfig | None = None,
) -> pd.DataFrame:
    """Row-level stratified split -- the prototype's original method.

    Reproduced *only* as a labelled reference run that quantifies how much leakage
    inflates the result. Never used as a headline number.
    """
    from sklearn.model_selection import train_test_split

    cfg = cfg or SplitConfig()
    out = df.copy()

    # Stratification needs >= 2 members per class in the split.
    counts = out[label_col].value_counts()
    stratify = out[label_col] if counts.min() >= 2 else None

    train_idx, test_idx = train_test_split(
        out.index, test_size=cfg.leaky_test_size,
        random_state=cfg.seed, stratify=stratify,
    )
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


def validate_split(
    df: pd.DataFrame,
    text_col: str = "clean_description",
    label_col: str = "categoryName",
    audit_threshold: float = 0.60,
    strict: bool = True,
) -> dict:
    """Verify the three leakage guarantees the brief requires.

    1. No group appears in more than one split.
    2. No exact cleaned text appears in more than one split.
    3. Near-duplicate audit across splits (reported; not automatically fatal, since
       lexical near-duplicates across *different* labels are legitimate).
    """
    log = get_logger()
    present = [s for s in SPLITS if (df["split"] == s).any()]

    groups = {s: set(df.loc[df["split"] == s, "group_id"]) for s in present}
    texts = {s: set(df.loc[df["split"] == s, text_col].astype(str)) for s in present}

    report: dict = {"splits_present": present, "checks": {}, "violations": []}

    # --- 1. group overlap ---------------------------------------------------
    group_overlaps = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            inter = groups[a] & groups[b]
            group_overlaps[f"{a}|{b}"] = len(inter)
            if inter:
                report["violations"].append(
                    f"GROUP OVERLAP {a}/{b}: {len(inter)} shared group ids "
                    f"(e.g. {sorted(inter)[:5]})"
                )
    report["checks"]["group_overlap"] = group_overlaps

    # --- 2. exact text overlap ---------------------------------------------
    text_overlaps = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            inter = texts[a] & texts[b]
            text_overlaps[f"{a}|{b}"] = len(inter)
            if inter:
                report["violations"].append(
                    f"EXACT TEXT OVERLAP {a}/{b}: {len(inter)} shared cleaned texts "
                    f"(e.g. {sorted(inter)[:3]})"
                )
    report["checks"]["exact_text_overlap"] = text_overlaps

    # --- 3. near-duplicate audit -------------------------------------------
    near: dict = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            pairs = find_near_duplicate_pairs(
                sorted(texts[a]), sorted(texts[b]), threshold=audit_threshold
            )
            same_label, cross_label = [], []
            lab_a = (
                df[df["split"] == a].drop_duplicates(text_col)
                .set_index(text_col)[label_col].to_dict()
            )
            lab_b = (
                df[df["split"] == b].drop_duplicates(text_col)
                .set_index(text_col)[label_col].to_dict()
            )
            for ta, tb, sim in pairs:
                entry = {"a": ta, "b": tb, "jaccard": sim,
                         "label_a": lab_a.get(ta), "label_b": lab_b.get(tb)}
                (same_label if entry["label_a"] == entry["label_b"] else cross_label).append(entry)
            near[f"{a}|{b}"] = {
                "n_pairs_total": len(pairs),
                "n_same_label": len(same_label),
                "n_cross_label": len(cross_label),
                "same_label_examples": same_label[:20],
                "cross_label_examples": cross_label[:10],
            }
            if same_label:
                report["violations"].append(
                    f"NEAR-DUPLICATE (same label) {a}/{b}: {len(same_label)} pairs at "
                    f"Jaccard >= {audit_threshold}"
                )
    report["checks"]["near_duplicate"] = near

    # --- descriptive stats --------------------------------------------------
    stats = {}
    for s in present:
        sub = df[df["split"] == s]
        stats[s] = {
            "n_rows": int(len(sub)),
            "n_groups": int(sub["group_id"].nunique()),
            "n_unique_texts": int(sub[text_col].nunique()),
            "n_classes": int(sub[label_col].nunique()),
            "rows_per_class": {str(k): int(v) for k, v in sub[label_col].value_counts().items()},
            "groups_per_class": {
                str(k): int(v) for k, v in sub.groupby(label_col)["group_id"].nunique().items()
            },
        }
    report["split_stats"] = stats

    all_classes = set(df[label_col].unique())
    if "test" in present:
        test_classes = set(df.loc[df["split"] == "test", label_col].unique())
        report["classes_absent_from_test"] = sorted(all_classes - test_classes)
        single_item = [
            str(c) for c, n in df[df["split"] == "test"].groupby(label_col)["group_id"]
            .nunique().items() if n == 1
        ]
        report["test_classes_with_single_group"] = sorted(single_item)
    if "validation" in present:
        cal_classes = set(df.loc[df["split"] == "validation", label_col].unique())
        report["classes_absent_from_validation"] = sorted(all_classes - cal_classes)

    hard = [v for v in report["violations"] if v.startswith(("GROUP OVERLAP", "EXACT TEXT"))]
    report["passed"] = not hard
    report["n_violations"] = len(report["violations"])

    for v in report["violations"]:
        log.error("LEAKAGE CHECK: %s", v)
    if hard and strict:
        raise AssertionError(
            "Leakage validation FAILED (group or exact-text overlap across splits):\n  "
            + "\n  ".join(hard)
        )
    if report["passed"]:
        log.info("Leakage validation PASSED: no group overlap, no exact text overlap")
    return report


def build_provenance_index(split_df: pd.DataFrame, text_col: str = "description") -> dict:
    """Map a legacy-normalised pattern to the set of current group keys it covers.

    The shipped synthetic corpus records ``source_text`` in the *prototype's* text
    normalisation (which, for example, collapsed ``B2B`` to ``b b``). The current
    cleaner is deliberately different, so a direct string comparison fails and every
    synthetic row would be discarded. Rebuilding the legacy key for each real row
    gives an exact bridge between the two normalisations.

    One legacy pattern can map to several current groups, because the current
    cleaner is finer-grained. Any such fan-out is handled conservatively by the
    caller: the synthetic row is kept only if *none* of the mapped groups is held out.
    """
    from .preprocessing import legacy_clean_description
    from .grouping import token_set_key

    index: dict[str, set[str]] = {}
    for raw, gkey in zip(split_df[text_col].astype(str), split_df["group_key"]):
        legacy = token_set_key(legacy_clean_description(raw))
        index.setdefault(legacy, set()).add(gkey)
    return index


def filter_synthetic_for_train(
    synth: pd.DataFrame,
    split_df: pd.DataFrame,
    text_col: str = "clean_description",
    label_col: str = "categoryName",
    require_source_in_train: bool = True,
    drop_collisions_with_heldout: bool = True,
    provenance_index: dict | None = None,
    near_duplicate_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Keep only synthetic rows that are provably safe for the training split.

    Two independent gates, both applied:

    A) provenance -- the row's ``source_text`` must resolve to real pattern(s) that
       are *all* in the TRAINING split. This is the strong guarantee: it does not
       matter which split the file was originally generated for.
    B) collision -- the synthetic text itself must not coincide with any validation
       or test pattern, either exactly or by canonical group key.
    C) near-duplicate -- the synthetic text must not reach `near_duplicate_threshold`
       token-Jaccard against any validation/test pattern.

    Gate C exists because gates A and B test only exact equality, while the project's
    grouping declares two texts to be the SAME leakage unit at Jaccard >= 0.60. Without
    it a synthetic row can enter training even though, had it been real data, grouping
    would have placed it inside a held-out group. Pass the SAME threshold the grouping
    uses (`GroupingConfig.near_duplicate_threshold`) so the two definitions cannot
    drift apart.
    """
    log = get_logger()
    from .grouping import token_set_key
    from .preprocessing import legacy_clean_description

    train_keys = set(split_df.loc[split_df["split"] == "train", "group_key"])
    heldout_texts = set(
        split_df.loc[split_df["split"] != "train", text_col].astype(str)
    )
    heldout_keys = set(split_df.loc[split_df["split"] != "train", "group_key"])

    out = synth.copy()
    n0 = len(out)
    report: dict = {"n_input": n0}

    # Gate A -------------------------------------------------------------
    if require_source_in_train:
        if "source_text" not in out.columns or out["source_text"].isna().all():
            raise ValueError(
                "Augmentation requires a `source_text` provenance column. "
                "Set augmentation.require_source_in_train=False to override "
                "(NOT recommended -- leakage can then not be verified)."
            )
        if provenance_index is None:
            provenance_index = build_provenance_index(split_df)

        def resolve(src: str) -> set[str] | None:
            legacy = token_set_key(legacy_clean_description(src))
            groups = provenance_index.get(legacy)
            if groups:
                return groups
            # Fall back to the current normalisation, in case a corpus was written
            # with texts that were never passed through the legacy cleaner.
            direct = token_set_key(str(src))
            return {direct} if direct in train_keys or direct in heldout_keys else None

        resolved = out["source_text"].astype(str).map(resolve)
        unresolved = resolved.isna()
        touches_heldout = resolved.map(
            lambda g: bool(g) and bool(g & heldout_keys) if g else False
        )
        fully_in_train = resolved.map(
            lambda g: bool(g) and g.issubset(train_keys) if g else False
        )

        report["n_dropped_source_unresolved"] = int(unresolved.sum())
        report["n_dropped_source_from_heldout"] = int(touches_heldout.sum())
        report["n_dropped_source_not_in_train"] = int((~fully_in_train).sum())

        if unresolved.any():
            log.warning(
                "Gate A: %d synthetic rows have a `source_text` that matches no real "
                "pattern; dropped as unverifiable (examples: %s)",
                int(unresolved.sum()),
                out.loc[unresolved, "source_text"].astype(str).unique()[:3].tolist(),
            )

        out = out[fully_in_train].copy()
        log.info(
            "Synthetic gate A (provenance): kept %d/%d rows | dropped %d whose source "
            "pattern is held out, %d unresolvable",
            len(out), n0, report["n_dropped_source_from_heldout"],
            report["n_dropped_source_unresolved"],
        )

    # Gate B -------------------------------------------------------------
    if drop_collisions_with_heldout:
        n1 = len(out)
        collides_text = out[text_col].astype(str).isin(heldout_texts)
        collides_key = out[text_col].astype(str).map(token_set_key).isin(heldout_keys)
        collides = collides_text | collides_key
        report["n_dropped_collision_with_heldout"] = int(collides.sum())
        out = out[~collides].copy()
        log.info("Synthetic gate B (collision): kept %d/%d rows; dropped %d colliding "
                 "with a validation/test pattern", len(out), n1, int(collides.sum()))

    # Gate C -------------------------------------------------------------
    if near_duplicate_threshold is not None and len(out):
        n1 = len(out)
        heldout_tokens = {k: set(k.split()) for k in heldout_keys}
        # Inverted index over held-out tokens: only keys sharing a token can reach a
        # positive Jaccard, so most comparisons are skipped.
        inverted: dict[str, list[str]] = {}
        for k, toks in heldout_tokens.items():
            for t in toks:
                inverted.setdefault(t, []).append(k)

        def max_similarity(text: str) -> float:
            toks = set(token_set_key(str(text)).split())
            cands = {c for t in toks for c in inverted.get(t, ())}
            best = 0.0
            for c in cands:
                j = jaccard(toks, heldout_tokens[c])
                if j > best:
                    best = j
            return best

        sims = out[text_col].astype(str).map(max_similarity)
        too_close = sims >= near_duplicate_threshold
        report["near_duplicate_threshold"] = float(near_duplicate_threshold)
        report["n_dropped_near_duplicate_of_heldout"] = int(too_close.sum())
        report["max_similarity_to_heldout_after_gate"] = round(
            float(sims[~too_close].max()) if (~too_close).any() else 0.0, 4)
        if too_close.any():
            examples = (out.loc[too_close, [text_col, label_col]]
                        .drop_duplicates(text_col).head(10))
            report["near_duplicate_examples_dropped"] = examples.to_dict(orient="records")
        out = out[~too_close].copy()
        log.info(
            "Synthetic gate C (near-duplicate, Jaccard >= %.2f): kept %d/%d rows; "
            "dropped %d that were near-copies of a validation/test pattern "
            "(max remaining similarity %.4f)",
            near_duplicate_threshold, len(out), n1, int(too_close.sum()),
            report["max_similarity_to_heldout_after_gate"],
        )

    # Deduplicate ---------------------------------------------------------
    n2 = len(out)
    out = out.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    report["n_dropped_duplicates"] = int(n2 - len(out))

    report["n_output"] = int(len(out))
    report["n_classes"] = int(out[label_col].nunique()) if len(out) else 0
    report["per_class"] = {
        str(k): int(v) for k, v in out[label_col].value_counts().items()
    } if len(out) else {}
    report["n_source_patterns_used"] = (
        int(out["source_text"].nunique()) if "source_text" in out.columns and len(out) else 0
    )
    log.info("Synthetic corpus after filtering: %d rows across %d classes",
             report["n_output"], report["n_classes"])
    return out, report
