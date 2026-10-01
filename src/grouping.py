"""Deterministic grouping of near-identical transaction descriptions.

Purpose
-------
The dataset is template-generated: ``Software Lizenz 2024-09``,
``Software Lizenz 2025-02`` and 71 further rows all reduce to ``software lizenz``
after cleaning. If such rows fall on both sides of a split, the model is tested on
strings it has already memorised. A *group* is therefore the unit of leakage: all
rows in a group must live in exactly one split.

Algorithm (two stages, both deterministic and both label-free)
--------------------------------------------------------------
1. **Canonical key.** Each cleaned description is reduced to its sorted set of
   unique tokens, so ``miete buero`` and ``buero miete`` collapse together.
2. **Near-duplicate merge.** Distinct keys whose token-Jaccard similarity reaches
   ``near_duplicate_threshold`` are merged by union-find.

**The target label is never consulted when forming groups.** Label purity of the
resulting groups is computed afterwards, as a diagnostic only (`grouping_report`).
This matters: merging only within a label would leak label information into the
split geometry, and would silently hide cross-category near-duplicates instead of
reporting them.

Choosing the threshold
----------------------
`near_duplicate_census` measures the actual similarity structure of a corpus, so the
threshold is chosen from data rather than convention. On the provided dataset:

    Jaccard >= 0.7 :   0 pairs                      (nothing to merge)
    Jaccard >= 0.6 :  61 pairs, 0 cross-category    <- plateau, chosen
    Jaccard >= 0.5 : 168 pairs, 7 cross-category    (would fuse categories)

0.6 is the highest threshold that captures real near-duplicates -- chiefly
milestone-numbered variants such as ``1 automation milestone projektabrechnung``
versus ``3 automation milestone projektabrechnung`` -- while merging nothing across
category boundaries. 0.55 yields an identical grouping, so 0.6 sits on a plateau
rather than on a knife edge.

Limitations
-----------
1. Grouping is lexical. Two semantically identical descriptions with disjoint
   vocabulary (``gehaltslauf`` vs ``payroll sammelueberweisung``) stay separate.
   That is conservative for correctness -- it never hides leakage -- but it makes
   the split harder than a semantic grouping would.
2. Jaccard over token sets ignores word order and token frequency.
3. Because merging is label-free, a merge *can* in principle fuse two categories.
   That is reported as a label-impure group rather than silently prevented; on this
   dataset at 0.6 it does not occur.
4. Group sizes stay very unequal (1 to 78 rows here), so a group-level split cannot
   simultaneously balance row counts. Both views are reported downstream.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from .config import GroupingConfig
from .utils import get_logger


def token_set_key(text: str) -> str:
    """Order-insensitive canonical key: sorted unique tokens."""
    return " ".join(sorted(set(str(text).split())))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: the lexicographically smaller root always wins.
            lo, hi = sorted([ra, rb])
            self.parent[hi] = lo


def near_duplicate_census(
    keys: list[str],
    thresholds: list[float] | None = None,
    labels: dict[str, str] | None = None,
) -> list[dict]:
    """Measure the similarity structure of a set of canonical keys.

    `labels` is used **only** to annotate how many pairs would cross a category
    boundary. It never influences grouping; it exists so that the threshold choice
    can be justified, and so cross-category near-duplicates are surfaced rather than
    hidden.
    """
    thresholds = thresholds or [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5]
    keys = sorted(set(keys))
    toks = {k: set(k.split()) for k in keys}
    out = []
    for thr in thresholds:
        same = cross = 0
        examples: list[dict] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s = jaccard(toks[keys[i]], toks[keys[j]])
                if s < thr:
                    continue
                if labels is None or labels.get(keys[i]) == labels.get(keys[j]):
                    same += 1
                else:
                    cross += 1
                    if len(examples) < 5:
                        examples.append({
                            "a": keys[i], "b": keys[j], "jaccard": round(s, 4),
                            "label_a": labels.get(keys[i]),
                            "label_b": labels.get(keys[j]),
                        })
        out.append({
            "threshold": thr,
            "n_pairs": same + cross,
            "n_same_label": same,
            "n_cross_label": cross,
            "cross_label_examples": examples,
        })
    return out


def assign_groups(
    df: pd.DataFrame,
    text_col: str = "clean_description",
    label_col: str = "categoryName",
    cfg: GroupingConfig | None = None,
) -> pd.DataFrame:
    """Add ``group_key`` and ``group_id``. Deterministic for a given input.

    `label_col` is accepted for reporting compatibility but is deliberately NOT used
    to form groups.
    """
    cfg = cfg or GroupingConfig()
    log = get_logger()
    out = df.copy()

    if cfg.mode == "exact":
        out["group_key"] = out[text_col].astype(str)
    elif cfg.mode in ("token_set", "near_dup"):
        out["group_key"] = out[text_col].astype(str).map(token_set_key)
    else:
        raise ValueError(f"Unknown grouping mode: {cfg.mode!r}")

    if cfg.mode == "near_dup":
        keys = sorted(out["group_key"].unique())
        toks = {k: set(k.split()) for k in keys}
        uf = _UnionFind(keys)
        merged = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if jaccard(toks[keys[i]], toks[keys[j]]) >= cfg.near_duplicate_threshold:
                    uf.union(keys[i], keys[j])
                    merged += 1
        out["group_key"] = out["group_key"].map(uf.find)
        log.info(
            "near_dup grouping: %d canonical keys, %d similar pairs merged "
            "(threshold %.2f, label-free) -> %d groups",
            len(keys), merged, cfg.near_duplicate_threshold, out["group_key"].nunique(),
        )

    ordered = sorted(out["group_key"].unique())
    out["group_id"] = out["group_key"].map({k: i for i, k in enumerate(ordered)})

    log.info("Grouping mode=%s: %d rows -> %d groups (mean size %.2f, max %d)",
             cfg.mode, len(out), out["group_id"].nunique(),
             len(out) / max(out["group_id"].nunique(), 1),
             int(out.groupby("group_id").size().max()))
    return out


def grouping_report(
    df: pd.DataFrame, label_col: str = "categoryName", text_col: str = "clean_description"
) -> dict:
    """Statistics needed to justify the split design."""
    sizes = df.groupby("group_id").size()
    per_cat = df.groupby(label_col)["group_id"].nunique().sort_values()
    rows_per_cat = df[label_col].value_counts()

    impure = df.groupby("group_id")[label_col].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    impure_detail = []
    for gid in impure:
        sub = df[df["group_id"] == gid]
        impure_detail.append({
            "group_id": int(gid),
            "group_key": str(sub["group_key"].iloc[0]),
            "labels": sorted(sub[label_col].unique().tolist()),
            "n_rows": int(len(sub)),
        })

    return {
        "n_rows": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "n_unique_cleaned_texts": int(df[text_col].nunique()),
        "group_size_min": int(sizes.min()),
        "group_size_mean": round(float(sizes.mean()), 2),
        "group_size_median": int(sizes.median()),
        "group_size_max": int(sizes.max()),
        "effective_sample_size_note": (
            f"{len(df)} rows correspond to only {df['group_id'].nunique()} independent "
            f"groups ({len(df) / max(df['group_id'].nunique(), 1):.1f}x overstatement "
            f"if rows are treated as independent)"
        ),
        "groups_per_category": {str(k): int(v) for k, v in per_cat.items()},
        "rows_per_category": {str(k): int(v) for k, v in rows_per_cat.items()},
        "categories_with_1_group": int((per_cat == 1).sum()),
        "categories_with_lt3_groups": int((per_cat < 3).sum()),
        "categories_with_lt4_groups": int((per_cat < 4).sum()),
        "median_groups_per_category": int(per_cat.median()),
        "largest_groups": [
            {"group_key": str(df.loc[df["group_id"] == gid, "group_key"].iloc[0]),
             "n_rows": int(n)}
            for gid, n in sizes.sort_values(ascending=False).head(15).items()
        ],
        "label_impure_groups": impure_detail,
        "n_label_impure_groups": len(impure_detail),
    }


def find_near_duplicate_pairs(
    texts_a: list[str], texts_b: list[str], threshold: float = 0.60
) -> list[tuple[str, str, float]]:
    """Cross-split near-duplicate audit (used by `splitting.validate_split`)."""
    ta = {t: set(t.split()) for t in set(texts_a)}
    tb = {t: set(t.split()) for t in set(texts_b)}
    index: dict[str, list[str]] = defaultdict(list)
    for t, toks in ta.items():
        for tok in toks:
            index[tok].append(t)

    pairs: list[tuple[str, str, float]] = []
    for b, tb_toks in tb.items():
        candidates = Counter()
        for tok in tb_toks:
            for a in index.get(tok, ()):
                candidates[a] += 1
        for a in candidates:
            s = jaccard(ta[a], tb_toks)
            if s >= threshold:
                pairs.append((a, b, round(s, 4)))
    return sorted(pairs, key=lambda x: -x[2])
