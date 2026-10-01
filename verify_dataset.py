#!/usr/bin/env python
"""Independent verification of the frozen Step-2 dataset artefact.

Reads only what is on disk. It does not import the split logic, so it cannot inherit
a bug from it -- it re-derives every claim from `split_assignment.csv`.

    python verify_dataset.py
    python verify_dataset.py --dataset results/dataset

Proves the five checks required by the brief:

  CHECK 1  no group occurs in more than one split
  CHECK 2  no exact duplicate cleaned description occurs across splits
  CHECK 3  near-duplicate leakage across splits is measured and reported
  CHECK 4  class distributions are reported for each split
  CHECK 5  rows and groups per split are reported

Plus integrity checks on the frozen artefact itself (checksum, manifest agreement).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PASSED: list[str] = []
FAILED: list[str] = []

SPLITS = ("train", "validation", "test")


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


def jaccard(a: set[str], b: set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="results/dataset")
    ap.add_argument("--near-dup-threshold", type=float, default=0.60)
    args = ap.parse_args()

    root = Path(args.dataset)
    split_path = root / "split_assignment.csv"
    if not split_path.exists():
        print(f"No frozen split at {split_path}. Run `python prepare_dataset.py` first.")
        return 2

    df = pd.read_csv(split_path, encoding="utf-8-sig")
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))

    print("=" * 78)
    print(f"VERIFYING FROZEN DATASET  {root.resolve()}")
    print("=" * 78)
    print(f"  rows={len(df)}  groups={df['group_id'].nunique()}  "
          f"categories={df['categoryName'].nunique()}")

    # ---- artefact integrity ---------------------------------------------
    print("\nARTEFACT INTEGRITY")
    # Fixed CRLF so the recorded checksums (computed on Windows) match on every OS.
    payload = df[["row_uid", "group_id", "split"]].to_csv(index=False, lineterminator="\r\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    check("frozen split matches the recorded sha256",
          digest == manifest["sha256_row_uid_group_split"],
          f"file {digest[:16]} vs manifest {manifest['sha256_row_uid_group_split'][:16]}")
    check("manifest row count agrees with the file",
          manifest["n_rows"] == len(df), f"{manifest['n_rows']} vs {len(df)}")
    check("manifest group count agrees with the file",
          manifest["n_groups"] == df["group_id"].nunique())
    check("row_uid is unique", df["row_uid"].is_unique,
          f"{int(df['row_uid'].duplicated().sum())} duplicated ids")
    check("every row has a split label",
          df["split"].isin(SPLITS).all(),
          f"unexpected: {sorted(set(df['split']) - set(SPLITS))}")

    groups = {s: set(g["group_id"]) for s, g in df.groupby("split")}
    texts = {s: set(g["clean_description"].astype(str)) for s, g in df.groupby("split")}
    present = [s for s in SPLITS if s in groups]

    # ---- CHECK 1 ---------------------------------------------------------
    print("\nCHECK 1  no group occurs in more than one split")
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            inter = groups[a] & groups[b]
            check(f"{a} and {b} share no group_id", not inter,
                  f"{len(inter)} shared, e.g. {sorted(inter)[:5]}")
    per_group = df.groupby("group_id")["split"].nunique()
    check("every group_id maps to exactly one split", bool((per_group == 1).all()),
          f"{int((per_group > 1).sum())} groups span multiple splits")

    # ---- CHECK 2 ---------------------------------------------------------
    print("\nCHECK 2  no exact duplicate cleaned description across splits")
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            inter = texts[a] & texts[b]
            check(f"{a} and {b} share no cleaned description", not inter,
                  f"{len(inter)} shared, e.g. {sorted(inter)[:3]}")
    per_text = df.groupby("clean_description")["split"].nunique()
    check("every cleaned description maps to exactly one split",
          bool((per_text == 1).all()),
          f"{int((per_text > 1).sum())} texts span multiple splits")

    # ---- CHECK 3 ---------------------------------------------------------
    print(f"\nCHECK 3  near-duplicate leakage across splits (Jaccard >= "
          f"{args.near_dup_threshold})")
    lab = df.drop_duplicates("clean_description").set_index("clean_description")["categoryName"]
    total_same = total_cross = 0
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            ka = {t: set(str(t).split()) for t in texts[a]}
            kb = {t: set(str(t).split()) for t in texts[b]}
            same = cross = 0
            worst = []
            for ta, sa in ka.items():
                for tb, sb in kb.items():
                    s = jaccard(sa, sb)
                    if s >= args.near_dup_threshold:
                        if lab.get(ta) == lab.get(tb):
                            same += 1
                            worst.append((round(s, 3), ta, tb, lab.get(ta)))
                        else:
                            cross += 1
            total_same += same
            total_cross += cross
            print(f"         {a:11s} vs {b:11s}: {same} same-category, "
                  f"{cross} cross-category pair(s)")
            for w in sorted(worst, reverse=True)[:3]:
                print(f"             {w[0]}  '{w[1]}'  ~  '{w[2]}'  [{w[3]}]")
    check("no same-category near-duplicate pair crosses a split", total_same == 0,
          f"{total_same} pairs at Jaccard >= {args.near_dup_threshold}")
    print(f"         (cross-category near-duplicates are reported, not treated as "
          f"leakage: {total_cross} pair(s))")

    # ---- CHECK 4 ---------------------------------------------------------
    print("\nCHECK 4  class distribution per split")
    dist = (df.pivot_table(index="categoryName", columns="split",
                           values="row_uid", aggfunc="count", fill_value=0))
    gdist = (df.drop_duplicates(["group_id"])
               .pivot_table(index="categoryName", columns="split",
                            values="group_id", aggfunc="count", fill_value=0))
    for s in present:
        if s not in dist.columns:
            dist[s] = 0
        if s not in gdist.columns:
            gdist[s] = 0
    check("every category appears in the training split",
          bool((dist.get("train", pd.Series(dtype=int)) > 0).all()),
          f"missing from train: {dist.index[dist['train'] == 0].tolist()}")
    absent_test = dist.index[dist.get("test", 0) == 0].tolist()
    print(f"         categories absent from test: {len(absent_test)} {absent_test}")
    print(f"         categories absent from validation: "
          f"{int((dist.get('validation', 0) == 0).sum())}")
    check("class distribution table was written by prepare_dataset.py",
          (root / "class_distribution.csv").exists())
    print("\n         rows per split, top 10 categories by training size:")
    top = dist.sort_values("train", ascending=False).head(10)
    print(top.to_string().replace("\n", "\n         "))

    # ---- CHECK 5 ---------------------------------------------------------
    print("\nCHECK 5  rows and groups per split")
    for s in present:
        sub = df[df["split"] == s]
        n_rows, n_groups = len(sub), sub["group_id"].nunique()
        m = manifest["splits"].get(s, {})
        check(f"{s}: manifest agrees with the file "
              f"({n_rows} rows, {n_groups} groups)",
              m.get("n_rows") == n_rows and m.get("n_groups") == n_groups,
              f"manifest says {m}")
        print(f"         {s:11s} {n_rows:5d} rows | {n_groups:3d} groups | "
              f"{sub['categoryName'].nunique():2d} categories | "
              f"{sub['clean_description'].nunique():3d} distinct texts")

    print("\n" + "=" * 78)
    print(f"{len(PASSED)} checks passed, {len(FAILED)} failed")
    if FAILED:
        print("\nFAILED:")
        for f in FAILED:
            print("  -", f)
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
