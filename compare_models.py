#!/usr/bin/env python
"""STEP 4 - aggregate the multi-seed model comparison from saved run artefacts.

    python compare_models.py
    python compare_models.py --runs-dir results/step4_comparison --out results/step4_comparison

Reads only what each run wrote to disk. It re-verifies that every run used the same
frozen split and the same augmentation corpus before comparing anything, because a
comparison across different data is worthless however tidy the table looks.

Uncalibrated results only. No temperature scaling is applied anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("top_2_accuracy", "Top-2"),
]


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(runs_dir.glob("*_seed*")):
        tm = d / "test_metrics.json"
        mm = d / "model_meta.json"
        if not (tm.exists() and mm.exists()):
            continue
        model, _, seed = d.name.rpartition("_seed")
        metrics = json.loads(tm.read_text(encoding="utf-8"))
        meta = json.loads(mm.read_text(encoding="utf-8"))
        runs.append({"dir": d, "model": model, "seed": int(seed),
                     "metrics": metrics, "meta": meta})
    return runs


def fairness_report(runs: list[dict]) -> tuple[bool, list[str]]:
    """Every run must share split, corpus, label mapping and evaluation protocol."""
    issues: list[str] = []

    def field(r, *path, default=None):
        cur = r["meta"]
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        return cur if cur is not None else default

    splits = {field(r, "data", "frozen_split_sha256") for r in runs}
    if len(splits) != 1:
        issues.append(f"runs used {len(splits)} different frozen splits: {splits}")

    corpora = {field(r, "data", "augmentation", "augmentation_corpus_sha256") for r in runs}
    if len(corpora) != 1:
        issues.append(f"runs used {len(corpora)} different augmentation corpora: {corpora}")

    versions = {field(r, "data", "augmentation", "augmentation_version") for r in runs}
    if len(versions) != 1:
        issues.append(f"runs used different corpus versions: {versions}")

    thresholds = {field(r, "confidence_threshold") for r in runs}
    if len(thresholds) != 1:
        issues.append(f"runs used different confidence thresholds: {thresholds}")

    # Identical hyperparameters except the checkpoint itself.
    keys = ["epochs", "learning_rate", "batch_size", "max_length", "loss",
            "focal_gamma", "use_class_weights", "warmup_mode", "warmup_ratio",
            "model_selection", "gradient_accumulation_steps"]
    for k in keys:
        vals = {field(r, "training", "config", k) for r in runs}
        if len(vals) != 1:
            issues.append(f"training config differs on {k!r}: {vals}")

    lm = set()
    for r in runs:
        p = r["dir"] / "label_mapping.json"
        if p.exists():
            lm.add(json.dumps(json.loads(p.read_text(encoding="utf-8"))["class_names"]))
    if len(lm) > 1:
        issues.append("runs used different label mappings")

    ns = {(r["metrics"]["group_level"]["n"], r["metrics"]["row_level"]["n"]) for r in runs}
    if len(ns) != 1:
        issues.append(f"runs evaluated different numbers of test items: {ns}")

    return (not issues), issues


def per_seed_table(runs: list[dict], level: str) -> pd.DataFrame:
    rows = []
    for r in runs:
        b = r["metrics"][level]
        cw = b["confidence_workflow"]
        cs = b["confidence_statistics"]
        rows.append({
            "Model": r["model"], "Seed": r["seed"], "N": b["n"],
            "Accuracy": b["accuracy"], "Macro F1": b["macro_f1"],
            "Weighted F1": b["weighted_f1"], "Top-2": b["top_2_accuracy"],
            "Mean conf": cs["mean"], "Max conf": cs["max"],
            ">=0.8 %": cw["pct_auto"], "Acc >=0.8": cw["accuracy_auto"],
        })
    return pd.DataFrame(rows).sort_values(["Model", "Seed"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in df.columns if c not in ("Model", "Seed", "N")]
    out = []
    for model, g in df.groupby("Model"):
        row = {"Model": model, "Seeds": len(g)}
        for c in num:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals) == 0:
                row[c] = None
                continue
            row[c] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}" if len(vals) > 1 \
                else f"{vals.mean():.4f}"
        out.append(row)
    return pd.DataFrame(out)


def paired_analysis(df: pd.DataFrame, metric: str = "Accuracy") -> dict:
    """Per-seed paired differences. Deliberately no p-value on 5 paired points."""
    piv = df.pivot(index="Seed", columns="Model", values=metric)
    if piv.shape[1] != 2:
        return {}
    a, b = sorted(piv.columns)
    diff = (piv[a] - piv[b]).dropna()
    return {
        "metric": metric,
        "model_a": a, "model_b": b,
        "per_seed_difference": {int(k): round(float(v), 5) for k, v in diff.items()},
        "mean_difference": round(float(diff.mean()), 5),
        "std_difference": round(float(diff.std(ddof=1)), 5) if len(diff) > 1 else None,
        "n_seeds_a_ahead": int((diff > 0).sum()),
        "n_seeds_b_ahead": int((diff < 0).sum()),
        "n_seeds_tied": int((diff == 0).sum()),
        "sign_consistent": bool((diff > 0).all() or (diff < 0).all()),
    }


def md(df: pd.DataFrame) -> str:
    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    d = df.copy()
    for c in d.columns:
        d[c] = d[c].map(cell)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |"
                                    for r in d.itertuples(index=False)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="results/step4_comparison")
    ap.add_argument("--out", default="results/step4_comparison")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir)
    if not runs:
        print(f"No completed runs found in {runs_dir}")
        return 2

    print("=" * 78)
    print(f"MODEL COMPARISON  ({len(runs)} runs)   UNCALIBRATED")
    print("=" * 78)

    ok, issues = fairness_report(runs)
    print("\nFAIRNESS PRECONDITIONS")
    if ok:
        m0 = runs[0]["meta"]
        print("  [PASS] all runs share one frozen split, one augmentation corpus,")
        print("         one label mapping, one threshold and one training configuration")
        print(f"         split sha256  : {m0['data']['frozen_split_sha256'][:24]}")
        aug = m0["data"]["augmentation"]
        print(f"         corpus        : {aug.get('augmentation_version')} "
              f"sha256 {str(aug.get('augmentation_corpus_sha256'))[:24]}")
        print(f"         gates         : {aug.get('augmentation_gates_applied')}")
        print(f"         threshold     : {m0['confidence_threshold']}")
    else:
        for i in issues:
            print(f"  [FAIL] {i}")
        print("\n  Comparison ABORTED: the runs are not comparable.")
        return 1

    results: dict = {"n_runs": len(runs), "fairness_ok": ok}
    for level in ("group_level", "row_level"):
        df = per_seed_table(runs, level)
        summ = summary_table(df)
        tag = level.replace("_level", "")
        df.to_csv(Path(args.out) / f"per_seed_{tag}.csv", index=False, encoding="utf-8-sig")
        summ.to_csv(Path(args.out) / f"summary_{tag}.csv", index=False, encoding="utf-8-sig")

        print(f"\n--- PER SEED ({tag} level) ---")
        print(df.to_string(index=False, na_rep="-"))
        print(f"\n--- MEAN +/- SD ({tag} level) ---")
        print(summ.to_string(index=False, na_rep="-"))

        results[f"per_seed_{tag}"] = df.to_dict(orient="records")
        results[f"summary_{tag}"] = summ.to_dict(orient="records")
        results[f"markdown_per_seed_{tag}"] = md(df)
        results[f"markdown_summary_{tag}"] = md(summ)

        if tag == "group":
            print("\n--- PAIRED PER-SEED DIFFERENCES (group level) ---")
            for metric in ("Accuracy", "Macro F1", "Top-2", ">=0.8 %"):
                pa = paired_analysis(df, metric)
                if not pa:
                    continue
                results.setdefault("paired", {})[metric] = pa
                print(f"  {metric:12s} {pa['model_a']} - {pa['model_b']}: "
                      f"mean {pa['mean_difference']:+.4f}"
                      + (f" (sd {pa['std_difference']:.4f})" if pa["std_difference"] else "")
                      + f" | {pa['model_a']} ahead on {pa['n_seeds_a_ahead']}/"
                        f"{len(pa['per_seed_difference'])} seeds"
                      + ("  [sign consistent]" if pa["sign_consistent"] else "  [sign FLIPS]"))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWritten to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
