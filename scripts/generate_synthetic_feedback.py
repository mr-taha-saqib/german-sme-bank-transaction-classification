"""Generate SYNTHETIC labelled feedback to exercise the retraining workflow.

THIS IS DEMONSTRATION DATA. Real retraining needs genuinely confirmed labels from
the client's bookkeeping. Modes (combine with --mode several times):

train_variants : TRAINING-split descriptions with new dates / reference numbers
                 (cleaning removes them, so they reinforce known training patterns)
heldout_leak   : VALIDATION / TEST descriptions with new dates. Must be rejected by
                 the leakage gate (exact or near-duplicate of a held-out pattern)
invalid        : unknown labels, empty / whitespace text, missing values, duplicates

    python scripts/generate_synthetic_feedback.py --mode train_variants --mode heldout_leak \\
        --mode invalid --n 200 --out data/feedback/synthetic_feedback.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_io import load_frozen_split, split_frame  # noqa: E402


def _variant(text: str, rng: np.random.Generator) -> str:
    year, month = int(rng.integers(2026, 2028)), int(rng.integers(1, 13))
    suffix = rng.choice([f"{year}-{month:02d}", f"RE-{year}-{int(rng.integers(1000, 9999))}",
                         f"{int(rng.integers(1, 28)):02d}.{month:02d}.{year}"])
    return f"{text} {suffix}"


def generate(modes: list[str], n: int, seed: int = 11, dataset_dir: Path = ROOT / "results" / "dataset") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frozen, _ = load_frozen_split(dataset_dir)
    parts = []
    if "train_variants" in modes:
        tr = split_frame(frozen, "train").sample(n=n, replace=True, random_state=seed)
        parts.append(pd.DataFrame({"description": [_variant(t, rng) for t in tr["description"]],
                                   "categoryName": tr["categoryName"].tolist()}))
    if "heldout_leak" in modes:
        ho = frozen[frozen["split"] != "train"].sample(n=max(10, n // 5), replace=True, random_state=seed)
        parts.append(pd.DataFrame({"description": [_variant(t, rng) for t in ho["description"]],
                                   "categoryName": ho["categoryName"].tolist()}))
    if "invalid" in modes:
        parts.append(pd.DataFrame({
            "description": ["Software Lizenz 2026-02", "   ", None, "Miete Lager", "Miete Lager", "2026-03 ###"],
            "categoryName": ["Nicht-Existierende-Kategorie", "IT & Software", "IT & Software", None,
                             "Miete & Nebenkosten", "Miete & Nebenkosten"]}))
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", action="append", choices=["train_variants", "heldout_leak", "invalid"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = generate(args.mode, args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} SYNTHETIC labelled rows ({'+'.join(args.mode)}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
