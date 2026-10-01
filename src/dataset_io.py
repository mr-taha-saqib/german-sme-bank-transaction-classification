"""Loading the frozen Step-2 dataset artefact.

Step 3 and everything after it must consume the split produced by
`prepare_dataset.py` rather than recomputing it. Recomputation would silently drift
if preprocessing, grouping or the seed ever changed, and the two models being
compared could then end up on different data without anyone noticing.

`load_frozen_split` therefore re-derives the SHA-256 of ``(row_uid, group_id,
split)`` and refuses to proceed if it disagrees with the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .utils import get_logger

REQUIRED_COLUMNS = [
    "row_uid", "description", "clean_description", "group_key", "group_id",
    "categoryName", "parentCategoryName", "split",
]
SPLITS = ("train", "validation", "test")


def split_checksum(df: pd.DataFrame) -> str:
    """Checksum of the split assignment itself, independent of column order."""
    # Fixed CRLF so the recorded checksums (computed on Windows) match on every OS.
    payload = df[["row_uid", "group_id", "split"]].to_csv(index=False, lineterminator="\r\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frozen_split(
    dataset_dir: str | Path = "results/dataset",
    require_checksum: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load `split_assignment.csv` and verify it against `split_manifest.json`."""
    log = get_logger()
    root = Path(dataset_dir)
    split_path = root / "split_assignment.csv"
    manifest_path = root / "split_manifest.json"

    if not split_path.exists():
        raise FileNotFoundError(
            f"No frozen split at {split_path}.\n"
            f"Run `python prepare_dataset.py` (Step 2) before training."
        )
    df = pd.read_csv(split_path, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frozen split is missing columns: {missing}")

    bad = sorted(set(df["split"]) - set(SPLITS))
    if bad:
        raise ValueError(f"Frozen split contains unexpected split labels: {bad}")

    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = split_checksum(df)
        expected = manifest.get("sha256_row_uid_group_split")
        if require_checksum and expected and actual != expected:
            raise ValueError(
                "Frozen split checksum mismatch -- the split file has changed since it "
                f"was frozen.\n  manifest: {expected}\n  file:     {actual}\n"
                "Re-run `python prepare_dataset.py` and re-train, or investigate."
            )
        log.info("Frozen split verified (sha256 %s)", actual[:16])
    elif require_checksum:
        raise FileNotFoundError(f"No manifest at {manifest_path}; cannot verify integrity.")

    # A group must never straddle splits. Cheap to assert, catastrophic if violated.
    per_group = df.groupby("group_id")["split"].nunique()
    if int((per_group > 1).sum()):
        raise ValueError(
            f"{int((per_group > 1).sum())} groups span more than one split -- "
            f"the frozen artefact is not leakage-safe."
        )

    log.info(
        "Loaded frozen split: %d rows | %d groups | %s",
        len(df), df["group_id"].nunique(),
        " | ".join(
            f"{s} {int((df['split'] == s).sum())}r/{df.loc[df['split'] == s, 'group_id'].nunique()}g"
            for s in SPLITS if (df["split"] == s).any()
        ),
    )
    return df, manifest


def build_label_mapping(df: pd.DataFrame, label_col: str = "categoryName") -> tuple[list[str], dict]:
    """Deterministic label mapping over the FULL category inventory.

    Built from every split, not just training, so that the same integer means the
    same category for every model and every run. Sorted, so it is reproducible.
    """
    class_names = sorted(df[label_col].astype(str).unique())
    return class_names, {c: i for i, c in enumerate(class_names)}


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df[df["split"] == split].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Split {split!r} is empty in the frozen artefact.")
    return out
