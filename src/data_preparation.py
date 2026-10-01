"""Loading and consolidating the provided datasets.

Primary labelled data
---------------------
`synthetic_b2b_transactions_2000.xlsx` is not a normal workbook: it is a
single-column sheet whose cells hold a 65-field semicolon-delimited record. The
loader detects this and reconstructs the columns. (Verified during the audit: all
2 000 rows split into exactly 65 fields, so no field contains an embedded
semicolon.)

Augmentation data
-----------------
Only synthetic CSVs carrying a `source_text` provenance column are accepted by
default, because that column is what allows leakage to be excluded *mechanically*
rather than by trusting the file to already be clean. See `src/splitting.py`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig
from .utils import get_logger

REQUIRED_RAW_COLUMNS = ["description", "categoryName", "parentCategoryName"]


def load_raw_transactions(path: str | Path, cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load the labelled base workbook and return a tidy frame."""
    cfg = cfg or DataConfig()
    log = get_logger()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Primary dataset not found: {path}\n"
            f"Expected the handover workbook at this location."
        )

    df = pd.read_excel(path, dtype=str)

    # The handover file packs 65 fields into one column.
    if df.shape[1] == 1:
        header = list(df.columns)[0].split(";")
        split = df.iloc[:, 0].astype(str).str.split(";", expand=True)
        n_fields = split.shape[1]
        if n_fields != len(header):
            raise ValueError(
                f"Semicolon payload has {n_fields} fields but the header declares "
                f"{len(header)}. The file may contain embedded semicolons."
            )
        split.columns = header
        df = split
        log.info("Detected semicolon-packed workbook: expanded to %d columns", n_fields)

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)[:15]}...")

    keep = [c for c in ["id", "amount", "transactionDate", "type", "vendorName",
                        "vendorGroupId", "vendorGroupName", "counterpartyNameRaw",
                        "duplicate", "duplicateGroupId", "fingerprint",
                        "sourceOrder"] if c in df.columns]
    out = df[REQUIRED_RAW_COLUMNS + keep].copy()

    before = len(out)
    out = out.dropna(subset=REQUIRED_RAW_COLUMNS)
    for c in REQUIRED_RAW_COLUMNS:
        out = out[out[c].astype(str).str.strip() != ""]
    out = out.reset_index(drop=True)
    if len(out) != before:
        log.warning("Dropped %d rows with missing description/label", before - len(out))

    out["is_synthetic"] = False
    out["source_text"] = pd.NA
    log.info(
        "Loaded %d labelled transactions | %d categories | %d parent categories",
        len(out), out["categoryName"].nunique(), out["parentCategoryName"].nunique(),
    )
    return out


def remove_duplicate_records(
    df: pd.DataFrame, cfg: DataConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate transaction records flagged by the source system.

    Two independent signals, both intrinsic to the data and neither derived from the
    labels or from any split:

    * ``duplicate == "true"`` with a shared ``duplicateGroupId`` -- the source
      system's own duplicate detection. Crucially, some of these pairs carry
      *different* descriptions (``Software Lizenz 2025-02`` and ``SaaS Subscription
      2025-02``, same amount, same category), so text grouping alone would allow one
      to land in train and its twin in test.
    * an identical ``fingerprint`` -- byte-identical records.

    One representative per duplicate set is kept, chosen deterministically by ``id``
    so the outcome does not depend on row order or on any seed.

    Removing rather than merging is deliberate. Merging the affected text groups was
    measured and costs six additional categories their only testable group (4 -> 10
    categories reduced to a single group), whereas removal costs 32 of 2 000 rows and
    leaves group structure and category coverage untouched.
    """
    cfg = cfg or DataConfig()
    log = get_logger()
    out = df.copy()
    report: dict = {"n_input": int(len(out))}

    if cfg.drop_source_flagged_duplicates and {"duplicate", "duplicateGroupId"} <= set(out.columns):
        flagged = out[out["duplicate"].astype(str).str.strip().str.lower() == "true"]
        flagged = flagged[flagged["duplicateGroupId"].astype(str).str.strip() != ""]
        if len(flagged):
            sort_key = "id" if "id" in out.columns else flagged.columns[0]
            keep = (flagged.sort_values(["duplicateGroupId", sort_key])
                           .groupby("duplicateGroupId").head(1).index)
            drop = flagged.index.difference(keep)
            cross = sum(
                1 for _, sub in flagged.groupby("duplicateGroupId")
                if sub["description"].nunique() > 1
            )
            report["n_flagged_duplicate_rows"] = int(len(flagged))
            report["n_duplicate_groups"] = int(flagged["duplicateGroupId"].nunique())
            report["n_duplicate_groups_spanning_multiple_texts"] = int(cross)
            report["n_dropped_source_flagged"] = int(len(drop))
            out = out.drop(index=drop)
            log.info(
                "Dropped %d source-flagged duplicate rows (%d flagged rows in %d groups; "
                "%d of those groups span more than one description)",
                len(drop), len(flagged), flagged["duplicateGroupId"].nunique(), cross,
            )

    if cfg.drop_fingerprint_duplicates and "fingerprint" in out.columns:
        counts = out["fingerprint"].value_counts()
        dup_fp = counts[counts > 1].index
        if len(dup_fp):
            sub = out[out["fingerprint"].isin(dup_fp)]
            sort_key = "id" if "id" in out.columns else sub.columns[0]
            keep = sub.sort_values(["fingerprint", sort_key]).groupby("fingerprint").head(1).index
            drop = sub.index.difference(keep)
            report["n_dropped_fingerprint_duplicates"] = int(len(drop))
            out = out.drop(index=drop)
            log.info("Dropped %d rows sharing an identical fingerprint", len(drop))
        else:
            report["n_dropped_fingerprint_duplicates"] = 0

    out = out.reset_index(drop=True)
    report["n_output"] = int(len(out))
    report["n_removed_total"] = int(report["n_input"] - report["n_output"])
    return out, report


def load_synthetic(paths: list[str], require_source_text: bool = True) -> pd.DataFrame:
    """Load and concatenate the LLM-generated augmentation corpora."""
    log = get_logger()
    frames: list[pd.DataFrame] = []

    for p in paths:
        path = Path(p)
        if not path.exists():
            log.warning("Synthetic file not found, skipping: %s", path)
            continue
        # The notebook wrote these with utf-8-sig.
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

        if "clean_description" not in df.columns or "categoryName" not in df.columns:
            log.warning("Skipping %s: missing clean_description/categoryName", path.name)
            continue
        if require_source_text and "source_text" not in df.columns:
            log.warning(
                "Skipping %s: no `source_text` provenance column, so leakage cannot be "
                "verified mechanically (see AUDIT.md section 7.1)", path.name,
            )
            continue

        sub = pd.DataFrame({
            "description": df["clean_description"].astype(str),
            "categoryName": df["categoryName"].astype(str),
            "source_text": df["source_text"].astype(str) if "source_text" in df.columns else pd.NA,
            "source_file": path.name,
        })
        frames.append(sub)
        log.info("Loaded synthetic corpus %s: %d rows, %d categories",
                 path.name, len(sub), sub["categoryName"].nunique())

    if not frames:
        log.warning("No synthetic data loaded; training will use real data only")
        return pd.DataFrame(
            columns=["description", "categoryName", "source_text", "source_file"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["parentCategoryName"] = pd.NA
    out["is_synthetic"] = True
    return out


def attach_parent_categories(synth: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Fill the synthetic rows' parent category from the real taxonomy."""
    mapping = (
        real.drop_duplicates("categoryName")
        .set_index("categoryName")["parentCategoryName"]
        .to_dict()
    )
    synth = synth.copy()
    synth["parentCategoryName"] = synth["categoryName"].map(mapping)
    return synth


def dataset_summary(df: pd.DataFrame, label_col: str = "categoryName") -> dict:
    counts = df[label_col].value_counts()
    return {
        "n_rows": int(len(df)),
        "n_classes": int(counts.size),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "largest_class": [str(counts.index[0]), int(counts.iloc[0])] if counts.size else None,
        "smallest_class": [str(counts.index[-1]), int(counts.iloc[-1])] if counts.size else None,
        "imbalance_ratio": round(float(counts.iloc[0] / counts.iloc[-1]), 2) if counts.size else None,
    }
