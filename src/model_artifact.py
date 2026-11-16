"""Package a ``train_model.py`` run directory as a model-registry version.

Registry version layout (weights are never committed to git)::

    model_registry/versions/<version_id>/
        model/                     weights (safetensors) + tokenizer, from <run>/model
        label_mapping.json         37 class names (from the run)
        config.json                full ExperimentConfig of the run (preprocessing, training)
        model_meta.json            training history, environment, data checksums (from the run)
        validation_metrics.json    test_metrics.json  (as recorded by train_model.py)
        category_parents.json      categoryName -> parentCategoryName (frozen split)
        train_vocabulary.json      cleaned training tokens (monitoring: unseen-token rate)
        reference_profile.json     monitoring reference (validation split)
        metadata.json              version, provenance, metrics, checksums

Usage::

    python -m src.model_artifact --run artifacts/runs/germanfinbert_seed42_cpu [--approve --reason "..."]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PreprocessingConfig
from .dataset_io import load_frozen_split, split_frame
from .mlops_settings import PROJECT_ROOT, get, load_mlops_config, resolve_path
from .monitoring import build_reference_profile
from .preprocessing import clean_description
from .registry import ModelRegistry

logger = logging.getLogger(__name__)
ARTIFACT_FORMAT = "transaction-classifier/v1"
RUN_FILES = ("label_mapping.json", "config.json", "model_meta.json", "validation_metrics.json",
             "test_metrics.json")


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_file(model_dir: Path) -> Path:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ArtifactError(f"No .safetensors weights in {model_dir} (pickle-based .bin is not accepted)")
    return files[0]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def training_vocabulary(frozen: pd.DataFrame, dataset_dir: Path, augmentation_version: str | None) -> set[str]:
    texts = split_frame(frozen, "train")["clean_description"].astype(str).tolist()
    if augmentation_version:
        corpus = dataset_dir / f"augmentation_corpus_{augmentation_version}.csv"
        if corpus.exists():
            texts += pd.read_csv(corpus, encoding="utf-8-sig")["clean_description"].astype(str).tolist()
    return {tok for t in texts for tok in t.split()}


def summarise_metrics(block: dict) -> dict[str, Any]:
    out = {}
    for level in ("group_level", "row_level"):
        m = block.get(level, {})
        cw = m.get("confidence_workflow", {})
        out[level] = {k: m.get(k) for k in ("n", "accuracy", "macro_f1", "weighted_f1", "top_2_accuracy")}
        out[level].update({"pct_auto": cw.get("pct_auto"), "accuracy_auto": cw.get("accuracy_auto"),
                           "pct_manual_review": cw.get("pct_manual_review")})
    return out


def package_run(run_dir: str | Path, registry: ModelRegistry, cfg: dict[str, Any],
                dataset_dir: str | Path = "results/dataset", extra_metadata: dict | None = None,
                version_prefix: str | None = None) -> str:
    """Copy a run into the registry as a new (unapproved) version. Returns its id."""
    run = Path(run_dir)
    model_dir = run / "model"
    for f in RUN_FILES:
        if not (run / f).exists():
            raise ArtifactError(f"{run} is not a complete training run: missing {f}")
    wfile = weights_file(model_dir)
    content_sha = sha256_file(wfile)
    run_cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    meta_run = json.loads((run / "model_meta.json").read_text(encoding="utf-8"))
    labels = json.loads((run / "label_mapping.json").read_text(encoding="utf-8"))["class_names"]
    dataset_dir = resolve_path(dataset_dir)
    frozen, manifest = load_frozen_split(dataset_dir, require_checksum=True)
    pre = PreprocessingConfig(**run_cfg["preprocessing"])
    threshold = float(meta_run.get("confidence_threshold", get(cfg, "model.confidence_threshold", 0.8)))
    aug = meta_run.get("data", {}).get("augmentation", {})
    vocab = training_vocabulary(frozen, dataset_dir, aug.get("augmentation_version") if aug.get("augmentation_enabled") else None)

    val = split_frame(frozen, "validation")
    # allow_pickle=False: only the numeric array is read (class_names is an object array we skip)
    with np.load(run / "logits.npz", allow_pickle=False) as z:
        val_probs = np.asarray(z["validation_probabilities"], dtype=float)
    if len(val_probs) != len(val):
        raise ArtifactError("validation probabilities do not align with the frozen validation split")
    raw = val["description"].astype(str).tolist()
    cleaned = [clean_description(t, pre) for t in raw]
    reference = build_reference_profile(raw, cleaned, val_probs, vocab, threshold)
    parents = (frozen.drop_duplicates("categoryName").set_index("categoryName")["parentCategoryName"]
               .astype(str).to_dict())

    model_name = meta_run.get("model", {}).get("short_name", "model")
    seed = run_cfg.get("training", {}).get("seed")
    version_id = f"{version_prefix or model_name}-seed{seed}-{content_sha[:10]}"
    test_m = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    val_m = json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "version_id": version_id,
        "model_name": get(cfg, "registry.model_name", "transaction-classifier"),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": str(run),
        "base_checkpoint": meta_run.get("model", {}).get("hf_name"),
        "base_revision": get(cfg, "model.base_revision"),
        "n_classes": len(labels),
        "seed": seed,
        "confidence_threshold": threshold,
        "calibration": "none (uncalibrated softmax is the primary operational result)",
        "content_sha256": content_sha,
        "weights_file": wfile.name,
        "training": {"config": run_cfg.get("training"), "seconds": meta_run.get("training", {}).get("seconds"),
                     "device": meta_run.get("device"), "environment": meta_run.get("environment")},
        "data": {"frozen_split_sha256": manifest.get("sha256_row_uid_group_split"),
                 "augmentation_corpus_sha256": aug.get("augmentation_corpus_sha256"),
                 "augmentation_version": aug.get("augmentation_version"),
                 "n_train_rows": meta_run.get("data", {}).get("n_train_rows")},
        "metrics": {"validation": summarise_metrics(val_m), "test": summarise_metrics(test_m),
                    "note": "test metrics are those recorded by train_model.py after fixed-epoch training; "
                            "33 independent test groups (see README limitations)"},
        "git_commit": git_commit(),
    }
    metadata.update(extra_metadata or {})
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        shutil.copytree(model_dir, dest / "model")
        for f in RUN_FILES:
            shutil.copy2(run / f, dest / f)
        (dest / "category_parents.json").write_text(json.dumps(parents, ensure_ascii=False, indent=1), encoding="utf-8")
        (dest / "train_vocabulary.json").write_text(json.dumps(sorted(vocab), ensure_ascii=False), encoding="utf-8")
        (dest / "reference_profile.json").write_text(json.dumps(reference), encoding="utf-8")
        registry.register(dest, version_id, metadata)
    logger.info("Packaged %s as registry version %s", run, version_id)
    return version_id


def verify_version(version_dir: Path) -> dict[str, Any]:
    """Integrity check used by the API before serving: format + weights checksum."""
    meta = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("artifact_format") != ARTIFACT_FORMAT:
        raise ArtifactError(f"Unsupported artifact format {meta.get('artifact_format')!r}")
    w = version_dir / "model" / meta["weights_file"]
    if not w.exists():
        raise ArtifactError(f"Missing weights {w}")
    if sha256_file(w) != meta["content_sha256"]:
        raise ArtifactError(f"Checksum mismatch for {w}: artifact is corrupted")
    return meta


def main(argv: list[str] | None = None) -> int:
    from .logging_utils import setup_logging
    ap = argparse.ArgumentParser(description="Register a training run in the model registry")
    ap.add_argument("--run", required=True, help="run directory written by train_model.py")
    ap.add_argument("--approve", action="store_true", help="approve for serving (explicit decision)")
    ap.add_argument("--reason", help="reason recorded with the approval")
    ap.add_argument("--dataset-dir", default="results/dataset")
    args = ap.parse_args(argv)
    setup_logging("INFO", fmt="text")
    cfg = load_mlops_config()
    reg = ModelRegistry(resolve_path(get(cfg, "registry.dir", "model_registry")))
    vid = package_run(args.run, reg, cfg, args.dataset_dir)
    if args.approve:
        if not args.reason:
            ap.error("--approve requires --reason")
        reg.approve(vid, {"reason": args.reason, "type": "manual_approval"})
    print(json.dumps({"version_id": vid, "approved": reg.approved_version() == vid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
