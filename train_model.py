#!/usr/bin/env python
"""STEP 3 - fine-tune ONE transformer on the frozen Step-2 split.\n\npython train_model.py --model germanfinbert\npython train_model.py --model germanfinbert --augment --run-name step3_augmented\npython train_model.py --model germanfinbert --batch-size 8 --grad-accum 2\n\nDeliberately model-agnostic: Step 4 reuses this script unchanged for FinBERT, which\nis what guarantees the two models get identical data, loss, schedule and evaluation.\nThe only thing that differs is `--model`.\n\nWhat this script does NOT do\n----------------------------\n* It does not recompute the split. It loads `results/dataset/split_assignment.csv`\nand verifies its checksum, so every model trains on byte-identical data.\n* It does not calibrate. Temperature scaling is a later step; probabilities reported\nhere are the raw softmax.\n* It does not use the test split for anything until training is completely finished.\n* It does not use the 0.8 confidence threshold for any training or selection\ndecision. The threshold is applied once, at reporting time, to measure how the\ntrained model naturally behaves.\n"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import softmax
from src.config import MODEL_REGISTRY, ExperimentConfig
from src.data_preparation import attach_parent_categories, load_synthetic
from src.dataset_io import build_label_mapping, load_frozen_split, split_frame
from src.evaluation import (
    classification_metrics,
    confidence_workflow,
    confusion_matrix_frame,
    plot_confusion,
    plot_threshold_curve,
    predictions_frame,
    threshold_sweep,
)
from src.grouping import token_set_key
from src.models import load_model_and_tokenizer, tokenizer_diagnostics
from src.preprocessing import clean_series
from src.splitting import build_provenance_index, filter_synthetic_for_train
from src.training import build_dataset, predict_logits, train_model
from src.utils import (
    describe_device,
    environment_report,
    get_device,
    get_logger,
    save_json,
    section,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="germanfinbert", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--dataset-dir", default="results/dataset")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--config", help="ExperimentConfig JSON to start from")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--loss", choices=["focal", "cross_entropy"], default=None)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--augment", action="store_true",
                   help="add the frozen, gated LLM corpus to the TRAINING split only")
    p.add_argument("--augmentation-version", default="v2",
                   help="which frozen corpus to use: v2 = gates A+B+C (corrected), "
                        "v1 = gates A+B only (pre-correction, leaky)")
    p.add_argument("--early-stopping", action="store_true",
                   help="select the best epoch on the validation split (documented "
                        "trade-off: consumes validation, see STEP3_GERMANFINBERT.md)")
    p.add_argument("--no-save-weights", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    cfg.models = [args.model]
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.training.gradient_accumulation_steps = args.grad_accum
    if args.max_length is not None:
        cfg.training.max_length = args.max_length
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.loss is not None:
        cfg.training.loss = args.loss
    if args.no_class_weights:
        cfg.training.use_class_weights = False
    if args.no_amp:
        cfg.training.use_amp = False
    if args.threshold is not None:
        cfg.evaluation.confidence_threshold = args.threshold
    cfg.training.model_selection = "validation_best" if args.early_stopping else "fixed_epochs"
    cfg.augmentation.enabled = bool(args.augment)
    return cfg


def assemble_training_corpus(frozen: pd.DataFrame, cfg: ExperimentConfig,
                             augment: bool, dataset_dir: str,
                             version: str) -> tuple[pd.DataFrame, dict]:
    """Training rows = Step-2 train split, optionally plus the FROZEN synthetic corpus.\n\nThe augmentation corpus is loaded from the versioned artefact written by\n`prepare_augmentation.py` and its checksum is verified, exactly as the split is.\nThat makes "both models saw the same corpus" auditable rather than dependent on\nthe gating code being deterministic.\n"""
    import hashlib
    import json

    log = get_logger()
    train_real = split_frame(frozen, "train")[
        ["clean_description", cfg.data.label_column]].copy()
    train_real["is_synthetic"] = False
    report: dict = {"n_real_train_rows": int(len(train_real)), "augmentation_enabled": augment}

    if not augment:
        log.info("Training corpus: %d real rows (no augmentation)", len(train_real))
        return train_real.reset_index(drop=True), report

    root = Path(dataset_dir)
    corpus_path = root / f"augmentation_corpus_{version}.csv"
    manifest_path = root / f"augmentation_manifest_{version}.json"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"No frozen augmentation corpus at {corpus_path}.\n"
            f"Run `python prepare_augmentation.py --version {version}` first."
        )

    add = pd.read_csv(corpus_path, encoding="utf-8-sig")
    # Fixed CRLF so the recorded checksums (computed on Windows) match on every OS.
    payload = add[["clean_description", cfg.data.label_column]].to_csv(index=False, lineterminator="\r\n")
    checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("sha256_text_label")
        if expected and expected != checksum:
            raise ValueError(
                f"Augmentation corpus checksum mismatch.\nmanifest: {expected}\n"
                f"  file:     {checksum}\nRe-run prepare_augmentation.py."
            )
        if manifest.get("frozen_split_sha256") not in (None, ""):
            log.info("Corpus %s was built against split %s",
                     version, str(manifest["frozen_split_sha256"])[:16])
    log.info("Augmentation corpus %s verified (sha256 %s): %d rows, gates %s",
             version, checksum[:16], len(add),
             ", ".join(manifest.get("gates_applied", ["unknown"])))

    report["augmentation_version"] = version
    report["augmentation_corpus_sha256"] = checksum
    report["augmentation_gates_applied"] = manifest.get("gates_applied")
    report["augmentation_gate"] = manifest.get("gate_report")
    report["near_duplicate_threshold"] = manifest.get("near_duplicate_threshold")
    report["max_similarity_to_heldout"] = manifest.get("max_similarity_to_heldout")

    add = add[["clean_description", cfg.data.label_column]].copy()
    add["is_synthetic"] = True
    corpus = pd.concat([train_real, add], ignore_index=True)
    log.info("Training corpus: %d real + %d synthetic = %d rows",
             len(train_real), len(add), len(corpus))
    return corpus, report


def evaluate(logits: np.ndarray, labels: np.ndarray, group_ids: np.ndarray,
             class_names: list[str], cfg: ExperimentConfig) -> dict:
    """Uncalibrated evaluation. No temperature scaling is applied anywhere here."""
    from src.evaluation import to_group_level

    probs = softmax(logits)
    thr = cfg.evaluation.confidence_threshold
    conf = probs.max(axis=1)

    block: dict = {"probabilities": "uncalibrated (raw softmax)"}
    block["row_level"] = classification_metrics(probs, labels, class_names,
                                                cfg.evaluation.top_k)
    block["row_level"]["confidence_statistics"] = {
        "mean": round(float(conf.mean()), 5),
        "median": round(float(np.median(conf)), 5),
        "std": round(float(conf.std()), 5),
        "min": round(float(conf.min()), 5),
        "max": round(float(conf.max()), 5),
        "p25": round(float(np.percentile(conf, 25)), 5),
        "p75": round(float(np.percentile(conf, 75)), 5),
    }
    block["row_level"]["confidence_workflow"] = confidence_workflow(probs, labels, thr)
    block["row_level"]["threshold_sweep"] = threshold_sweep(
        probs, labels, cfg.evaluation.threshold_sweep).to_dict(orient="records")

    gp, gy, guniq = to_group_level(probs, labels, group_ids)
    gconf = gp.max(axis=1)
    block["group_level"] = classification_metrics(gp, gy, class_names, cfg.evaluation.top_k)
    block["group_level"]["n_independent_groups"] = int(len(guniq))
    block["group_level"]["confidence_statistics"] = {
        "mean": round(float(gconf.mean()), 5),
        "median": round(float(np.median(gconf)), 5),
        "std": round(float(gconf.std()), 5),
        "min": round(float(gconf.min()), 5),
        "max": round(float(gconf.max()), 5),
    }
    block["group_level"]["confidence_workflow"] = confidence_workflow(gp, gy, thr)
    block["group_level"]["threshold_sweep"] = threshold_sweep(
        gp, gy, cfg.evaluation.threshold_sweep).to_dict(orient="records")
    return block


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    run_name = args.run_name or f"step3_{args.model}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "train.log")

    section(f"step 3 - fine-tune {args.model}")
    log.info("Model        : %s -> %s", args.model, MODEL_REGISTRY[args.model])
    log.info("Dataset      : %s (frozen Step-2 split)", args.dataset_dir)
    log.info("Output       : %s", out_dir.resolve())
    log.info("Device       : %s", describe_device())
    log.info("Seed         : %d", cfg.training.seed)
    log.info("Augmentation : %s%s", cfg.augmentation.enabled,
             f" (corpus {args.augmentation_version})" if cfg.augmentation.enabled else "")
    log.info("Selection    : %s", cfg.training.model_selection)

    # -- data --------------------------------------------------------------
    section("data")
    frozen, manifest = load_frozen_split(args.dataset_dir, require_checksum=True)
    class_names, label2id = build_label_mapping(frozen, cfg.data.label_column)
    log.info("Label mapping: %d classes, built over the full inventory", len(class_names))

    train_df, aug_report = assemble_training_corpus(
        frozen, cfg, cfg.augmentation.enabled, args.dataset_dir, args.augmentation_version)
    train_df["label"] = train_df[cfg.data.label_column].map(label2id)

    val_df = split_frame(frozen, "validation")
    test_df = split_frame(frozen, "test")
    for d in (val_df, test_df):
        d["label"] = d[cfg.data.label_column].map(label2id)

    missing_train = sorted(set(class_names) - set(train_df[cfg.data.label_column]))
    if missing_train:
        log.warning("%d classes have no training example: %s",
                    len(missing_train), missing_train)

    log.info("Sizes: train %d rows | validation %d rows / %d groups | test %d rows / %d groups",
             len(train_df), len(val_df), val_df["group_id"].nunique(),
             len(test_df), test_df["group_id"].nunique())

    # -- model -------------------------------------------------------------
    section("model")
    set_seed(cfg.training.seed)
    device = get_device()
    model, tokenizer, minfo = load_model_and_tokenizer(
        args.model, len(class_names), {i: c for i, c in enumerate(class_names)})
    diag = tokenizer_diagnostics(tokenizer, train_df["clean_description"].tolist(),
                                 cfg.training.max_length)
    log.info("Tokenizer: %.3f subtokens/word | mean %.1f tokens | p95 %d | %d truncated | %d UNK",
             diag["subtokens_per_word"], diag["mean_tokens"], diag["p95_tokens"],
             diag["n_truncated"], diag["n_unk_tokens"])

    train_ds = build_dataset(train_df["clean_description"].tolist(),
                             train_df["label"].tolist(), tokenizer, cfg.training.max_length)
    val_ds = build_dataset(val_df["clean_description"].tolist(),
                           val_df["label"].tolist(), tokenizer, cfg.training.max_length)

    # -- train -------------------------------------------------------------
    section("training")
    if cfg.training.model_selection == "fixed_epochs":
        log.info("Fixed %d epochs, no checkpoint selection. Validation metrics are "
                 "recorded for MONITORING only; no configuration depends on them, so "
                 "the validation split stays available as an untouched calibration set.",
                 cfg.training.epochs)
    else:
        log.info("Best epoch will be selected on validation macro-F1 (documented "
                 "trade-off: validation is then no longer pristine for calibration).")

    t0 = time.time()
    tres = train_model(model, train_ds, len(class_names), cfg.training, device,
                       dev_dataset=val_ds)
    seconds = time.time() - t0
    log.info("Training finished in %.1f s", seconds)

    # -- validation (reporting only) --------------------------------------
    section("validation metrics")
    v_logits, v_labels = predict_logits(model, val_ds, device)
    val_block = evaluate(v_logits, v_labels, val_df["group_id"].to_numpy(),
                         class_names, cfg)
    log.info("validation | row-level acc %.4f macroF1 %.4f | group-level acc %.4f "
             "macroF1 %.4f (n=%d groups)",
             val_block["row_level"]["accuracy"], val_block["row_level"]["macro_f1"],
             val_block["group_level"]["accuracy"], val_block["group_level"]["macro_f1"],
             val_block["group_level"]["n_independent_groups"])

    # -- test (only now) ---------------------------------------------------
    section("test evaluation (after training is complete)")
    test_ds = build_dataset(test_df["clean_description"].tolist(),
                            test_df["label"].tolist(), tokenizer, cfg.training.max_length)
    t_logits, t_labels = predict_logits(model, test_ds, device)
    test_block = evaluate(t_logits, t_labels, test_df["group_id"].to_numpy(),
                          class_names, cfg)

    thr = cfg.evaluation.confidence_threshold
    for level in ("group_level", "row_level"):
        m = test_block[level]
        cw = m["confidence_workflow"]
        log.info("test %-12s | n=%-4d acc %.4f | macroF1 %.4f | weightedF1 %.4f | "
                 "top-2 %.4f | mean conf %.4f | >=%.2f: %.1f%% of predictions, acc %s",
                 level, m["n"], m["accuracy"], m["macro_f1"], m["weighted_f1"],
                 m.get(f"top_{cfg.evaluation.top_k}_accuracy", float("nan")),
                 m["confidence_statistics"]["mean"], thr, cw["pct_auto"],
                 f"{cw['accuracy_auto']:.4f}" if cw["accuracy_auto"] is not None else "n/a")

    # -- artefacts ---------------------------------------------------------
    section("saving artefacts")
    probs = softmax(t_logits)
    preds = predictions_frame(test_df["clean_description"].tolist(), t_labels,
                              test_df["group_id"].to_numpy(), t_logits, class_names,
                              calibrator=None, threshold=thr)
    preds["decision"] = np.where(preds["confidence_uncal"] >= thr,
                                 "automatic", "manual_review")
    preds.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix_frame(probs, t_labels, class_names)
    cm.to_csv(out_dir / "confusion_matrix_test.csv", encoding="utf-8-sig")
    (out_dir / "classification_report_test.txt").write_text(
        test_block["row_level"]["classification_report_text"], encoding="utf-8")

    # Raw logits and probabilities for BOTH held-out splits. Step 5 fits temperature
    # scaling on the validation logits, so they must be persisted here rather than
    # regenerated -- regenerating would require re-running the model.
    np.savez_compressed(
        out_dir / "logits.npz",
        test_logits=t_logits, test_probabilities=probs,
        test_labels=t_labels, test_group_ids=test_df["group_id"].to_numpy(),
        validation_logits=v_logits, validation_probabilities=softmax(v_logits),
        validation_labels=v_labels, validation_group_ids=val_df["group_id"].to_numpy(),
        class_names=np.array(class_names, dtype=object),
    )

    save_json({"class_names": class_names, "label2id": label2id,
               "n_classes": len(class_names)}, out_dir / "label_mapping.json")
    save_json(val_block, out_dir / "validation_metrics.json")
    save_json(test_block, out_dir / "test_metrics.json")
    save_json({
        "model": minfo,
        "tokenizer_diagnostics": diag,
        "device": describe_device(),
        "training": {
            "history": tres.history,
            "total_optimisation_steps": tres.total_steps,
            "warmup_steps": tres.warmup_steps,
            "selected_epoch": tres.best_epoch,
            "seconds": round(seconds, 1),
            "config": cfg.training.__dict__,
            "optimizer": "AdamW",
            "scheduler": "linear warmup then linear decay",
            "loss": "%s%s (gamma=%.1f)" % (
                "class-weighted " if cfg.training.use_class_weights else "unweighted ",
                cfg.training.loss,
                cfg.training.focal_gamma if cfg.training.loss == "focal" else 0.0),
        },
        "data": {
            "dataset_dir": str(args.dataset_dir),
            "frozen_split_sha256": manifest.get("sha256_row_uid_group_split"),
            "n_train_rows": int(len(train_df)),
            "n_train_real_rows": aug_report["n_real_train_rows"],
            "n_train_synthetic_rows": int(len(train_df) - aug_report["n_real_train_rows"]),
            "n_validation_rows": int(len(val_df)),
            "n_validation_groups": int(val_df["group_id"].nunique()),
            "n_test_rows": int(len(test_df)),
            "n_test_groups": int(test_df["group_id"].nunique()),
            "train_class_distribution": {
                str(k): int(v) for k, v in
                train_df[cfg.data.label_column].value_counts().items()},
            "classes_without_training_data": missing_train,
            "augmentation": aug_report,
        },
        "environment": environment_report(),
        "confidence_threshold": thr,
        "calibration": "NOT APPLIED - Step 3 reports raw softmax probabilities only",
    }, out_dir / "model_meta.json")
    cfg.save(out_dir / "config.json")

    if cfg.evaluation.make_plots:
        try:
            plot_confusion(cm, f"Confusion matrix - {args.model} (test, uncalibrated)",
                           out_dir / "confusion_matrix_test.png")
            plot_threshold_curve(
                threshold_sweep(probs, t_labels, cfg.evaluation.threshold_sweep),
                f"Confidence threshold - {args.model} (test, row-level, uncalibrated)",
                out_dir / "threshold_curve_test.png", thr)
        except Exception as exc:
            log.warning("Plotting failed: %s", exc)

    if not args.no_save_weights:
        model.save_pretrained(out_dir / "model")
        tokenizer.save_pretrained(out_dir / "model")
        log.info("Saved weights and tokenizer to %s", out_dir / "model")

    section("step 3 complete")
    log.info("Artefacts in %s", out_dir.resolve())
    log.info("No calibration was applied. No FinBERT comparison was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
