"""Train a candidate model in the same run-directory layout as ``train_model.py``.

This is a thin orchestration of the research code itself: it imports
``train_model.assemble_training_corpus`` and ``train_model.evaluate`` and the
``src.training`` loop, so loss, optimiser, schedule, seeding and metrics are
identical. Two things differ, both needed by the MLOps layer:

* the base checkpoint can be a local directory (tests and CI use a tiny BERT built
  offline; production uses the pinned Hugging Face checkpoint), and
* extra labelled rows (new, validated feedback) can be appended to TRAINING data.

The TEST split is evaluated only when ``evaluate_test=True``. The retraining
workflow never sets it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import softmax
from .config import ExperimentConfig
from .dataset_io import build_label_mapping, load_frozen_split, split_frame
from .models import load_model_and_tokenizer, tokenizer_diagnostics
from .training import build_dataset, predict_logits, train_model
from .utils import describe_device, environment_report, get_device, get_logger, save_json, set_seed


def train_run(base_model: str, out_dir: str | Path, exp_cfg: ExperimentConfig,
              dataset_dir: str | Path = "results/dataset", augmentation_version: str | None = "v2",
              extra_rows: pd.DataFrame | None = None, evaluate_test: bool = False,
              save_weights: bool = True, short_name: str = "germanfinbert") -> dict[str, Any]:
    """Fine-tune ``base_model`` and write a complete run directory. Returns a summary."""
    import train_model as tm  # research script at the repository root

    log = get_logger()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen, manifest = load_frozen_split(dataset_dir, require_checksum=True)
    class_names, label2id = build_label_mapping(frozen, exp_cfg.data.label_column)
    augment = augmentation_version is not None
    exp_cfg.augmentation.enabled = augment
    train_df, aug_report = tm.assemble_training_corpus(frozen, exp_cfg, augment, str(dataset_dir),
                                                       augmentation_version or "v2")
    n_extra = 0
    if extra_rows is not None and len(extra_rows):
        add = extra_rows[["clean_description", exp_cfg.data.label_column]].copy()
        add["is_synthetic"] = False
        train_df = pd.concat([train_df, add], ignore_index=True)
        n_extra = len(add)
    train_df["label"] = train_df[exp_cfg.data.label_column].map(label2id)
    if train_df["label"].isna().any():
        raise ValueError("training rows contain labels outside the taxonomy")
    val_df = split_frame(frozen, "validation")
    val_df["label"] = val_df[exp_cfg.data.label_column].map(label2id)

    set_seed(exp_cfg.training.seed)
    device = get_device()
    model, tokenizer, minfo = load_model_and_tokenizer(base_model, len(class_names),
                                                       {i: c for i, c in enumerate(class_names)})
    minfo["short_name"] = short_name
    diag = tokenizer_diagnostics(tokenizer, train_df["clean_description"].tolist(), exp_cfg.training.max_length)
    train_ds = build_dataset(train_df["clean_description"].tolist(), train_df["label"].astype(int).tolist(),
                             tokenizer, exp_cfg.training.max_length)
    val_ds = build_dataset(val_df["clean_description"].tolist(), val_df["label"].tolist(),
                           tokenizer, exp_cfg.training.max_length)
    t0 = time.time()
    tres = train_model(model, train_ds, len(class_names), exp_cfg.training, device, dev_dataset=val_ds)
    seconds = time.time() - t0
    v_logits, v_labels = predict_logits(model, val_ds, device)
    val_block = tm.evaluate(v_logits, v_labels, val_df["group_id"].to_numpy(), class_names, exp_cfg)
    arrays = {"validation_logits": v_logits, "validation_probabilities": softmax(v_logits),
              "validation_labels": v_labels, "validation_group_ids": val_df["group_id"].to_numpy()}
    if evaluate_test:
        test_df = split_frame(frozen, "test")
        test_df["label"] = test_df[exp_cfg.data.label_column].map(label2id)
        test_ds = build_dataset(test_df["clean_description"].tolist(), test_df["label"].tolist(),
                                tokenizer, exp_cfg.training.max_length)
        t_logits, t_labels = predict_logits(model, test_ds, device)
        save_json(tm.evaluate(t_logits, t_labels, test_df["group_id"].to_numpy(), class_names, exp_cfg),
                  out / "test_metrics.json")
        arrays.update({"test_logits": t_logits, "test_probabilities": softmax(t_logits),
                       "test_labels": t_labels, "test_group_ids": test_df["group_id"].to_numpy()})
    else:
        save_json({"not_evaluated": "test split deliberately not used (candidate model)"},
                  out / "test_metrics.json")
    np.savez_compressed(out / "logits.npz", **arrays)
    save_json({"class_names": class_names, "label2id": label2id, "n_classes": len(class_names)},
              out / "label_mapping.json")
    save_json(val_block, out / "validation_metrics.json")
    save_json({
        "model": minfo, "tokenizer_diagnostics": diag, "device": describe_device(),
        "training": {"history": tres.history, "total_optimisation_steps": tres.total_steps,
                     "warmup_steps": tres.warmup_steps, "seconds": round(seconds, 1),
                     "config": exp_cfg.training.__dict__},
        "data": {"frozen_split_sha256": manifest.get("sha256_row_uid_group_split"),
                 "n_train_rows": int(len(train_df)), "n_extra_feedback_rows": n_extra,
                 "n_validation_rows": int(len(val_df)), "augmentation": aug_report},
        "environment": environment_report(),
        "confidence_threshold": exp_cfg.evaluation.confidence_threshold,
        "test_split_used": bool(evaluate_test),
    }, out / "model_meta.json")
    exp_cfg.save(out / "config.json")
    if save_weights:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    log.info("Candidate run written to %s (%.1fs, %d extra rows)", out, seconds, n_extra)
    return {"run_dir": str(out), "seconds": seconds, "validation": val_block,
            "n_train_rows": int(len(train_df)), "n_extra_rows": n_extra}


def build_tiny_base_model(out_dir: str | Path, dataset_dir: str | Path = "results/dataset",
                          hidden: int = 32, seed: int = 0) -> Path:
    """Create a TINY randomly initialised BERT + WordPiece tokenizer, fully offline.

    For tests and CI only. Its vocabulary comes from the cleaned TRAINING texts (and
    corpus v2). It is not the production model and must never be presented as one.
    """
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen, _ = load_frozen_split(dataset_dir)
    class_names, _ = build_label_mapping(frozen)
    texts = split_frame(frozen, "train")["clean_description"].astype(str).tolist()
    corpus = Path(dataset_dir) / "augmentation_corpus_v2.csv"
    if corpus.exists():
        texts += pd.read_csv(corpus, encoding="utf-8-sig")["clean_description"].astype(str).tolist()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({t for x in texts for t in x.split()})
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)
    torch.manual_seed(seed)
    conf = BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=hidden * 2, max_position_embeddings=128, num_labels=len(class_names),
                      id2label=dict(enumerate(class_names)), label2id={c: i for i, c in enumerate(class_names)})
    BertForSequenceClassification(conf).save_pretrained(out)
    tok.save_pretrained(out)
    return out
