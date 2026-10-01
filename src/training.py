"""Fine-tuning with a weighted focal loss.

Follows Table 3 of the reference paper (lr 2e-5, batch 16, 3 epochs, focal gamma 2,
balanced class weights) with two deliberate corrections documented in AUDIT.md:

  * The focal loss is the mathematically correct form
    ``L = -alpha_t (1 - p_t)^gamma log(p_t)``, with ``p_t`` read from the softmax and
    the class weight applied *outside* the modulation. The prototype's first version
    computed ``p_t = exp(-CE_weighted)``, which is only equal to ``p_t`` when the
    class weight is 1.
  * Warmup defaults to a *ratio* of total steps. The paper's literal ``500`` warmup
    steps exceeds the total step count on a corpus this small, which would leave the
    learning rate ramping for the whole run and never reaching 2e-5. The fixed value
    is still selectable and emits a warning when it is unsafe.

Model selection defaults to ``fixed_epochs`` (no checkpoint picking). That is what
keeps the calibration split completely untouched by any training decision.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .utils import get_logger


class TransactionDataset:
    """Thin torch Dataset over pre-tokenised encodings."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        import torch

        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataset(texts: list[str], labels: list[int], tokenizer: Any, max_length: int):
    enc = tokenizer(
        list(texts), truncation=True, padding="max_length", max_length=max_length
    )
    return TransactionDataset(enc, labels)


def make_focal_loss(class_weights, gamma: float = 2.0):
    """Correct weighted focal loss:  L = -alpha_t (1 - p_t)^gamma log(p_t)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class WeightedFocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma_: float = 2.0):
            super().__init__()
            # `alpha` is a plain tensor already placed on the target device; it is
            # not a learnable parameter and needs no buffer registration.
            self.alpha = alpha
            self.gamma = gamma_

        def forward(self, logits, targets):
            log_probs = F.log_softmax(logits, dim=1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp().clamp(min=1e-8, max=1.0)
            alpha_t = self.alpha[targets] if self.alpha is not None else 1.0
            loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
            return loss.mean()

    return WeightedFocalLoss(class_weights, gamma)


def compute_class_weights(labels: list[int], num_labels: int, device=None):
    """Balanced inverse-frequency weights; classes absent from training get 1.0."""
    import torch

    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(float)
    weights = np.ones(num_labels, dtype=np.float32)
    present = counts > 0
    weights[present] = len(labels) / (present.sum() * counts[present])
    t = torch.tensor(weights, dtype=torch.float32)
    return t.to(device) if device is not None else t


@dataclass
class TrainingResult:
    history: list[dict]
    total_steps: int
    warmup_steps: int
    best_epoch: int | None
    n_train: int
    n_dev: int


def _seed_worker(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    import random

    random.seed(seed)


def train_model(
    model: Any,
    train_dataset: TransactionDataset,
    num_labels: int,
    cfg: TrainingConfig,
    device: Any,
    dev_dataset: TransactionDataset | None = None,
) -> TrainingResult:
    """Fine-tune in place. Returns the training history."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    log = get_logger()
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        generator=generator, worker_init_fn=_seed_worker, num_workers=cfg.num_workers,
        drop_last=False,
    )
    dev_loader = (
        DataLoader(dev_dataset, batch_size=cfg.batch_size, shuffle=False,
                   num_workers=cfg.num_workers)
        if dev_dataset is not None else None
    )

    class_weights = (
        compute_class_weights(train_dataset.labels, num_labels, device)
        if cfg.use_class_weights else None
    )
    # Cross-entropy is exactly focal loss with gamma = 0, so one implementation
    # covers both and the only thing that changes is the focusing exponent.
    gamma = cfg.focal_gamma if getattr(cfg, "loss", "focal") == "focal" else 0.0
    criterion = make_focal_loss(class_weights, gamma)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // max(1, cfg.gradient_accumulation_steps))
    total_steps = steps_per_epoch * cfg.epochs

    if cfg.warmup_mode == "fixed":
        warmup = cfg.warmup_steps
        if warmup >= total_steps:
            log.warning(
                "warmup_steps=%d >= total_steps=%d: the learning rate would never reach "
                "%.0e. Falling back to warmup_ratio=%.2f.",
                warmup, total_steps, cfg.learning_rate, cfg.warmup_ratio,
            )
            warmup = int(cfg.warmup_ratio * total_steps)
    else:
        warmup = int(cfg.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log.info(
        "Training: %d examples | batch %d x accum %d | %d steps/epoch x %d epochs "
        "= %d steps | warmup %d | amp=%s | loss=%s(gamma=%.1f) | class_weights=%s",
        len(train_dataset), cfg.batch_size, cfg.gradient_accumulation_steps,
        steps_per_epoch, cfg.epochs, total_steps, warmup, use_amp,
        getattr(cfg, "loss", "focal"), gamma, cfg.use_class_weights,
    )

    history: list[dict] = []
    best_metric, best_epoch, best_state = -1.0, None, None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n_batches, correct, seen = 0.0, 0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits.float(), labels) / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * cfg.gradient_accumulation_steps
            n_batches += 1
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        rec = {
            "epoch": epoch,
            "train_loss": round(running / max(n_batches, 1), 5),
            "train_accuracy": round(correct / max(seen, 1), 5),
            "lr_end_of_epoch": scheduler.get_last_lr()[0],
        }

        if dev_loader is not None:
            dev = _evaluate_loader(model, dev_loader, criterion, device, use_amp)
            rec.update({"dev_loss": round(dev["loss"], 5),
                        "dev_accuracy": round(dev["accuracy"], 5),
                        "dev_macro_f1": round(dev["macro_f1"], 5)})
            selects = cfg.model_selection in ("synthetic_dev", "validation_best")
            if selects and dev["macro_f1"] > best_metric:
                best_metric, best_epoch = dev["macro_f1"], epoch
                best_state = copy.deepcopy(model.state_dict())

        history.append(rec)
        log.info("  epoch %d/%d  %s", epoch, cfg.epochs,
                 "  ".join(f"{k}={v}" for k, v in rec.items() if k != "epoch"))

    if cfg.model_selection in ("synthetic_dev", "validation_best") and best_state is not None:
        model.load_state_dict(best_state)
        log.info("Restored best checkpoint from epoch %d (dev macro-F1 %.4f)",
                 best_epoch, best_metric)
    else:
        best_epoch = cfg.epochs

    return TrainingResult(
        history=history, total_steps=total_steps, warmup_steps=warmup,
        best_epoch=best_epoch, n_train=len(train_dataset),
        n_dev=len(dev_dataset) if dev_dataset is not None else 0,
    )


def _evaluate_loader(model, loader, criterion, device, use_amp: bool) -> dict:
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    total, n, preds, trues = 0.0, 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            total += criterion(logits.float(), labels).item()
            n += 1
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return {
        "loss": total / max(n, 1),
        "accuracy": float(np.mean(np.array(preds) == np.array(trues))),
        "macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
    }


def predict_logits(model: Any, dataset: TransactionDataset, device: Any,
                   batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (uncalibrated) logits and the gold labels, in dataset order."""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_all.append(out.float().cpu().numpy())
            labels_all.append(batch["labels"].numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def make_synthetic_dev_split(
    train_df: pd.DataFrame, fraction: float, seed: int, label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a dev slice out of the AUGMENTED TRAINING data only.

    Never touches the calibration or test split, so model selection remains isolated
    from both. Used only when ``training.model_selection == "synthetic_dev"``.
    """
    from sklearn.model_selection import train_test_split

    counts = train_df[label_col].value_counts()
    stratify = train_df[label_col] if counts.min() >= 2 else None
    tr, dev = train_test_split(
        train_df, test_size=fraction, random_state=seed, stratify=stratify
    )
    return tr.reset_index(drop=True), dev.reset_index(drop=True)
