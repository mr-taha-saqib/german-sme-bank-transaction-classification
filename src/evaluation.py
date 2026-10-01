"""Evaluation, confidence workflow and reporting.

Two views of every metric are produced:

**row-level**    one observation per transaction. Comparable to the prototype and to
                 the reference paper, but optimistic about precision: this dataset has
                 ~12 rows per distinct pattern, so 590 test rows are not 590
                 independent observations.

**group-level**  one observation per independent pattern (the first row of each
                 group). This is the honest effective sample size and is reported as
                 the headline. The number of independent patterns is printed next to
                 every group-level figure.

The confidence workflow implements the agreed decision rule:

    confidence >= threshold  ->  automatic processing
    confidence <  threshold  ->  manual review

A high-confidence accuracy figure is an *observed conditional accuracy on this test
set*, never a guarantee that an individual prediction above the threshold is correct.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .calibration import calibration_metrics, softmax
from .utils import get_logger


# --------------------------------------------------------------------------- #
#  Core metrics
# --------------------------------------------------------------------------- #
def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 2) -> float:
    if probs.shape[1] < k:
        k = probs.shape[1]
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([lab in row for lab, row in zip(labels, topk)]))


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    top_k: int = 2,
) -> dict:
    preds = probs.argmax(axis=1)
    present = sorted(set(labels.tolist()) | set(preds.tolist()))

    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=present, zero_division=0
    )
    per_class = {
        class_names[c]: {
            "precision": round(float(p), 5),
            "recall": round(float(r), 5),
            "f1": round(float(f), 5),
            "support": int(s),
        }
        for c, p, r, f, s in zip(present, prec, rec, f1, support)
    }

    return {
        "n": int(len(labels)),
        "n_classes_present": len(set(labels.tolist())),
        "accuracy": round(float(accuracy_score(labels, preds)), 5),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 5),
        "weighted_f1": round(
            float(f1_score(labels, preds, average="weighted", zero_division=0)), 5
        ),
        f"top_{top_k}_accuracy": round(top_k_accuracy(probs, labels, top_k), 5),
        "per_class": per_class,
        "classification_report_text": classification_report(
            labels, preds, labels=present,
            target_names=[class_names[c] for c in present], zero_division=0, digits=4,
        ),
    }


def confusion_matrix_frame(
    probs: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    present = sorted(set(labels.tolist()) | set(preds.tolist()))
    cm = confusion_matrix(labels, preds, labels=present)
    names = [class_names[c] for c in present]
    return pd.DataFrame(cm, index=pd.Index(names, name="true"),
                        columns=pd.Index(names, name="predicted"))


# --------------------------------------------------------------------------- #
#  Confidence-based decision workflow
# --------------------------------------------------------------------------- #
def confidence_workflow(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.80
) -> dict:
    """Split predictions into automatic vs manual-review at `threshold`."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels

    auto = conf >= threshold
    review = ~auto
    n = len(labels)

    def _acc(mask):
        return round(float(correct[mask].mean()), 5) if mask.any() else None

    return {
        "threshold": threshold,
        "n_total": int(n),
        "n_auto": int(auto.sum()),
        "pct_auto": round(float(auto.mean()) * 100, 3),
        "accuracy_auto": _acc(auto),
        "n_errors_in_auto": int((~correct & auto).sum()),
        "n_manual_review": int(review.sum()),
        "pct_manual_review": round(float(review.mean()) * 100, 3),
        "accuracy_manual_review": _acc(review),
        "overall_accuracy": round(float(correct.mean()), 5),
        "interpretation": (
            f"{int(auto.sum())}/{n} predictions ({auto.mean() * 100:.1f}%) reach "
            f"confidence >= {threshold} and would be auto-processed; of those, "
            f"{int((~correct & auto).sum())} are wrong. This is an observed "
            f"conditional accuracy on this test set, NOT a guarantee for an "
            f"individual prediction."
        ),
    }


def threshold_sweep(probs: np.ndarray, labels: np.ndarray,
                    thresholds: list[float]) -> pd.DataFrame:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    rows = []
    for t in thresholds:
        mask = conf >= t
        rows.append({
            "threshold": t,
            "n_auto": int(mask.sum()),
            "coverage_pct": round(float(mask.mean()) * 100, 2),
            "accuracy_auto": round(float(correct[mask].mean()), 5) if mask.any() else np.nan,
            "n_manual_review": int((~mask).sum()),
            "manual_review_pct": round(float((~mask).mean()) * 100, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Group-level view
# --------------------------------------------------------------------------- #
def to_group_level(
    probs: np.ndarray, labels: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce to one observation per group.

    Rows inside a group share an identical cleaned text under `exact`/`token_set`
    grouping, so their probabilities are identical and taking the first row is exact.
    Under `near_dup` a group can hold slightly different texts, so probabilities are
    averaged within the group -- documented, and it only affects that mode.
    """
    order = np.argsort(group_ids, kind="stable")
    g = group_ids[order]
    p = probs[order]
    y = labels[order]

    uniq, start = np.unique(g, return_index=True)
    ends = np.append(start[1:], len(g))
    out_p = np.stack([p[s:e].mean(axis=0) for s, e in zip(start, ends)])
    out_y = np.array([y[s] for s in start])
    return out_p, out_y, uniq


# --------------------------------------------------------------------------- #
#  Full evaluation of one (model, calibration) pair
# --------------------------------------------------------------------------- #
def evaluate_split(
    logits: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    class_names: list[str],
    threshold: float = 0.80,
    top_k: int = 2,
    ece_bins: int = 15,
    thresholds: list[float] | None = None,
    calibrator=None,
) -> dict:
    """Evaluate uncalibrated and (optionally) calibrated probabilities."""
    result: dict = {}

    variants = {"uncalibrated": softmax(logits)}
    if calibrator is not None:
        variants["calibrated"] = calibrator.predict_proba(logits)

    for name, probs in variants.items():
        block: dict = {"row_level": {}, "group_level": {}}

        block["row_level"].update(classification_metrics(probs, labels, class_names, top_k))
        block["row_level"]["calibration"] = calibration_metrics(probs, labels, ece_bins)
        block["row_level"]["confidence_workflow"] = confidence_workflow(probs, labels, threshold)
        if thresholds:
            block["row_level"]["threshold_sweep"] = threshold_sweep(
                probs, labels, thresholds
            ).to_dict(orient="records")

        gp, gy, guniq = to_group_level(probs, labels, group_ids)
        block["group_level"].update(classification_metrics(gp, gy, class_names, top_k))
        block["group_level"]["n_independent_patterns"] = int(len(guniq))
        block["group_level"]["calibration"] = calibration_metrics(gp, gy, ece_bins)
        block["group_level"]["confidence_workflow"] = confidence_workflow(gp, gy, threshold)
        if thresholds:
            block["group_level"]["threshold_sweep"] = threshold_sweep(
                gp, gy, thresholds
            ).to_dict(orient="records")

        result[name] = block

    return result


def predictions_frame(
    texts: list[str],
    labels: np.ndarray,
    group_ids: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
    calibrator=None,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Per-example predictions -- the audit trail behind every reported number."""
    probs_u = softmax(logits)
    df = pd.DataFrame({
        "text": texts,
        "group_id": group_ids,
        "true_label_id": labels,
        "true_category": [class_names[i] for i in labels],
        "pred_id_uncal": probs_u.argmax(axis=1),
        "pred_category_uncal": [class_names[i] for i in probs_u.argmax(axis=1)],
        "confidence_uncal": probs_u.max(axis=1),
    })
    df["correct_uncal"] = df["true_label_id"] == df["pred_id_uncal"]

    if calibrator is not None:
        probs_c = calibrator.predict_proba(logits)
        df["pred_id_cal"] = probs_c.argmax(axis=1)
        df["pred_category_cal"] = [class_names[i] for i in probs_c.argmax(axis=1)]
        df["confidence_cal"] = probs_c.max(axis=1)
        df["correct_cal"] = df["true_label_id"] == df["pred_id_cal"]
        df["decision"] = np.where(df["confidence_cal"] >= threshold,
                                  "automatic", "manual_review")
        top2 = np.argsort(-probs_c, axis=1)[:, :2]
        df["second_choice_category"] = [class_names[r[1]] for r in top2]
    return df


# --------------------------------------------------------------------------- #
#  Comparison table
# --------------------------------------------------------------------------- #
def build_comparison_table(all_results: dict, level: str = "group_level") -> pd.DataFrame:
    """The single headline table required by the brief.

    `all_results` maps model name -> {"test": <evaluate_split output>, ...}.
    """
    rows = []
    for model_name, per_split in all_results.items():
        for split_name in ("validation", "test"):
            block = per_split.get(split_name)
            if not block:
                continue
            for variant in ("uncalibrated", "calibrated"):
                if variant not in block:
                    continue
                m = block[variant][level]
                cal = m["calibration"]
                cw = m["confidence_workflow"]
                topk_key = next((k for k in m if k.startswith("top_")), None)
                rows.append({
                    "Model": model_name,
                    "Split": split_name,
                    "Probabilities": variant,
                    "N": m["n"],
                    "Accuracy": m["accuracy"],
                    "Macro F1": m["macro_f1"],
                    "Weighted F1": m["weighted_f1"],
                    "Top-2 Accuracy": m.get(topk_key),
                    "ECE": cal["ece"],
                    "NLL": cal["nll"],
                    "Brier": cal["brier"],
                    "Mean Conf.": cal["mean_confidence"],
                    "High-Conf Accuracy": cw["accuracy_auto"],
                    "Auto-Processing Rate %": cw["pct_auto"],
                    "Manual Review %": cw["pct_manual_review"],
                })
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no results)"
    return df.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.4f}")


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #
def plot_reliability(bins_uncal: list[dict], bins_cal: list[dict] | None,
                     title: str, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if bins_cal else 1, figsize=(11 if bins_cal else 6, 4.6),
                             squeeze=False)
    for ax, bins, sub in zip(
        axes[0], [bins_uncal, bins_cal] if bins_cal else [bins_uncal],
        ["Uncalibrated", "Calibrated"] if bins_cal else ["Uncalibrated"],
    ):
        xs = [b["confidence"] for b in bins if b["count"] > 0]
        ys = [b["accuracy"] for b in bins if b["count"] > 0]
        ns = [b["count"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
        if xs:
            ax.scatter(xs, ys, s=[20 + 4 * n for n in ns], alpha=0.75,
                       edgecolor="black", linewidth=0.5)
            ax.plot(xs, ys, lw=1, alpha=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(sub)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: pd.DataFrame, title: str, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cm)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.34), max(5, n * 0.30)))
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cm.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(cm.index, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(sweep: pd.DataFrame, title: str, path: str | Path,
                         threshold: float = 0.80) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sweep["threshold"], sweep["accuracy_auto"], "o-", label="accuracy above threshold")
    ax.plot(sweep["threshold"], sweep["coverage_pct"] / 100, "s--",
            label="coverage (auto-processing rate)")
    ax.axvline(threshold, color="crimson", lw=1, ls=":", label=f"operating point {threshold}")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def log_headline(model_name: str, block: dict, threshold: float) -> None:
    log = get_logger()
    for variant in ("uncalibrated", "calibrated"):
        if variant not in block:
            continue
        g = block[variant]["group_level"]
        r = block[variant]["row_level"]
        cw = g["confidence_workflow"]
        log.info(
            "%-14s %-12s | group-level: acc %.4f  macroF1 %.4f  top2 %.4f  (n=%d patterns) "
            "| row-level acc %.4f (n=%d) | ECE %.4f NLL %.4f | >=%.2f: %.1f%% auto, acc %s",
            model_name, variant, g["accuracy"], g["macro_f1"],
            g.get("top_2_accuracy", float("nan")), g["n"], r["accuracy"], r["n"],
            g["calibration"]["ece"], g["calibration"]["nll"],
            threshold, cw["pct_auto"],
            f"{cw['accuracy_auto']:.4f}" if cw["accuracy_auto"] is not None else "n/a",
        )
