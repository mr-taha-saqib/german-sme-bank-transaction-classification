"""Operational and model monitoring for the transaction-classification API.

Privacy: raw transaction descriptions are NEVER stored or logged. The monitor keeps
only derived, non-reversible numbers per request (character-length bucket, token
count bucket, share of tokens unseen in training, digit share bucket) plus the
prediction (category, confidence, decision). Prediction ids are random UUIDs.

Signals are kept apart:

* input drift       PSI of description length / token-count / unseen-token buckets
                    against the validation-split reference;
* prediction drift  PSI of the predicted-category distribution and the confidence
                    histogram against the approved model's validation predictions;
* performance       accuracy against DELAYED ground-truth labels sent to
                    ``POST /feedback``; reported only once enough labels exist.
                    Drift is never reported as performance degradation.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, Histogram
from prometheus_client import Counter as PCounter

logger = logging.getLogger(__name__)

EPS = 1e-4
CHAR_EDGES = (0, 10, 20, 30, 40, 60, 10 ** 9)
TOKEN_EDGES = (0, 2, 3, 4, 5, 7, 10 ** 9)
UNSEEN_EDGES = (0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 + 1e-9)
DIGIT_EDGES = (0.0, 1e-9, 0.1, 0.2, 0.4, 1.0 + 1e-9)
CONF_EDGES = tuple(np.round(np.linspace(0, 1, 11), 2)[:-1].tolist()) + (1.0 + 1e-9,)
_DIGIT = re.compile(r"\d")


def _bucket(value: float, edges: tuple) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def text_features(raw: str, cleaned: str, vocabulary: set[str]) -> dict[str, int]:
    """Non-reversible numeric features of one description (bucket indices)."""
    tokens = cleaned.split()
    unseen = (sum(1 for t in tokens if t not in vocabulary) / len(tokens)) if tokens else 1.0
    digits = len(_DIGIT.findall(raw)) / max(len(raw), 1)
    return {"char_bucket": _bucket(len(raw), CHAR_EDGES),
            "token_bucket": _bucket(len(tokens), TOKEN_EDGES),
            "unseen_bucket": _bucket(unseen, UNSEEN_EDGES),
            "digit_bucket": _bucket(digits, DIGIT_EDGES),
            "empty_after_cleaning": int(cleaned in ("", "nodescription"))}


FEATURE_BINS = {"char_bucket": len(CHAR_EDGES) - 1, "token_bucket": len(TOKEN_EDGES) - 1,
                "unseen_bucket": len(UNSEEN_EDGES) - 1, "digit_bucket": len(DIGIT_EDGES) - 1}


def histogram(values: list[int], n_bins: int) -> list[float]:
    h = np.bincount(np.asarray(values, dtype=np.int64), minlength=n_bins)[:n_bins].astype(float)
    return (h / h.sum()).tolist() if h.sum() else h.tolist()


def psi(expected: list[float] | np.ndarray, actual: list[float] | np.ndarray) -> float:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.sum() <= 0 or a.sum() <= 0:
        return float("nan")
    e = np.clip(e / e.sum(), EPS, None)
    a = np.clip(a / a.sum(), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_status(value: float, n: int, min_n: int, warn: float, alert: float) -> str:
    if n < min_n or math.isnan(value):
        return "insufficient_data"
    return "alert" if value >= alert else "warning" if value >= warn else "ok"


def build_reference_profile(raw_texts: list[str], cleaned: list[str], label_idx: list[int], n_classes: int,
                            heldout_probs: np.ndarray, vocabulary: set[str], threshold: float) -> dict[str, Any]:
    """Reference distributions for drift checks, from HELD-OUT real data.

    New production transactions are, by definition, patterns the model did not train
    on, so the reference is the held-out population (validation + test rows of the
    frozen split): their description features (input drift), this model's PREDICTED
    category mix and uncalibrated confidence on them (prediction drift: predictions are
    compared with predictions, never with true labels), and the true category mix
    (stored for information). Training rows would be the wrong
    reference (seen patterns, near-certain predictions); the validation split alone
    (16 groups, 16 of 37 classes) is too narrow. This is a monitoring baseline only;
    it is never used for model selection.
    """
    feats = [text_features(r, c, vocabulary) for r, c in zip(raw_texts, cleaned)]
    conf = heldout_probs.max(axis=1)
    return {
        "source": {"features_and_categories": f"{len(raw_texts)} held-out original descriptions (validation + test)",
                   "predictions_and_confidence": f"{len(conf)} held-out predictions of this model (uncalibrated)"},
        "n": len(raw_texts),
        "features": {k: histogram([f[k] for f in feats], n) for k, n in FEATURE_BINS.items()},
        "category_distribution": histogram(heldout_probs.argmax(axis=1).tolist(), n_classes),
        "true_category_distribution": histogram(list(label_idx), n_classes),
        "confidence_distribution": histogram([_bucket(c, CONF_EDGES) for c in conf], len(CONF_EDGES) - 1),
        "auto_rate": float((conf >= threshold).mean()),
        "edges": {"char": list(CHAR_EDGES[:-1]), "token": list(TOKEN_EDGES[:-1]),
                  "unseen": list(UNSEEN_EDGES[:-1]), "digit": list(DIGIT_EDGES[:-1]),
                  "confidence": list(CONF_EDGES[:-1])},
    }


class TransactionMonitor:
    """In-process monitor; one per API process. Thread-safe."""

    def __init__(self, reference: dict[str, Any] | None, cfg: dict[str, Any], model_version: str,
                 class_names: list[str], threshold: float, log_dir: Path | None = None):
        self.reference = reference or {}
        self.class_names = list(class_names)
        self.threshold = float(threshold)
        self.window = int(cfg.get("window_size", 5000))
        self.psi_warn = float(cfg.get("psi_warning", 0.10))
        self.psi_alert = float(cfg.get("psi_alert", 0.25))
        self.min_requests = int(cfg.get("min_requests_for_drift", 50))
        self.min_labels = int(cfg.get("min_labels_for_performance", 30))
        self.model_version = model_version
        self.log_path = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log_path = Path(log_dir) / "events.jsonl"
        self._lock = threading.Lock()
        self.started = time.time()
        self._preds: deque[dict[str, Any]] = deque(maxlen=self.window)
        self._latency: deque[float] = deque(maxlen=self.window)
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()   # prediction_id -> prediction
        self._labelled: deque[dict[str, Any]] = deque(maxlen=self.window)
        self.totals: Counter[str] = Counter()

        self.registry = CollectorRegistry()
        self.m_requests = PCounter("txclf_requests_total", "API requests", ["endpoint", "status"],
                                   registry=self.registry)
        self.m_latency = Histogram("txclf_request_latency_seconds", "Request latency", ["endpoint"],
                                   buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
                                   registry=self.registry)
        self.m_predictions = PCounter("txclf_predictions_total", "Predictions by decision", ["decision"],
                                      registry=self.registry)
        self.m_category = PCounter("txclf_predicted_category_total", "Predictions by category", ["category"],
                                   registry=self.registry)
        self.m_conf = Histogram("txclf_confidence", "Prediction confidence",
                                buckets=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0), registry=self.registry)
        self.m_errors = PCounter("txclf_errors_total", "Errors by type", ["type"], registry=self.registry)
        self.m_labels = PCounter("txclf_feedback_labels_total", "Delayed ground-truth labels", ["outcome"],
                                 registry=self.registry)
        self.m_model = Gauge("txclf_model_info", "Loaded model version", ["version"], registry=self.registry)
        self.m_psi = Gauge("txclf_drift_psi", "Latest PSI value", ["signal"], registry=self.registry)
        self.m_model.labels(version=model_version).set(1)

    # ── recording ─────────────────────────────────────────────────────────────
    def _append(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_request(self, endpoint: str, status: int, latency_s: float) -> None:
        self.m_requests.labels(endpoint=endpoint, status=str(status)).inc()
        self.m_latency.labels(endpoint=endpoint).observe(latency_s)
        with self._lock:
            self.totals["requests"] += 1
            self._latency.append(latency_s)
            if status >= 500:
                self.totals["server_errors"] += 1
            elif status >= 400:
                self.totals["client_errors"] += 1

    def record_error(self, kind: str) -> None:
        self.m_errors.labels(type=kind).inc()
        with self._lock:
            self.totals[f"error:{kind}"] += 1

    def record_prediction(self, prediction_id: str, features: dict[str, int], category_idx: int,
                          confidence: float, decision: str) -> None:
        cat = self.class_names[category_idx]
        self.m_predictions.labels(decision=decision).inc()
        self.m_category.labels(category=cat).inc()
        self.m_conf.observe(confidence)
        rec = {"features": features, "category_idx": int(category_idx), "confidence": float(confidence),
               "decision": decision}
        with self._lock:
            self.totals["predictions"] += 1
            self.totals[f"decision:{decision}"] += 1
            if features.get("empty_after_cleaning"):
                self.totals["empty_after_cleaning"] += 1
            self._preds.append(rec)
            self._pending[prediction_id] = rec
            while len(self._pending) > self.window:
                self._pending.popitem(last=False)
        self._append({"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": "prediction",
                      "prediction_id": prediction_id, "model_version": self.model_version,
                      "category": cat, "confidence": round(float(confidence), 5), "decision": decision,
                      **{k: int(v) for k, v in features.items()}})

    def record_label(self, prediction_id: str, true_category: str) -> str:
        """Attach a delayed ground-truth label. Returns 'matched' or 'unknown_prediction'."""
        with self._lock:
            rec = self._pending.pop(prediction_id, None)
        if rec is None:
            self.m_labels.labels(outcome="unknown_prediction").inc()
            return "unknown_prediction"
        correct = self.class_names[rec["category_idx"]] == true_category
        with self._lock:
            self.totals["labels"] += 1
            self._labelled.append({"correct": correct, "decision": rec["decision"]})
        self.m_labels.labels(outcome="matched").inc()
        self._append({"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": "label",
                      "prediction_id": prediction_id, "model_version": self.model_version,
                      "correct": bool(correct), "decision": rec["decision"]})
        return "matched"

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        with self._lock:
            preds = list(self._preds)
            labelled = list(self._labelled)
            lat = np.asarray(self._latency, dtype=float)
            totals = dict(self.totals)
        n = len(preds)
        ref = self.reference
        input_drift: dict[str, Any] = {}
        for feat, n_bins in FEATURE_BINS.items():
            actual = np.bincount([p["features"][feat] for p in preds], minlength=n_bins)[:n_bins] if n else np.zeros(n_bins)
            val = psi(ref.get("features", {}).get(feat, []), actual) if n else float("nan")
            input_drift[feat] = {"psi": None if math.isnan(val) else round(val, 4),
                                 "status": drift_status(val, n, self.min_requests, self.psi_warn, self.psi_alert)}
            if not math.isnan(val):
                self.m_psi.labels(signal=f"input_{feat}").set(val)
        n_cls = len(self.class_names)
        cat_actual = np.bincount([p["category_idx"] for p in preds], minlength=n_cls)[:n_cls] if n else np.zeros(n_cls)
        conf_actual = (np.bincount([_bucket(p["confidence"], CONF_EDGES) for p in preds],
                                   minlength=len(CONF_EDGES) - 1) if n else np.zeros(len(CONF_EDGES) - 1))
        cat_psi = psi(ref.get("category_distribution", []), cat_actual) if n else float("nan")
        conf_psi = psi(ref.get("confidence_distribution", []), conf_actual) if n else float("nan")
        for name, v in (("prediction_category", cat_psi), ("prediction_confidence", conf_psi)):
            if not math.isnan(v):
                self.m_psi.labels(signal=name).set(v)
        auto = sum(1 for p in preds if p["decision"] == "automatic")
        if len(labelled) >= self.min_labels:
            auto_l = [x for x in labelled if x["decision"] == "automatic"]
            perf = {"status": "available", "n_labelled": len(labelled),
                    "accuracy": round(float(np.mean([x["correct"] for x in labelled])), 4),
                    "accuracy_automatic": (round(float(np.mean([x["correct"] for x in auto_l])), 4)
                                           if auto_l else None),
                    "n_labelled_automatic": len(auto_l),
                    "note": "From delayed ground-truth labels. Labels may be a biased sample "
                            "(e.g. mostly manual-review cases)."}
        else:
            perf = {"status": "insufficient_labels", "n_labelled": len(labelled), "required": self.min_labels,
                    "note": "No performance claim is made without enough labelled outcomes."}
        top_cats = sorted(((self.class_names[i], int(c)) for i, c in enumerate(cat_actual) if c),
                          key=lambda x: -x[1])[:10]
        return {
            "model_version": self.model_version,
            "uptime_seconds": round(time.time() - self.started, 1),
            "predictions_in_window": n,
            "totals": totals,
            "latency_ms": {q: (round(float(np.percentile(lat, int(q[1:]))) * 1000, 2) if len(lat) else None)
                           for q in ("p50", "p95", "p99")},
            "decisions": {"threshold": self.threshold, "automatic": auto, "manual_review": n - auto,
                          "manual_review_rate": round((n - auto) / n, 4) if n else None,
                          "reference_auto_rate": ref.get("auto_rate")},
            "confidence": {"mean": round(float(np.mean([p["confidence"] for p in preds])), 4) if n else None,
                           "histogram": [int(x) for x in conf_actual], "bin_edges": ref.get("edges", {}).get("confidence")},
            "category_distribution_top10": top_cats,
            "input_drift": {"signal": "description characteristics vs validation reference (PSI)", **input_drift},
            "prediction_drift": {
                "category": {"psi": None if math.isnan(cat_psi) else round(cat_psi, 4),
                             "status": drift_status(cat_psi, n, self.min_requests, self.psi_warn, self.psi_alert)},
                "confidence": {"psi": None if math.isnan(conf_psi) else round(conf_psi, 4),
                               "status": drift_status(conf_psi, n, self.min_requests, self.psi_warn, self.psi_alert)},
            },
            "observed_performance": perf,
            "data_quality": {"empty_after_cleaning": totals.get("empty_after_cleaning", 0),
                             "validation_errors": totals.get("error:validation", 0),
                             "batch_item_errors": totals.get("error:batch_item", 0)},
            "privacy": "raw descriptions are not stored; only bucketed numeric features",
            "thresholds": {"psi_warning": self.psi_warn, "psi_alert": self.psi_alert,
                           "min_requests_for_drift": self.min_requests,
                           "min_labels_for_performance": self.min_labels},
        }
