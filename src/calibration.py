"""Post-hoc confidence calibration.

Temperature scaling (Guo et al., 2017): a single scalar ``T`` is fitted by minimising
the negative log-likelihood on a held-out calibration set, and applied as
``softmax(z / T)``. Because it is monotone in the logits, scalar temperature scaling
**cannot change any argmax prediction** -- accuracy is identical before and after.
Only the confidence distribution moves. Any claim that temperature scaling improves
accuracy therefore indicates that something other than a scalar temperature was
applied (see the vector variant below).

The reference paper uses ``z / T + b`` with a learnable per-class bias, which *can*
reorder predictions. That variant is implemented as `VectorScaler` and reported as a
clearly-labelled secondary result, because fitting ``|L| + 1`` parameters on a small
calibration set overfits easily -- and on this dataset the calibration set contains
only a handful of independent patterns.

Critical invariant enforced by the pipeline: the calibration set is a group-disjoint
split used for *nothing else*. It is not the test set and, with the default
``model_selection="fixed_epochs"``, it is not used for epoch selection either.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import get_logger


# --------------------------------------------------------------------------- #
#  Calibration metrics
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """Equal-width-bin ECE plus the per-bin table used for reliability diagrams."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": 0,
                         "accuracy": None, "confidence": None, "gap": None})
            continue
        acc = float(correct[mask].mean())
        cf = float(conf[mask].mean())
        ece += (count / n) * abs(acc - cf)
        bins.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": count,
                     "accuracy": acc, "confidence": cf, "gap": float(acc - cf)})
    return float(ece), bins


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    _, bins = expected_calibration_error(probs, labels, n_bins)
    gaps = [abs(b["gap"]) for b in bins if b["count"] > 0]
    return float(max(gaps)) if gaps else 0.0


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error against the one-hot target)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    ece, bins = expected_calibration_error(probs, labels, n_bins)
    conf = probs.max(axis=1)
    return {
        "ece": round(ece, 5),
        "mce": round(maximum_calibration_error(probs, labels, n_bins), 5),
        "nll": round(negative_log_likelihood(probs, labels), 5),
        "brier": round(brier_score(probs, labels), 5),
        "mean_confidence": round(float(conf.mean()), 5),
        "median_confidence": round(float(np.median(conf)), 5),
        "std_confidence": round(float(conf.std()), 5),
        "min_confidence": round(float(conf.min()), 5),
        "max_confidence": round(float(conf.max()), 5),
        "accuracy": round(float(np.mean(probs.argmax(axis=1) == labels)), 5),
        "n_bins": n_bins,
        "reliability_bins": bins,
    }


# --------------------------------------------------------------------------- #
#  Scalers
# --------------------------------------------------------------------------- #
@dataclass
class TemperatureScaler:
    """Single scalar temperature, fitted by LBFGS on calibration-set NLL."""

    temperature: float = 1.0
    fitted: bool = False
    fit_info: dict = field(default_factory=dict)

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            lr: float = 0.05, max_iter: int = 500) -> "TemperatureScaler":
        import torch
        import torch.nn.functional as F

        log = get_logger()
        z = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        # Optimise log(T) so T stays strictly positive.
        log_t = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        nll_before = F.cross_entropy(z, y).item()

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / torch.exp(log_t), y)
            loss.backward()
            return loss

        opt.step(closure)

        self.temperature = float(torch.exp(log_t).item())
        nll_after = F.cross_entropy(z / self.temperature, y).item()
        self.fitted = True
        self.fit_info = {
            "method": "temperature",
            "n_calibration_rows": int(len(labels)),
            "n_parameters": 1,
            "temperature": round(self.temperature, 5),
            "nll_before": round(nll_before, 5),
            "nll_after": round(nll_after, 5),
            "nll_improvement": round(nll_before - nll_after, 5),
        }
        log.info("Temperature scaling: T=%.4f | NLL %.4f -> %.4f (fitted on %d rows)",
                 self.temperature, nll_before, nll_after, len(labels))
        if self.temperature < 1.0:
            log.info("  T < 1 means the model was UNDER-confident on the calibration set; "
                     "calibration sharpens the probabilities.")
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits))

    def to_dict(self) -> dict:
        return {"method": "temperature", "temperature": self.temperature,
                "fit_info": self.fit_info}


@dataclass
class VectorScaler:
    """Paper variant: ``z / T + b`` with one bias per class (|L| + 1 parameters).

    Unlike scalar temperature scaling this *can* change predictions, which is why the
    paper reports an accuracy change from calibration. It also overfits readily on a
    small calibration set -- treat as secondary evidence.
    """

    temperature: float = 1.0
    bias: np.ndarray | None = None
    fitted: bool = False
    fit_info: dict = field(default_factory=dict)

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            lr: float = 0.05, max_iter: int = 500) -> "VectorScaler":
        import torch
        import torch.nn.functional as F

        log = get_logger()
        z = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        n_classes = logits.shape[1]

        log_t = torch.nn.Parameter(torch.zeros(1))
        b = torch.nn.Parameter(torch.zeros(n_classes))
        opt = torch.optim.LBFGS([log_t, b], lr=lr, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        nll_before = F.cross_entropy(z, y).item()

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / torch.exp(log_t) + b, y)
            loss.backward()
            return loss

        opt.step(closure)

        self.temperature = float(torch.exp(log_t).item())
        self.bias = b.detach().numpy().astype(np.float64)
        nll_after = F.cross_entropy(
            z / self.temperature + torch.tensor(self.bias, dtype=torch.float32), y
        ).item()
        self.fitted = True
        n_cal = int(len(labels))
        self.fit_info = {
            "method": "vector",
            "n_calibration_rows": n_cal,
            "n_parameters": int(n_classes + 1),
            "temperature": round(self.temperature, 5),
            "nll_before": round(nll_before, 5),
            "nll_after": round(nll_after, 5),
            "nll_improvement": round(nll_before - nll_after, 5),
            "overfitting_risk": (
                f"{n_classes + 1} free parameters fitted on {n_cal} calibration rows"
            ),
        }
        log.info("Vector scaling (paper variant): T=%.4f + %d biases | NLL %.4f -> %.4f",
                 self.temperature, n_classes, nll_before, nll_after)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        out = logits / self.temperature
        return out + self.bias if self.bias is not None else out

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits))

    def to_dict(self) -> dict:
        return {"method": "vector", "temperature": self.temperature,
                "bias": None if self.bias is None else self.bias.tolist(),
                "fit_info": self.fit_info}


def fit_calibrator(method: str, logits: np.ndarray, labels: np.ndarray,
                   lr: float, max_iter: int):
    if method == "temperature":
        return TemperatureScaler().fit(logits, labels, lr, max_iter)
    if method == "vector":
        return VectorScaler().fit(logits, labels, lr, max_iter)
    raise ValueError(f"Unknown calibration method: {method!r}")


def load_calibrator(d: dict):
    if d["method"] == "temperature":
        s = TemperatureScaler(temperature=float(d["temperature"]), fitted=True)
        s.fit_info = d.get("fit_info", {})
        return s
    s = VectorScaler(
        temperature=float(d["temperature"]),
        bias=np.asarray(d["bias"]) if d.get("bias") is not None else None,
        fitted=True,
    )
    s.fit_info = d.get("fit_info", {})
    return s
