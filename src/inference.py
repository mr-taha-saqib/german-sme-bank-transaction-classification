"""Inference on new transaction descriptions with the confidence workflow applied.

Loads a trained model and applies the same preprocessing and confidence workflow used
in evaluation. Confidences are UNCALIBRATED by default, which is the primary operational
result in this proof of concept: temperature scaling improved validation likelihood but
did not generalise to the test set (see STEP5_CALIBRATION.md). Pass --calibration with a
fitted temperature file to apply one anyway. Usage:

    python -m src.inference --run results/step4_comparison/germanfinbert_seed42 \\
        --text "SaaS Subscription 2025-04" --text "Gehaltslauf 2025-06"

    python -m src.inference --run results/step4_comparison/germanfinbert_seed42 \\
        --csv new_transactions.csv --column description --out predictions.csv

The run directory must contain trained weights, which train_model.py writes to
<run>/model/. Weights are not shipped with this package, so train once first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import load_calibrator, softmax
from .config import ExperimentConfig, PreprocessingConfig
from .preprocessing import clean_description
from .utils import get_device, load_json, setup_logging


def _find_upwards(start: Path, filename: str, levels: int = 2) -> Path | None:
    """Locate `filename` at `start` or up to `levels` directories above it.

    Two run layouts exist: train_model.py writes weights to `<run>/model/` and the
    metadata to `<run>/`, while the earlier prototype used `<run>/models/<name>/`.
    Searching upwards resolves both without the caller needing to know which.
    """
    node = start
    for _ in range(levels + 1):
        candidate = node / filename
        if candidate.exists():
            return candidate
        node = node.parent
    return None


def _has_weights(d: Path) -> bool:
    return (d / "config.json").exists() and (
        any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists()
    )


def resolve_model_dir(run: Path, name: str = "model") -> Path:
    """Return the directory holding the weights, supporting both run layouts."""
    seen: list[Path] = []
    for candidate in (run / "model", run / name, run / "models" / name, run):
        if candidate not in seen:
            seen.append(candidate)
        if _has_weights(candidate):
            return candidate
    looked = ", ".join(str(c) for c in seen)
    raise SystemExit(
        "No trained weights under " + str(run) + "." + chr(10)
        + "Looked in: " + looked + chr(10)
        + "Model weights are not shipped with this package (~420 MB each). "
        + "Train one first, for example:" + chr(10)
        + "  python train_model.py --model germanfinbert --seed 42 --augment "
        + "--run-name my_run" + chr(10)
        + "then pass --run results/my_run"
    )


def _load_calibration_file(path: Path):
    """Build a calibrator from either a saved calibrator or a Step-5 result file."""
    d = load_json(path)
    if "method" not in d:
        d = dict(d, method="temperature")
    return load_calibrator(d)



class TransactionClassifier:
    """Trained model + calibrator + preprocessing, as one reusable object."""

    def __init__(self, model_dir: str | Path, threshold: float = 0.80,
                 calibration: str | Path | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        meta_path = _find_upwards(self.model_dir, "model_meta.json")
        if meta_path is None:
            raise SystemExit(
                "No model_meta.json at or above " + str(self.model_dir) + ". "
                "It records the class names and preprocessing settings, and is "
                "written next to the run by train_model.py."
            )
        meta = load_json(meta_path)
        run_dir = meta_path.parent

        # Class names, max_length and the preprocessing settings live in the run
        # directory alongside model_meta.json. The prototype kept them all inside
        # model_meta.json, so fall back to that shape when the files are absent.
        label_path = _find_upwards(self.model_dir, "label_mapping.json")
        if label_path is not None:
            self.class_names: list[str] = load_json(label_path)["class_names"]
        else:
            self.class_names = meta["class_names"]

        cfg_path = run_dir / "config.json"
        run_cfg = load_json(cfg_path) if cfg_path.exists() else {}
        pre = run_cfg.get("preprocessing", meta.get("preprocessing"))
        if pre is None:
            raise SystemExit(
                "No preprocessing settings found for " + str(run_dir) + ". "
                "Inference must clean text exactly as training did, so this is fatal."
            )
        self.pre_cfg = PreprocessingConfig(**pre)

        self.max_length: int = int(
            run_cfg.get("training", {}).get("max_length", meta.get("max_length", 128))
        )
        self.threshold = float(meta.get("confidence_threshold", threshold))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.device = get_device()
        self.model.to(self.device).eval()

        self.calibrator = None
        if calibration is not None:
            self.calibrator = _load_calibration_file(Path(calibration))
        else:
            cal_path = _find_upwards(self.model_dir, "calibrator.json")
            if cal_path is not None:
                self.calibrator = _load_calibration_file(cal_path)

    def predict(self, texts: list[str], batch_size: int = 64) -> pd.DataFrame:
        import torch

        cleaned = [clean_description(t, self.pre_cfg) for t in texts]
        logits: list[np.ndarray] = []
        for i in range(0, len(cleaned), batch_size):
            enc = self.tokenizer(
                cleaned[i:i + batch_size], truncation=True,
                padding="max_length", max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits.append(self.model(**enc).logits.float().cpu().numpy())
        z = np.concatenate(logits, axis=0)

        probs_u = softmax(z)
        probs = self.calibrator.predict_proba(z) if self.calibrator else probs_u

        order = np.argsort(-probs, axis=1)
        conf = probs.max(axis=1)
        return pd.DataFrame({
            "input": texts,
            "cleaned": cleaned,
            "predicted_category": [self.class_names[i] for i in order[:, 0]],
            "confidence": np.round(conf, 5),
            "second_choice": [self.class_names[i] for i in order[:, 1]]
            if probs.shape[1] > 1 else None,
            "second_confidence": np.round(probs[np.arange(len(probs)), order[:, 1]], 5)
            if probs.shape[1] > 1 else None,
            "confidence_uncalibrated": np.round(probs_u.max(axis=1), 5),
            "decision": np.where(conf >= self.threshold, "automatic", "manual_review"),
        })


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify German bank transaction descriptions")
    ap.add_argument("--run", required=True,
                    help="run directory, e.g. results/step4_comparison/germanfinbert_seed42")
    ap.add_argument("--model", default="model", help="weights subdirectory name")
    ap.add_argument("--calibration", help="calibration JSON (defaults to the run's, if any)")
    ap.add_argument("--text", action="append", default=[], help="description (repeatable)")
    ap.add_argument("--csv", help="CSV of descriptions to classify")
    ap.add_argument("--column", default="description", help="column in --csv")
    ap.add_argument("--out", help="write predictions to this CSV")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    setup_logging()
    model_dir = resolve_model_dir(Path(args.run), args.model)

    cfg_path = Path(args.run) / "config.json"
    threshold = args.threshold
    if threshold is None and cfg_path.exists():
        try:
            threshold = ExperimentConfig.load(cfg_path).evaluation.confidence_threshold
        except (TypeError, KeyError, ValueError):
            # A run written under an older config schema. The threshold is also
            # recorded in model_meta.json, which the classifier reads, so leave it
            # unset rather than refusing to run.
            threshold = None
    clf = TransactionClassifier(model_dir, threshold or 0.80, args.calibration)

    if args.csv:
        texts = pd.read_csv(args.csv)[args.column].astype(str).tolist()
    elif args.text:
        texts = args.text
    else:
        raise SystemExit("Provide --text (repeatable) or --csv")

    out = clf.predict(texts)
    if args.out:
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(out)} predictions to {args.out}")
    else:
        with pd.option_context("display.width", 200, "display.max_colwidth", 44):
            print(out.to_string(index=False))
    n_auto = int((out["decision"] == "automatic").sum())
    print(f"\nThreshold {clf.threshold}: {n_auto}/{len(out)} automatic, "
          f"{len(out) - n_auto} routed to manual review.")
    print("Note: confidence >= threshold is a routing heuristic, not a correctness "
          "guarantee.")
    if clf.calibrator is None:
        print("Confidences are uncalibrated (the primary operational result). "
              "Pass --calibration to apply a fitted temperature.")
    else:
        print(f"Confidences are calibrated with T={clf.calibrator.temperature:.4f}.")


if __name__ == "__main__":
    main()
