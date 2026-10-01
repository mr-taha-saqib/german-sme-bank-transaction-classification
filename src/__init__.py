"""Leakage-safe German SME transaction classification (bachelor-thesis proof of concept).

Modules
-------
config            experiment configuration dataclasses and the model registry
data_preparation  loading the handover workbook and the LLM synthetic corpora
preprocessing     German-aware, conservative text cleaning (T0-T13)
grouping          deterministic grouping of structurally equivalent descriptions
splitting         group-based train/calibration/test split + leakage validation
models            model and tokenizer loading (GermanFinBERT / FinBERT)
training          fine-tuning with the corrected weighted focal loss
calibration       temperature scaling and calibration metrics (ECE, NLL, Brier)
evaluation        row-level and group-level metrics, confidence workflow, plots
inference         apply a trained model + calibrator to new descriptions
"""

__version__ = "1.0.0"
