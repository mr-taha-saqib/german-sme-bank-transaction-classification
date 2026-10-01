# Project Index: German SME Transaction Classification

Documentation map for the academic proof of concept. Read in this order.

| # | Document | What it contains |
|---|---|---|
| 0 | [`README.md`](README.md) | Objective, scope, setup, how to reproduce everything, final results at a glance, limitations |
| 1 | [`AUDIT.md`](AUDIT.md) | **Phase 1, audit of the client's prototype.** 144-cell notebook and all five datasets inspected; 13 problems and a 10-entry leakage risk register; the decisive structural finding that 2 000 transactions collapse to ~184 distinct texts; recommended experimental design |
| 2 | [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | **Step 2, data preparation.** Dataset selection with reasons; full column census and metadata analysis; source-flagged duplicate removal; the 14 preprocessing transformations; label-free grouping and how the 0.60 threshold was measured; the leakage-safe train/validation/test split; the five validation checks |
| 3 | [`STEP3_GERMANFINBERT.md`](STEP3_GERMANFINBERT.md) | **Step 3, primary model.** Checkpoint identification and the `_FP` vs `_SC` ambiguity; full training configuration; evidence for reproducing the paper's weighted focal loss; first test evaluation; the finding that without augmentation the model never reaches 0.8 confidence |
| 4 | [`AUGMENTATION_DECISION.md`](AUGMENTATION_DECISION.md) | **Step 3A, leakage finding and correction.** Adversarial verification of the synthetic corpus; discovery that 9 rows were near-duplicates of two held-out test patterns; the three gates; corpus **v2**; the honest outcome that removing them changed no prediction |
| 5 | [`STEP4_FINBERT.md`](STEP4_FINBERT.md) | **Step 4, comparison model.** `ProsusAI/finbert` verification; mechanically-checked fairness preconditions; five-seed experiment (42–46); per-seed and mean ± sd results; why the accuracy gap must not be over-claimed |
| 6 | [`STEP5_CALIBRATION.md`](STEP5_CALIBRATION.md) | **Step 5, temperature scaling.** Fitted on validation logits only, applied unchanged to test; the negative result and its diagnosed cause; why accuracy cannot change under a scalar temperature |
| 7 | [`results/final_comparison.md`](results/final_comparison.md) | **Step 6, final comparison.** Integrity audit, final tables, per-seed results, the 0.8 workflow, calibration analysis, error analysis, conclusions, paper comparison |
| 8 | [`FINAL_REPORT.md`](FINAL_REPORT.md) | **The complete written report.** Everything above in one continuous narrative, from objective to limitations |

## Supporting documents

| Document | Purpose |
|---|---|
| [`RESULTS_ARCHIVE_PLAN.md`](RESULTS_ARCHIVE_PLAN.md) | Inventory and cleanup options for the *working* project's `results/` tree. Informational |

## Code map

| Script | Step | Purpose |
|---|---|---|
| `prepare_dataset.py` | 2 | Clean, group, split, freeze with checksum |
| `verify_dataset.py` | 2 | 19 independent checks on the frozen split |
| `prepare_augmentation.py` | 3A | Apply gates A+B+C, freeze corpus v2 with checksum |
| `verify_augmentation.py` | 3A | 11 adversarial leakage checks on the corpus |
| `train_model.py` | 3/4 | Fine-tune one model, model-agnostic, so both get identical treatment |
| `compare_models.py` | 4 | Aggregate seeds; refuses to compare runs that differ in data or config |
| `calibrate_models.py` | 5 | Temperature scaling from saved logits (no GPU, no retraining) |
| `final_evaluation.py` | 6 | Final tables, integrity audit, traceability, error analysis |
| `tests/test_pipeline.py` | — | 66 unit checks on preprocessing, grouping, splitting, gating, calibration, evaluation |

`src/` holds the modules: `config`, `data_preparation`, `preprocessing`, `grouping`,
`splitting`, `dataset_io`, `models`, `training`, `calibration`, `evaluation`,
`inference`, `utils`.

Earlier phase-1 entry points (`run_experiment.py`, `verify_results.py`,
`summarize_runs.py`) did not produce the final results and are **not included in
this package**.

## Results map

| Path | Contents |
|---|---|
| `results/dataset/` | Frozen split + corpus v2, manifests, checksums, class distribution, near-duplicate census |
| `results/step4_comparison/` | 10 runs (2 models × 5 seeds): logits, predictions, metrics, confusion matrices |
| `results/step5_calibration/` | Fitted temperatures, calibrated probabilities, 20 reliability diagrams |
| `results/final_results.csv` / `.json` | **The authoritative final numbers**, 40 rows plus integrity and error analysis |
| `results/final_summary.csv` / `.xlsx` | Mean ± sd summary tables |
| `results/final_error_analysis.csv` | Per-group correctness for both models across all seeds |

## The one-line summary

Under the experimental conditions and test set used in this proof of concept,
GermanFinBERT reached 0.6485 ± 0.0271 group-level accuracy against standard
FinBERT's 0.6121 ± 0.0254, leading on every metric and every seed. But the
decision-relevant difference is that GermanFinBERT automates 60.6 % of transactions
at the 0.8 threshold while FinBERT automates 0 %. Temperature scaling did not
improve calibration on unseen data and is reported as an experiment, not an improvement.
The 33-group test set is too small to establish that any of this generalises.
