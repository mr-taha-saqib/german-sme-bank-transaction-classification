# Delivery Manifest

German SME Bank Transaction Classification, Academic Proof of Concept

| | |
|---|---|
| Package size | **9.4MB**, 208 files |
| Frozen split | sha256 `f715f040d8a5daa621164ae8…` |
| Augmentation corpus v2 | sha256 `4eb242d9f384041fc5ec67ce…` |
| Validated | run standalone inside this folder, see §8 |

Start here: [`PROJECT_INDEX.md`](PROJECT_INDEX.md) → [`FINAL_REPORT.md`](FINAL_REPORT.md)

---

## 1. Documentation (11 files)

| File | Purpose |
|---|---|
| `PROJECT_INDEX.md` | **Entry point.** Documentation map, code map, results map, one-line summary |
| `FINAL_REPORT.md` | **The complete report.** 16 sections: objective, dataset, preprocessing, grouping, leakage prevention, augmentation v2, both model methodologies, five-seed experiment, temperature scaling, confidence threshold, final results, error analysis, paper comparison, limitations, reproducibility |
| `README.md` | Project overview, setup, hardware requirements, full reproduction commands, where results are written, limitations |
| `AUDIT.md` | Phase-1 audit of the client's prototype: 13 problems, 10-entry leakage risk register, dataset structure, recommended experimental design |
| `DATA_PIPELINE.md` | Step 2, dataset selection, column census, metadata analysis, duplicate removal, the 14 preprocessing transformations, label-free grouping and threshold measurement, leakage-safe splitting, five validation checks |
| `STEP3_GERMANFINBERT.md` | Step 3, checkpoint identification, training configuration, focal-loss evidence, first test evaluation |
| `AUGMENTATION_DECISION.md` | Step 3A, leakage finding (9 near-duplicate rows), the three gates, corpus v2, and the honest outcome that the correction changed no prediction |
| `STEP4_FINBERT.md` | Step 4, FinBERT verification, fairness preconditions, five-seed results, why the accuracy gap must not be over-claimed |
| `STEP5_CALIBRATION.md` | Step 5, temperature scaling, the negative result, and its diagnosed cause |
| `FINAL_E2E_ACCEPTANCE.md` | **Pre-delivery acceptance test.** End-to-end re-verification of every client requirement, the client-provided originals, the data pipeline, both models, calibration, the 0.8 workflow, an independent recomputation of all final numbers from raw logits, and a clean-environment run |
| `RESULTS_ARCHIVE_PLAN.md` | Inventory of the *working* project's `results/` tree and the cleanup carried out there. Informational. That tree is not part of this package |

---

## 2. Final results (6 files)

| File | Purpose |
|---|---|
| `results/final_results.csv` | **Authoritative final numbers.** 40 rows: 2 models × 5 seeds × 2 calibration states × 2 levels |
| `results/final_results.json` | The above plus integrity audit, traceability record and error analysis |
| `results/final_summary.csv` | Mean ± sd tables |
| `results/final_summary.xlsx` | Workbook: Summary / PerSeed / Integrity / Errors / Provenance sheets |
| `results/final_comparison.md` | The written final comparison with conclusions and paper comparison |
| `results/final_error_analysis.csv` | **Error-analysis artefact**, per-group correctness for both models across all five seeds, with modal predictions |

---

## 3. Reproducibility artefacts: `results/dataset/` (14 files)

These are what make the experiment reproducible; the scripts verify their checksums on load and abort if they change.

| File | Purpose |
|---|---|
| `split_assignment.csv` | **The frozen train/validation/test split**, 1 968 rows with `group_id`, `group_key` and `split` |
| `split_manifest.json` | Per-split counts, group→split map, **SHA-256 checksum** |
| `split_report.json` | Full leakage validation: group overlap, text overlap, near-duplicate audit, per-class distributions |
| `augmentation_corpus_v2.csv` | **The final augmentation corpus**, 4 541 gated synthetic rows |
| `augmentation_manifest_v2.json` | Gates applied, counts, class distribution, **SHA-256 checksum** |
| `augmentation_corpus_v1.csv`, `augmentation_manifest_v1.json` | Pre-correction corpus, retained so the gate-C fix is auditable |
| `augmentation_verification.json` | Output of the 11 adversarial leakage checks |
| `grouping_report.json` | Groups per category, size distribution, label purity, effective sample size |
| `near_duplicate_census.json` | Similarity structure at 7 thresholds, the evidence for choosing 0.60 |
| `preprocessing_audit.json` | Per-step cleaning audit, vocabulary reduction, removed token types |
| `preprocessing_examples.csv` | Before/after table for 40 descriptions |
| `class_distribution.csv` | Rows and groups per category per split |
| `column_census.csv` | All 65 source columns: missingness, cardinality, examples |
| `dataset_summary.json` | Every Step-2 figure in one file |
| `config.json` | Exact configuration used to build the split |

---

## 4. Per-run evidence: `results/step4_comparison/` (10 runs × 9 files + 6 aggregates)

For each of `germanfinbert_seed{42..46}` and `finbert_seed{42..46}`:

| File | Purpose |
|---|---|
| `logits.npz` | **Raw test + validation logits, probabilities, labels, group ids.** The ultimate source of every reported number, and what lets calibration re-run without a GPU |
| `test_predictions.csv` | Per-example prediction, confidence, correctness, decision at 0.8 |
| `test_metrics.json` | All metrics, group level and row level |
| `validation_metrics.json` | Monitoring metrics (no decision taken from them) |
| `model_meta.json` | Checkpoint, tokenizer diagnostics, full training config, per-epoch history, device, **split and corpus checksums**, environment |
| `label_mapping.json` | 37-class mapping. Identical across all runs |
| `config.json` | Exact `ExperimentConfig` used |
| `confusion_matrix_test.csv` | Full 37-class confusion matrix |
| `classification_report_test.txt` | Per-class precision / recall / F1 |

Aggregates: `per_seed_group.csv`, `per_seed_row.csv`, `summary_group.csv`, `summary_row.csv`, `comparison.json`.

Model weights are deliberately excluded (roughly 440 MB each). They are fully regenerable: the pipeline is deterministic, seeded and checksum-guarded, so `train_model.py` reproduces any model exactly in ~2 minutes. Metrics, logits and predictions (which cannot be regenerated without a GPU) are all included.

---

## 5. Calibration artefacts: `results/step5_calibration/` (43 files)

| File | Purpose |
|---|---|
| `calibration_results.json` | Every temperature, fit record, and before/after metric |
| `per_seed_calibration.csv` | 40 rows: per model, seed, level and calibration state |
| `summary_calibration.csv` | Mean ± sd tables |
| `<model>_seed<N>_calibration.json` | Fitted `T`, NLL before/after, validation and test metrics (10 files) |
| `<model>_seed<N>_calibrated_probs.npz` | Calibrated test probabilities + temperature (10 files) |
| `reliability_<model>_seed<N>_<level>.png` | **20 reliability diagrams**, before vs after calibration |

---

## 6. Implementation

### Experiment scripts (8 files, project root)

| Script | Step | Purpose |
|---|---|---|
| `prepare_dataset.py` | 2 | Clean, group, split, freeze with checksum |
| `verify_dataset.py` | 2 | 19 independent checks on the frozen split |
| `prepare_augmentation.py` | 3A | Apply gates A+B+C, freeze corpus v2 |
| `verify_augmentation.py` | 3A | 11 adversarial leakage checks |
| `train_model.py` | 3/4 | Fine-tune one model, model-agnostic, so both get identical treatment |
| `compare_models.py` | 4 | Aggregate seeds; refuses to compare runs differing in data or config |
| `calibrate_models.py` | 5 | Temperature scaling from saved logits (no GPU) |
| `final_evaluation.py` | 6 | Final tables, integrity audit, traceability, error analysis |

### `src/`: pipeline modules (13 files)

| Module | Purpose |
|---|---|
| `config.py` | Configuration dataclasses, model registry: every knob that affects a result |
| `data_preparation.py` | Workbook loading, source-flagged duplicate removal, synthetic corpus loading |
| `preprocessing.py` | German-aware cleaning, T0–T13, plus the legacy cleaner for provenance resolution |
| `grouping.py` | Label-free canonical keys, near-duplicate merging, similarity census |
| `splitting.py` | Group-based splitting, leakage validation, the three augmentation gates |
| `dataset_io.py` | Frozen-split loading with checksum verification |
| `models.py` | Model/tokenizer loading, tokenizer diagnostics |
| `training.py` | Fine-tuning, corrected weighted focal loss |
| `calibration.py` | Temperature and vector scaling, ECE / MCE / NLL / Brier |
| `evaluation.py` | Metrics, confidence workflow, group-level reduction, plots |
| `inference.py` | Apply a trained model to new descriptions |
| `utils.py` | Seeding, logging, environment capture |
| `__init__.py` | Package docstring and version |

### Tests and configuration

| File | Purpose |
|---|---|
| `tests/test_pipeline.py` | **66 unit checks**, preprocessing, label-free grouping, near-duplicate census, duplicate removal, split policy, split reproducibility, augmentation gates, calibration invariants, evaluation arithmetic |
| `requirements.txt` | Pinned dependencies with PyTorch install instructions. **No `openai` dependency**. The pipeline makes no API calls |

Run configuration is captured per run in `config.json` files (`results/dataset/`, each `results/step4_comparison/*/`), so no separate configuration file is needed.

---

## 7. Originals preserved

| Path | Contents |
|---|---|
| `data/` | **All five original datasets, unmodified**: `synthetic_b2b_transactions_2000.xlsx` (primary) plus the four LLM corpora |
| `notebooks/finbert_transaction_classification_sanitized.ipynb` | **The client's original 144-cell notebook, byte-identical.** Never modified or executed by this pipeline |
| `reference/Ergebnisuebersicht_FinBERT.xlsx` | The prototype's own results workbook |
| `reference/audit_evidence/` | Scripts and reports backing the Phase-1 audit's numbers. Retained as evidence of how those figures were produced; the two `inspect_*.py` scripts read the original handover tree, which is not part of this package, so they are records rather than runnable tools |
| `reference/example_new_transactions.csv` | Sample input for the inference demo |

---

## 8. Validation

The package was validated standalone, by running the full check suite from inside this folder:

| Check | Result |
|---|---|
| `verify_dataset.py` | **19/19 passed** |
| `verify_augmentation.py` | **11/11 passed, 0 findings** |
| `tests/test_pipeline.py` | **66/66 passed** |
| `final_evaluation.py`, integrity | **12/12 passed** |
| `final_evaluation.py`, traceability | **120/120 recomputations matched** |
| Regenerated finals vs master project | **every metric identical** |

That last row matters: `final_evaluation.py` recomputes all 40 result rows from the raw logits shipped here, and they match the working project bit-for-bit. The results in this package are self-verifying.

---

## 9. Excluded, and why

| Excluded | Reason |
|---|---|
| `RESULTS_SUPERSEDED_PHASE1.md` | Superseded phase-1 results document, removed at the client's request so this package contains only current/final documentation |
| `model.safetensors` (~440 MB each) | Regenerable in ~2 min each; deterministic and checksum-guarded |
| `.venv/` | Environment, rebuild from `requirements.txt` |
| `__pycache__/`, `*.pyc` | Caches |
| `*.log` | Run logs, regenerated on each run |
| Credentials, API keys, `.env` files | **None exist anywhere in this project.** The pipeline makes no API calls and needs no credentials |
| Superseded run directories (`main/`, `provenance/`, `ablation_*`, `seeds/`, `step3*`) | Phase-1 and intermediate outputs from earlier pipeline versions, replaced by the Step 2-6 results included here |
| `run_experiment.py`, `verify_results.py`, `summarize_runs.py` | Phase-1 entry points that do not produce the final results |

---

## 10. Getting started

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0+cu126
pip install -r requirements.txt

# Verify the shipped results without training anything (seconds, no GPU)
python verify_dataset.py          # 19 checks
python verify_augmentation.py     # 11 checks
python tests/test_pipeline.py     # 66 checks
python final_evaluation.py        # 12 integrity + 120 traceability checks

# Reproduce the full experiment from scratch (~25 min on an RTX 3060)
python prepare_dataset.py
python prepare_augmentation.py --version v2
for SEED in 42 43 44 45 46; do
  for MODEL in germanfinbert finbert; do
    python train_model.py --model $MODEL --augment --augmentation-version v2 \
        --seed $SEED --output-dir results/step4_comparison \
        --run-name ${MODEL}_seed${SEED}
  done
done
python compare_models.py
python calibrate_models.py
python final_evaluation.py
```

Python 3.10+ (verified on 3.13.2). GPU optional. CUDA is used automatically when
available, with CPU fallback. First training run downloads ~880 MB of pre-trained
weights from Hugging Face; everything else is offline.

---

## 11. Headline result

Under the experimental conditions and test set used in this proof of concept, on
33 independent test groups across five seeds:

| Model | Accuracy | Macro F1 | Top-2 | ≥0.8 Coverage | ≥0.8 Accuracy |
|---|---|---|---|---|---|
| **GermanFinBERT** (uncalibrated) | **0.6485 ± 0.0271** | **0.5442 ± 0.0408** | **0.7576 ± 0.0214** | **60.61 ± 5.67 %** | 0.8022 ± 0.0335 |
| standard FinBERT (uncalibrated) | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.6788 ± 0.0271 | **0.00 %** | n/a |

GermanFinBERT led on every metric and every seed. The decision-relevant difference is
coverage: FinBERT never reaches 0.8 confidence, so it automates nothing.

Temperature scaling did not improve calibration on unseen test data and is reported
as an experiment, not an improvement. **This project does not reproduce the reference
paper**, different language, 37 classes rather than 11, synthetic data, and 33 test
groups rather than thousands of rows. The 33-group test set is too small to establish
that the model difference generalises.
