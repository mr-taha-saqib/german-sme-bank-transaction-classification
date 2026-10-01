# Results Archive Plan: Inventory and Executed Cleanup

This document is the inventory of the working project's `results/` tree, the options
that were considered, and a record of the option that was carried out. Sections 1 to 7
describe the tree as it stood before the cleanup; section 8 records what was done.

At the time of the inventory `results/` occupied roughly 2.5 GB, of which almost all was
six copies of model weights. **Option A was executed**, and the tree now stands at
**907.4 MB**.

---

## 1. Current and required: never delete

These are the final experiment. Deleting any of them breaks reproducibility or
verification.

| Path | Size | Why it must stay |
|---|---|---|
| `results/dataset/` | 1.5 MB | **Frozen split** (`split_assignment.csv` + manifest, sha256 `f715f040d8a5daa6…`) and **augmentation corpus v2** (`augmentation_corpus_v2.csv` + manifest, sha256 `4eb242d9f384041f…`). Every training run verifies these checksums on load. Also holds the grouping report, near-duplicate census, class distribution and preprocessing audit |
| `results/step4_comparison/*/logits.npz` | ~9 MB total | **Raw test and validation logits for all 10 runs.** The ultimate source of every reported number, and what makes Step 5 re-runnable without retraining |
| `results/step4_comparison/*/test_predictions.csv` | ~3 MB | Per-example predictions, confidence and decisions |
| `results/step4_comparison/*/test_metrics.json`, `validation_metrics.json`, `model_meta.json`, `label_mapping.json`, `config.json` | ~4 MB | Metrics, full training configuration, checksums, environment |
| `results/step4_comparison/*/confusion_matrix_test.csv`, `classification_report_test.txt`, `*.png` | ~5 MB | Per-class evidence and plots |
| `results/step5_calibration/` | 3.5 MB | Fitted temperatures, calibrated probabilities, 20 reliability diagrams |
| `results/final_results.csv` / `.json`, `final_summary.csv` / `.xlsx`, `final_comparison.md`, `final_error_analysis.csv` | 0.1 MB | **The authoritative final results** |

Total that must be preserved: ~27 MB excluding weights.

---

## 2. Superseded: phase-1 pipeline

Produced *before* the Step-2 rework, using `token_set` grouping with a label-aware
near-duplicate merge at Jaccard 0.80 and no duplicate removal (184 groups instead of
152). Verified by reading each directory's own `config.json`.

| Path | Size | Note |
|---|---|---|
| `results/main/` | **841.0 MB** | phase-1 headline run; 839 MB is two weight files |
| `results/seeds/` | 5.4 MB | phase-1 3-seed study |
| `results/provenance/` | 2.4 MB | phase-1 provenance-aligned split |
| `results/ablation_no_aug/` | 2.1 MB | phase-1 2×2 cell |
| `results/ablation_prov_no_aug/` | 2.0 MB | phase-1 2×2 cell |
| `results/step2_inspection/` | 0.0 MB | scratch output from the Step-2 investigation |
| **Subtotal** | **853.0 MB** | |

Their findings are described in the working project's phase-1 result document, which is retained there.

---

## 3. Intermediate Step-3 runs: superseded but small

| Path | Size | Note |
|---|---|---|
| `results/step3_germanfinbert/` | **420.5 MB** | Step-3 primary (no augmentation). Metrics still cited in `STEP3_GERMANFINBERT.md`; 419.5 MB is weights |
| `results/step3a_germanfinbert_augmented_v2/` | **420.5 MB** | Step-3A corrected run. Superseded by `step4_comparison/germanfinbert_seed42`, which reproduces it exactly; 419.5 MB is weights |
| `results/step3_variant_crossentropy/` | 0.3 MB | loss control referenced in `STEP3_GERMANFINBERT.md` §3 |
| `results/step3_variant_augmented/` | 0.3 MB | pre-correction augmented variant, referenced in `AUGMENTATION_DECISION.md` §6 |

The two small variant directories are cited in current documentation and are cheap.
Recommend keeping. Their *metrics* should be kept even if weights go.

---

## 4. Model weights: the real cost

| Path | Size | Assessment |
|---|---|---|
| `results/step4_comparison/germanfinbert_seed42/model/` | 419.5 MB | **Keep**. The deliverable model, used by `src/inference.py` |
| `results/step4_comparison/finbert_seed42/model/` | 417.8 MB | Optional, comparison model; keep if inference on FinBERT is wanted |
| `results/step3a_germanfinbert_augmented_v2/model/` | 419.5 MB | **Redundant**. Byte-equivalent training to `step4_comparison/germanfinbert_seed42` (verified: identical metrics to 5 dp) |
| `results/step3_germanfinbert/model/` | 419.5 MB | Superseded, the no-augmentation model, retrainable in 26 s |
| `results/main/models/germanfinbert/` | 419.5 MB | Phase-1, superseded |
| `results/main/models/finbert/` | 417.8 MB | Phase-1, superseded |

Weights are regenerable. Every run is deterministic and checksum-guarded, so any
model can be reproduced exactly from the frozen split and corpus. Metrics, logits and
predictions are what cannot be regenerated without a GPU.

---

## 5. Never delete, under any option

* `results/dataset/`, the frozen split and corpus v2 with their checksums
* all `logits.npz`, `test_predictions.csv` and `*_metrics.json` under
  `results/step4_comparison/`
* everything under `results/step5_calibration/`
* `results/final_*` files
* all top-level documentation
* `data/`, `notebooks/` (the client's original notebook), `reference/`

---

## 6. Options and recovered space

| Option | Action | Recovered | Risk |
|---|---|---|---|
| **A, conservative** *(recommended)* | Delete the 4 superseded weight files only (`main/models/*`, `step3_germanfinbert/model`, `step3a_.../model`). Keep every directory, metric and log | **1 676 MB** | None. All metrics, logits and documentation survive; models regenerable in ~2 min each |
| **B, moderate** | Option A, plus delete the 5 phase-1 directories entirely and `step2_inspection` | **1 690 MB** | Low. The phase-1 findings remain documented in the working project, but the underlying per-example evidence for those claims would be gone |
| **C, minimal** | Delete `results/main/models/*` only | 837 MB | None |
| **D, aggressive** | Option B, plus `step4_comparison/finbert_seed42/model` | **2 108 MB** | Low-moderate. FinBERT inference would then require a 2-minute retrain |

Under Option A, `results/` drops from 2 541 MB to ~865 MB. Under Option B, to
~851 MB. Under Option D, to ~433 MB.

A fifth option, archiving rather than deleting, e.g. zipping the phase-1 directories to
`archive/phase1.zip`, recovers roughly 12 MB of the small directories but not the
weights, since those compress poorly. It is only worth doing if the phase-1 per-example
evidence must be retained in auditable form.

---

## 7. Recommendation

Option A. It recovers 1 676 MB (66 % of the tree) with zero loss of evidence:
every metric, every set of logits, every prediction file, every log and every document
survives. The only things removed are four regenerable weight files, and the pipeline's
determinism plus checksum verification means any of them can be reproduced exactly.

If disk pressure is not a concern, do nothing is also defensible. 2.5 GB is modest,
and keeping the phase-1 weights makes the earlier results independently re-verifiable.

---

## 8. Status: Option A executed

**Option A was carried out.** Four superseded weight files were deleted, recovering
**1 757.6 MB**. `results/` went from 2 665.0 MB to **907.4 MB**.

| Deleted | Size |
|---|---|
| `results/main/models/germanfinbert/model.safetensors` | 439.8 MB |
| `results/main/models/finbert/model.safetensors` | 438.1 MB |
| `results/step3_germanfinbert/model/model.safetensors` | 439.8 MB |
| `results/step3a_germanfinbert_augmented_v2/model/model.safetensors` | 439.8 MB |

Only the `model.safetensors` files were removed, not the directories containing them.
That distinction matters: `results/main/models/*` also held phase-1 `metrics.json`,
`test_predictions.csv`, `calibrator.json`, classification reports and plots, and all 12
such files per model were kept. Deleting the directories would have destroyed per-example
evidence that Option A promised to preserve.

**Retained**, because they are the deliverable and comparison models used by
`src/inference.py`:

| Kept | Size |
|---|---|
| `results/step4_comparison/germanfinbert_seed42/model/model.safetensors` | 439.8 MB |
| `results/step4_comparison/finbert_seed42/model/model.safetensors` | 438.1 MB |

### Verification after deletion

Re-run as this plan required, with nothing missing:

| Check | Result |
|---|---|
| `verify_dataset.py` | 19/19 passed |
| `verify_augmentation.py` | 11/11 passed, 0 findings |
| `tests/test_pipeline.py` | 66/66 passed |
| `final_evaluation.py --verify-only` | 12/12 integrity, 120/120 recomputations matched |
| Inference on the retained model | runs, classifies, applies the 0.8 workflow |
| Phase-1 predictions and metrics | all present |

Every deleted file is regenerable: the pipeline is deterministic, seeded and
checksum-guarded, so `train_model.py` reproduces any of these models exactly in about two
minutes. No metric, logit, prediction, log or document was lost.
