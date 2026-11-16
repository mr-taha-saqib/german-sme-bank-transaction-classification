# Final End-to-End Acceptance Test

Independent pre-delivery verification of the complete project and the client-facing
delivery package. **No models were retrained, no methodology changed, no result
altered, no file deleted.**

| | |
|---|---|
| Frozen split | `f715f040d8a5daa621164ae8…`, **unchanged** |
| Augmentation corpus v2 | `4eb242d9f384041fc5ec67ce…`, **unchanged** |
| Confidence threshold | 0.80, **unchanged** |
| Seeds | 42, 43, 44, 45, 46. All reported, none selected on test |
| Defects found | **2**, both fixed and re-verified (see §8) |

---

## 1. Client requirements

| # | Requirement | Evidence | Status |
|---|---|---|---|
| 1 | German transaction classification | 37 German categories, 1 968 rows after de-duplication; German-aware preprocessing preserving umlauts/ß and expanding 18 banking abbreviations (`results/dataset/preprocessing_audit.json`) | **PASS** |
| 2 | GermanFinBERT as primary model | `scherrmann/GermanFinBert_FP` verified in all 5 run metas; vocab 31 102, 110.0 M params, 37-class head | **PASS** |
| 3 | Standard FinBERT as comparison | `ProsusAI/finbert` verified in all 5 run metas; vocab 30 522, 109.5 M params, 37-class head | **PASS** |
| 4 | Same split and evaluation for both | Training-config differences between models: **NONE** across 12 hyperparameters; identical split sha, corpus sha, label mapping, test group ids and ordering | **PASS** |
| 5 | Preprocessing | 14 numbered transformations, T0–T13; 1 427 distinct raw → 184 distinct cleaned; 0 empty | **PASS** |
| 6 | Grouping of similar descriptions | Label-free canonical key + union-find merge at Jaccard ≥ 0.60; 184 keys → **152 groups**, 0 label-impure; threshold measured (`near_duplicate_census.json`) | **PASS** |
| 7 | Leakage-safe group-based train/validation/test split | train 1 093/103 · validation 228/16 · test 647/33; 0 group overlap, 0 text overlap, 0 same-label near-duplicates | **PASS** |
| 8 | Synthetic augmentation using corrected v2 | Gates A+B+C → 4 541 rows; 0 rows violate Jaccard ≥ 0.60; max similarity to held-out 0.50 | **PASS** |
| 9 | Confidence scores | Per-example confidence in all 10 `test_predictions.csv`; full statistics (mean/median/std/min/max) in every metrics block | **PASS** |
| 10 | Temperature scaling / calibration | 10 independent fits, each on 228 validation rows, applied unchanged to test; `calibration_results.json` | **PASS** |
| 11 | 0.8 confidence threshold | `{0.8}` in all 10 run configs; configurable via `--threshold`; never lowered | **PASS** |
| 12 | Automatic vs manual-review logic | 40 workflow blocks re-derived by hand, **0 discrepancies**; counts, coverage and conditional accuracy all correct | **PASS** |
| 13 | Classification + confidence/calibration metrics | Accuracy, macro/weighted F1, top-2, per-class P/R/F1, confusion matrix, ECE, MCE, NLL, Brier, coverage, high-confidence accuracy | **PASS** |
| 14 | Reproducible local implementation | Split and corpus checksums **regenerate bit-identically** from raw data; verification runs with no GPU and no DL stack | **PASS** |
| 15 | Documentation of implementation and methodology | 12 documents; 22 internal links checked, 0 broken | **PASS** |
| 16 | Final error analysis | `results/final_error_analysis.csv` (per-group correctness × 5 seeds × 2 models) + `final_comparison.md` §6 + `FINAL_REPORT.md` §13 | **PASS** |
| 17 | Comparison with the academic paper | `FINAL_REPORT.md` §14, 13-row difference table, explicit non-reproduction statement | **PASS** |

No requirement was silently omitted.

Scope: the implementation contains no ERP integration, receipt matching, web
application, database, API, authentication or any production infrastructure. It has no
`openai` dependency and makes zero network calls at verification time (`grep` for
`requests`/`urllib`/`http` in the verification scripts returns 0). It remains within
the agreed academic proof-of-concept scope.

---

## 2. Client-provided files verified

All originals byte-identical (SHA-256).

| File | Check | Status |
|---|---|---|
| `ErstesPaper.pdf` (academic paper) | present, 1 417 KB, sha `8b18e19d1403f2eb…` | **PASS** |
| prototype handover archive (`.zip`) | present, 652 KB, sha `39a555a627b0542f…`; 10 entries, **0 mismatches** vs the extracted copy | **PASS** |
| `finbert_transaction_classification_sanitized.ipynb` | handover original == working copy == delivery copy | **PASS** |
| `synthetic_b2b_transactions_2000.xlsx` | byte-identical | **PASS** |
| `final_synthetic_clean_7033.csv` | byte-identical | **PASS** |
| `final_new_synthetic.csv` | byte-identical | **PASS** |
| `synthetic_fold1_paper_style.csv` | byte-identical | **PASS** |
| `synthetic_transactions_openai.csv` | byte-identical | **PASS** |
| `Ergebnisuebersicht_FinBERT.xlsx` | byte-identical | **PASS** |
| handover README (developer brief) | present in handover | **PASS** |
| original `requirements.txt` (handover) | present in handover | **PASS** |
| `.env.example` (handover template) | present in handover; **excluded from delivery**. No credentials shipped | **PASS** |

No original file is missing or altered.

---

## 3. Data pipeline end-to-end

The pipeline was re-run from the original workbook into a temporary directory, never
overwriting the authoritative artefacts.

| Check | Result | Status |
|---|---|---|
| Split checksum reproduces | `f715f040d8a5daa621164ae8ddfbeae6870ca5fef1a36b06fb5261143233347a`, **exact match** | **PASS** |
| Corpus v2 checksum reproduces | `4eb242d9f384041fc5ec67ce19aa5ab67eda09506f68e1c5d2330bbb1e58d02d`, **exact match** | **PASS** |
| `split_assignment.csv` identical | True | **PASS** |
| `augmentation_corpus_v2.csv` identical | True | **PASS** |
| Group ids deterministic | 152 groups, 0 label-impure | **PASS** |
| Groups disjoint across splits | `{train\|validation: 0, train\|test: 0, validation\|test: 0}` | **PASS** |
| Exact duplicate leakage | 0 shared cleaned texts in all three pairs | **PASS** |
| Near-duplicate leakage (Jaccard ≥ 0.60) | 0 same-label pairs across splits | **PASS** |
| Augmentation gates | A 2 483 dropped · B 0 · C 9 · reconciles 7 033 − 4 541 = 2 492 | **PASS** |
| Synthetic rows | **4 541** (expected 4 541) | **PASS** |
| Real training rows | **1 093** (expected 1 093) | **PASS** |
| Total training rows | **5 634** (expected 5 634) | **PASS** |

---

## 4. Model verification

| Check | GermanFinBERT | FinBERT | Status |
|---|---|---|---|
| Checkpoint | `scherrmann/GermanFinBert_FP` | `ProsusAI/finbert` | **PASS** |
| Tokenizer | BertTokenizer, 31 102, cased | BertTokenizer, 30 522, uncased | **PASS** |
| Classification head | 37 classes | 37 classes | **PASS** |
| Label mapping | 37, identical across seeds | 37, identical across seeds | **PASS** |
| Seeds in configs | [42, 43, 44, 45, 46] | [42, 43, 44, 45, 46] | **PASS** |
| Frozen split sha | `f715f040d8a5daa6` | `f715f040d8a5daa6` | **PASS** |
| Corpus sha / version | `4eb242d9f384041f` / v2 | `4eb242d9f384041f` / v2 | **PASS** |
| Train rows | 5 634 (1 093 + 4 541) | 5 634 (1 093 + 4 541) | **PASS** |
| Config | lr 2e-5, bs 16, len 128, 3 ep, focal γ2, fixed_epochs | identical | **PASS** |
| CUDA/GPU | `cuda:0` RTX 3060 12 GB, torch 2.13.0+cu126 | same | **PASS** |
| Saved logits/predictions/metrics | 6 artefact types × 5 seeds | 6 artefact types × 5 seeds | **PASS** |

Comparability: training-config differences NONE; identical test group ids, labels
and class names. **Seed-42 GermanFinBERT reproduces the established Step-3A result
exactly**, `[0.66667, 0.56381, 0.59798, 0.75758]` on both.

---

## 5. Calibration verification

| Check | Result | Status |
|---|---|---|
| Fitted on validation only | all 10 fits used **228 rows** (validation), never 647 (test) | **PASS** |
| Test labels never used to fit or select | fit sizes never equal the test size; no temperature chosen by test comparison | **PASS** |
| Each seed has its own temperature | 5 distinct values per model | **PASS** |
| Applied unchanged to test logits | recomputing `softmax(z/T)` reproduces the stored calibrated probabilities in all 10 runs | **PASS** |
| Test metrics computed after calibration | present in `calibration_results.json` and `final_results.csv` | **PASS** |
| Scalar temperature does not change argmax | verified for all 10 runs | **PASS** |
| Saved outputs match final reported results | included in the 400-comparison recomputation below | **PASS** |
| Temperatures refitted or tuned | **no**. Read from saved artefacts only | **PASS** |

Validation NLL improved in 10/10 runs (the fitted objective); test ECE worsened in
7/10 and test NLL in 6/10. Reported as a generalisation limitation, not an improvement.

---

## 6. 0.8 confidence workflow

| Check | Result | Status |
|---|---|---|
| Threshold exactly 0.80 | `{0.8}` across all 10 run configs | **PASS** |
| Configurable | `--threshold` flag present | **PASS** |
| Not lowered to improve coverage | operating point 0.80 throughout; sweep 0.50–0.95 is reporting only | **PASS** |
| Automatic coverage correct | 40 workflow blocks re-derived, 0 discrepancies | **PASS** |
| Manual-review coverage correct | auto + manual = 100 % in every block | **PASS** |
| Accuracy among automatic correct | re-derived, matches to the stored rounding | **PASS** |
| Both calibrated and uncalibrated | all 40 blocks cover both states × both levels | **PASS** |

Spot-check (GermanFinBERT seed 42, group level): 21 auto / 12 manual / 33 total,
4 errors auto-accepted: integer counts match the stored values exactly.

---

## 7. Final results reproduction

Every final row recomputed from the raw saved logits, not from any markdown summary.

| Check | Result | Status |
|---|---|---|
| `final_results.csv` rows | 40 (2 models × 5 seeds × 2 calibration states × 2 levels) | **PASS** |
| Independent metric comparisons | **400** (accuracy, macro/weighted F1, top-2, ECE, NLL, Brier, coverage, manual %, high-conf accuracy) | **PASS** |
| Mismatches | **0** | **PASS** |
| `final_evaluation.py` traceability | **120/120 recomputations matched** | **PASS** |
| Integrity checks | **12/12** | **PASS** |
| Confusion matrix recomputed == saved | True | **PASS** |
| Per-class metrics recomputed == saved | True (35 classes present) | **PASS** |

---

## 8. Documentation consistency

| Check | Result | Status |
|---|---|---|
| Obsolete split sizes (1 117 / 236) presented as current | none | **PASS** |
| Obsolete corpus size (4 687) presented as current | none | **PASS** |
| Obsolete headline (0.6061 vs 0.5758) presented as final | none | **PASS** |
| Current values present in key documents | 0.6485 / 0.6121 / 152 / 4 541 / 5 634 / 33 in README, FINAL_REPORT, final_comparison, PROJECT_INDEX, MANIFEST | **PASS** |
| Model names correct | `GermanFinBert_FP` in 5 docs, `ProsusAI/finbert` in 6; 0 wrong variants | **PASS** |
| Commands correspond to real files | all 9 referenced scripts exist | **PASS** |
| Broken links | main project 25 checked / **0 broken**; delivery 22 checked / **0 broken** | **PASS** |
| Historical values clearly labelled | Jaccard 0.80 and 5 643 appear only inside explicitly historical or v1-vs-v2 contexts | **PASS** |

### Defects found and fixed

#### Defect 1: docs referenced scripts absent from the package

 `README.md` stated the legacy entry points
(`run_experiment.py`, `verify_results.py`, `summarize_runs.py`) *"still run"*, and
`PROJECT_INDEX.md` described them. But they are absent from the delivery package,
where `MANIFEST.md` correctly lists them as excluded. The package therefore contradicted
itself and pointed a reader at files that were not there.

*Fix*: both sentences reworded in the delivery copies only to state that these
phase-1 entry points did not produce the final results and are not included in the
package. No result, number, methodology or main-project file was touched. Re-verified,
0 broken links, all suites still pass.

#### Defect 2: the documented inference command could not run

`README.md` documented an inference command that fails immediately, and `src/inference.py`
was still written against the prototype's run layout:

* it looked for weights at `<run>/models/<name>/`, but `train_model.py` writes them to
  `<run>/model/`, so no run in this project could ever resolve;
* on failure it told the user to run `run_experiment.py --save-models`, a phase-1 script
  not included in the package;
* `TransactionClassifier` read `class_names`, `max_length` and `preprocessing` from
  `model_meta.json`, but the current pipeline records them in `label_mapping.json` and
  `config.json`, so it raised `KeyError: 'class_names'` even once the path was corrected;
* the module docstring promised calibrated confidences, but **no code writes
  `calibrator.json`**, so calibration silently never applied.

*Fix*, applied identically to the main project and the delivery package: a
`resolve_model_dir()` that accepts both layouts, artefact lookup that searches the run
directory (with a fallback to the old single-file schema), an error message naming
`train_model.py`, a new `--calibration` flag, and a docstring and README that state
confidences are uncalibrated by default. Consistent with the uncalibrated results
being the primary operational result. No result, split, threshold or model was touched.

*Verified by execution*, not by inspection. Four paths were run against the retained
seed-42 weights: current layout, current layout with an explicit temperature
(`T = 0.9863`, calibrated confidence 0.9166 vs uncalibrated 0.9109), `--csv` batch mode on
the shipped `reference/example_new_transactions.csv` (10 rows, 5 automatic), and the
legacy phase-1 layout. The no-weights error message was checked too. All four behave
correctly, and the full suite still passes (19/19, 11/11, 66/66, 12/12 + 120/120).

*Why the first pass missed it*: the acceptance test verified that every file referenced by
the documentation exists, but did not execute the documented commands. Model
weights are excluded from the package, so the inference path was never exercised. The
check has been performed now.

### One false alarm, recorded for transparency

An early workflow check reported `False` for GermanFinBERT seed 42. Investigation showed
the acceptance script compared unrounded values against stored values that the source
code rounds to 3 dp (percentages) and 5 dp (accuracies). Integer counts matched exactly
(21/12/33/4) and `round(exact, n) == stored` in every case. Not a project defect.
The acceptance script's tolerance was wrong, and the full re-check across all 40 workflow
blocks returned 0 discrepancies.

---

## 9. Reproducibility from the clean package

A fresh virtual environment was created and the suite was run **from inside the delivery
package**, in two stages.

Stage 1, no deep-learning stack at all. Only `numpy`, `pandas`, `scikit-learn` and
`openpyxl`; no torch, no transformers, no GPU.

| Check | Result | Status |
|---|---|---|
| `verify_dataset.py` | **19/19 passed** | **PASS** |
| `verify_augmentation.py` | **11/11 passed, 0 findings** | **PASS** |
| `final_evaluation.py` integrity | **12/12 passed** | **PASS** |
| `final_evaluation.py` traceability | **120/120 matched** | **PASS** |
| Regenerated finals == shipped finals | every metric identical | **PASS** |
| Network calls in verification scripts | **0** | **PASS** |
| API keys or external services required | **none** | **PASS** |
| README commands correspond to real files | all present | **PASS** |

Stage 2. CPU-only torch added. `torch 2.13.0+cpu`, `torch.cuda.is_available()`
returns `False`, `torch.version.cuda` is `None`. Still no GPU and no `transformers`.

| Check | Result | Status |
|---|---|---|
| `tests/test_pipeline.py` | **66/66 passed, 0 failed** | **PASS** |

**A GPU is not required to verify the supplied final results, and neither is a
deep-learning stack.** The three verification entry points run on
`numpy`/`pandas`/`scikit-learn` alone, because they read the saved logits rather than
executing a model. The unit-test suite additionally needs `torch` (its calibration tests exercise the LBFGS fit), but CPU torch is sufficient, as demonstrated above.

Training from scratch is the only step that needs more. It requires `torch` and
`transformers`, plus network access on first run to download the two Hugging Face
checkpoints, and for practical runtime the RTX 3060: roughly 2 minutes per run and
about 25 minutes for the full 10-run experiment. On CPU the same training would take
substantially longer. This is documented for the client's benefit; it is not needed to
check any number in this package.

---

## 10. Security and package cleanliness

| Check | Expected | Actual | Status |
|---|---|---|---|
| Credentials | 0 | 0 | **PASS** |
| API keys | 0 | 0 real keys | **PASS** |
| `.env` files | 0 | 0 | **PASS** |
| `__pycache__` | 0 | 0 | **PASS** |
| `*.pyc` | 0 | 0 | **PASS** |
| `.venv` | 0 | 0 | **PASS** |
| Model weights (intentionally excluded) | 0 | 0 `*.safetensors` | **PASS** |
| Log files | 0 | 0 | **PASS** |
| Absolute/private user paths | 0 | 0 | **PASS** |
| Broken links | 0 | 0 | **PASS** |
| External service dependency | none | none. `openai` appears only in a comment stating it is *not* a dependency | **PASS** |

The only text matching a credential pattern is inside the **client's own original
notebook**, which we were instructed to preserve byte-identical: two lines reading
`api_key=os.environ["AZURE_OPENAI_API_KEY"]`. That is an environment-variable *lookup*
in the client's file. No key value is present anywhere in the package.

---

## 11. Final file inventory

| Category | Count |
|---|---|
| **Total files** | **208** |
| **Package size** | **9.4 MB** |
| Documentation (`.md`) | 13 (12 root + 1 in `results/`) |
| Source modules (`src/`) | 13 |
| Experiment scripts (root `.py`) | 8 |
| Tests | 1 (66 checks) |
| Datasets (`data/`) | 5 |
| Final results (`results/final_*`) | 6 |
| Reproducibility artefacts (`results/dataset/`) | 16 |
| Calibration artefacts (`results/step5_calibration/`) | 43 |
| Per-run evidence (`results/step4_comparison/`) | 10 directories, 90 files + 6 aggregates |
| Original client files | notebook ×1, reference ×8 entries, datasets ×5 |

`MANIFEST.md` matches the actual contents: the documentation count went to 10 root
entries when the superseded document was removed (recorded in the manifest's exclusion
table), then to 11 when this acceptance report was added: 12 root `.md` files in
total, since the manifest does not list itself. Verified after the update: **208 files,
13 `src/` modules, 0 `__pycache__`, 0 `*.pyc`**.

*Two changes the acceptance test itself made to the package, both recorded for
transparency:* this report was copied in and listed in the manifest; and running
`tests/test_pipeline.py` from inside the package created `src/__pycache__` (9 `.pyc`
files), which was deleted afterwards to restore the package to its verified state.
No result, dataset, artefact or source file was modified.

---

## 12. Final project integrity: automated checks

| Suite | Baseline | Main project (GPU) | Clean CPU package | Status |
|---|---|---|---|---|
| `verify_dataset.py` | 19/19 | **19/19** | **19/19** | **PASS** |
| `verify_augmentation.py` | 11/11 | **11/11** | **11/11** | **PASS** |
| `tests/test_pipeline.py` | 66/66 | **66/66** | **66/66** (CPU torch) | **PASS** |
| `final_evaluation.py` integrity | 12/12 | **12/12** | **12/12** | **PASS** |
| `final_evaluation.py` traceability | 120/120 | **120/120** | **120/120** | **PASS** |
| Broken-link check | 0 | **0** (25 links) | **0** (22 links) | **PASS** |
| Independent recomputation | — | **400 comparisons, 0 mismatches** | — | **PASS** |
| Package validation | — | — | **all suites pass in place** | **PASS** |

---

## 13. Known limitations

Carried forward unchanged from the project documentation; all are genuine and already
recorded.

1. **Small final test set. 33 independent groups.** One group is worth 3.03 accuracy
   points; the model gap is 1.2 groups. A 95 % CI on 0.65 spans roughly ±0.16, more
   than four times the observed difference.
2. **Small calibration set, 16 groups covering 16 of 37 classes.** Direct cause of
   temperature scaling failing to generalise.
3. **Five seeds establish robustness to training randomness only.** All runs share one
   test set, so seed variance is not test-set variance; the paired p-value says nothing
   about generalisation.
4. **Calibration did not generalise.** Validation NLL improved 10/10; test ECE worsened
   7/10, test NLL 6/10, and coverage stability degraded (sd 5.67 → 19.24). Reported as an
   experiment, not an improvement. The uncalibrated results are the primary operational
   result.
5. **German vs English model comparison.** GermanFinBERT is German-pretrained and
   FinBERT English-pretrained; the English tokenizer fragments German text ≈23 % more
   heavily. The comparison is informative but not a like-for-like language match.
6. **Synthetic, template-generated data.** 2 000 rows collapse to 184 distinct texts.
   Real German SME descriptions are noisier and less consistent; results do not transfer
   directly.
7. **Every test category holds one group**, so per-class recall is 0.0 or 1.0 and
   macro-F1 has high variance; 4 categories cannot appear in test at all.
8. **8 categories have no synthetic support**, and augmentation *increases* class
   imbalance (67.5× → 89.7×).
9. **Text lower-cased**, discarding German noun capitalisation for a cased checkpoint.
10. **Metadata unused**. `amount` would separate 8 of 13 observed errors by its sign.
11. **No hyperparameter search**; configuration taken from the paper.
12. **The paper was not reproduced**, different language, 37 classes vs 11, synthetic
    vs real data, 33 test groups vs thousands of rows.

---

## 14. Changes made after the acceptance test

The checks above were run first; the following changes were made afterwards, at the
client's request or as a consequence of them. Every suite was re-run after each one, and
all of them still pass. None touched a result, a dataset, the frozen split, the
augmentation corpus, the 0.8 threshold or any model.

| # | Change | Reason | Re-verified |
|---|---|---|---|
| 1 | Inference entry point repaired (`src/inference.py`) | Defect 2 above: the documented command could not run | 4 execution paths, plus all suites |
| 2 | Engagement-specific wording neutralised in this report | Client request: the package may be read alongside a thesis | 0 remaining occurrences |
| 3 | `reference/audit_evidence/inspect_*.py` repointed at `data/` | They referenced a directory not shipped in this package, so they could not run | Both scripts reproduce their shipped reports line for line (1 760/1 760 and 163/163) |
| 4 | Documentation restyled: em-dash and emphasis density reduced | Client request, for a plainer register | Digits, headings, links, code spans and table rows verified identical in all 13 documents |
| 5 | Severity markers normalised to `(critical)`, `(high)` and so on | The previous heavy-asterisk form rendered as broken markup | 0 remaining |
| 6 | `results/` cleanup carried out in the working project | Archive plan Option A approved | All four suites plus an inference run; see `RESULTS_ARCHIVE_PLAN.md` §8 |

Change 6 affects only the working project, not this package. It deleted 1 757.6 MB of
regenerable model weights and kept every metric, prediction, log and document.

The one substantive correction is change 1. Changes 2 to 5 are presentation only, and
change 3 also fixed two scripts that could not previously run from the package.

---

## Final decision

# READY FOR DELIVERY

Two defects were found, both in the client-facing surface rather than the experiment:
documentation referencing absent scripts, and a broken inference entry point carried over
from the prototype's directory layout. Both are fixed and re-verified by execution. Every
other check passed at baseline.

The frozen split, augmentation corpus v2, 0.8 threshold, seeds, models and all reported
results are unchanged, and every final number was independently reproduced from the raw
saved logits.
