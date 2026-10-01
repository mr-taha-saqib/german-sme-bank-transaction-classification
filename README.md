# German SME Bank Transaction Classification: Academic Proof of Concept

A leakage-safe, locally reproducible implementation of the pipeline described in
*"Categorising SME Bank Transactions with Machine Learning and Synthetic Data
Generation"* (Aluffi et al., 2025, arXiv:2508.05425), adapted to German SME
transaction descriptions and a 37-class taxonomy.

* **Primary model:** GermanFinBERT (`scherrmann/GermanFinBert_FP`)
* **Comparison model:** standard FinBERT (`ProsusAI/finbert`, the paper's model)

New here? Start with [`PROJECT_INDEX.md`](PROJECT_INDEX.md), then
[`FINAL_REPORT.md`](FINAL_REPORT.md).

### What this repository contains

* German synthetic B2B transaction data (`data/`) and a gated LLM augmentation corpus
* Deterministic preprocessing of German transaction descriptions (§5)
* Label-free grouping of similar transaction descriptions (§6)
* Leakage-safe, group-based train / validation / test splitting, frozen with a checksum (§7)
* Fine-tuning and evaluation of GermanFinBERT and FinBERT under identical conditions (§9)
* Five-seed evaluation (seeds 42–46) with mean ± sd reporting (§2)
* Temperature scaling (post-hoc calibration), reported as an experiment (§9)
* Confidence-based decision routing: `automatic` (≥ 0.8) vs `manual_review` (§9)
* Threshold analysis: coverage and accuracy at 0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
  0.90 and 0.95, recorded per run (§9)
* Error analysis (`results/final_error_analysis.csv`, `FINAL_REPORT.md`)
* Full documentation, audit trail, verification scripts and a unit-test suite

Headline result (group level, 33 independent test groups, five seeds):
**GermanFinBERT 0.649 ± 0.027 accuracy vs FinBERT 0.612 ± 0.025.**

---

## 1. Objective and scope

Classify short German bank-transaction descriptions into 37 categories, measure how
reliable the model's confidence is, and route low-confidence predictions to human
review at a 0.8 threshold.

This is an academic proof of concept for a bachelor's thesis. It is deliberately
not a production system: there is no ERP integration, receipt matching, web
application, database, API or authentication. It runs locally, offline apart from the
initial Hugging Face model download, and needs no credentials of any kind.

The project's purpose is a *scientifically defensible measurement*, not a good number.
Where a result is unfavourable (and several are), it is reported as measured.

---

## 2. Final results

Group level, mean ± standard deviation over seeds 42–46. **n = 33 independent test
groups.** Generated from `results/final_results.csv`.

| Model | Calibration | Accuracy | Macro F1 | Weighted F1 | Top-2 | ECE | NLL | ≥0.8 Coverage | ≥0.8 Accuracy | Manual Review |
|---|---|---|---|---|---|---|---|---|---|---|
| **germanfinbert** | **uncalibrated** | **0.6485 ± 0.0271** | **0.5442 ± 0.0408** | **0.5772 ± 0.0433** | **0.7576 ± 0.0214** | **0.1754 ± 0.0366** | 1.4546 ± 0.0821 | **60.61 ± 5.67 %** | 0.8022 ± 0.0335 | 39.39 % |
| germanfinbert | calibrated | 0.6485 ± 0.0271 | 0.5442 ± 0.0408 | 0.5772 ± 0.0433 | 0.7576 ± 0.0214 | 0.2100 ± 0.0391 | 1.4488 ± 0.0698 | 44.24 ± 19.24 % | 0.8634 ± 0.1049 | 55.76 % |
| finbert | uncalibrated | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.5451 ± 0.0292 | 0.6788 ± 0.0271 | 0.2138 ± 0.0321 | 1.9054 ± 0.0605 | **0.00 %** | n/a | 100 % |
| finbert | calibrated | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.5451 ± 0.0292 | 0.6788 ± 0.0271 | 0.2497 ± 0.0451 | 1.9444 ± 0.0920 | **0.00 %** | n/a | 100 % |

The uncalibrated GermanFinBERT row is the primary operational result. Temperature
scaling improved its fitting objective but did not improve calibration on unseen test
data (§9), so it is reported as an experiment, not as an improvement.

Row-level figures and all 40 per-seed rows: `results/final_results.csv`.

Three things to read carefully:

* Accuracy, macro-F1, weighted-F1 and top-2 are identical between calibrated and
  uncalibrated rows. A scalar temperature is monotone and cannot reorder predictions;
  this is asserted in code.
* **FinBERT never reaches 0.8 confidence on any seed** (maximum anywhere: 0.7766), so it
  automates nothing. The threshold was not lowered to accommodate it.
* Roughly one in five automatically-processed transactions is misfiled. That, not the
  headline accuracy, is the number a deployment decision rests on.

---

## 3. Project structure

```
german-sme-bank-transaction-classification/   (delivered as GermanFinBERT_ProofOfConcept/)
├── PROJECT_INDEX.md                 start here — documentation map
├── FINAL_REPORT.md                  the complete written report
├── README.md                        this file
├── AUDIT.md                         Phase-1 audit of the client's prototype
├── DATA_PIPELINE.md                 Step 2 — preprocessing, grouping, splitting
├── STEP3_GERMANFINBERT.md           Step 3 — primary model
├── AUGMENTATION_DECISION.md         Step 3A — leakage finding and correction
├── STEP4_FINBERT.md                 Step 4 — comparison model, five seeds
├── STEP5_CALIBRATION.md             Step 5 — temperature scaling
├── RESULTS_ARCHIVE_PLAN.md          results/ inventory and the cleanup carried out
│
├── prepare_dataset.py               Step 2  — build + freeze the split
├── prepare_augmentation.py          Step 3A — build + freeze the corpus
├── train_model.py                   Step 3/4 — fine-tune ONE model
├── compare_models.py                Step 4  — aggregate the comparison
├── calibrate_models.py              Step 5  — temperature scaling
├── final_evaluation.py              Step 6  — final tables + traceability
├── verify_dataset.py                19 checks on the frozen split
├── verify_augmentation.py           11 checks on the corpus
├── tests/test_pipeline.py           66 unit checks
│
├── data/                            provided datasets (unmodified)
├── notebooks/                       ORIGINAL prototype notebook, preserved
├── reference/                       prototype results + audit evidence
├── results/                         all generated outputs
└── src/                             pipeline modules
```

The scripts listed above are the complete pipeline. Earlier phase-1 entry points
(`run_experiment.py`, `verify_results.py`, `summarize_runs.py`) did not produce the
final results and are not included in this package.

---

## 4. Datasets

| File | Role |
|---|---|
| `data/synthetic_b2b_transactions_2000.xlsx` | **primary**, 2 000 labelled transactions, 37 categories, 11 parent categories |
| `data/final_synthetic_clean_7033.csv` | LLM augmentation source (gated to 4 541 rows as corpus **v2**) |
| `data/synthetic_fold1_paper_style.csv` | not used, generated from a leaky fold |
| `data/final_new_synthetic.csv` | not used, subset of the 7 033 file |
| `data/synthetic_transactions_openai.csv` | not used, no `source_text` provenance |

The workbook is a single-column sheet holding 65 semicolon-delimited fields per row;
the loader reconstructs the columns. Selection reasoning: `DATA_PIPELINE.md` §1.

For privacy, the workbook's document metadata (author name and saved file path) was
removed before publication. Its cell contents are unchanged, and the frozen split
still regenerates with the recorded checksum.

---

## 5. Preprocessing

Fourteen numbered, deterministic transformations (`src/preprocessing.py`), deliberately
conservative: remove only what is provably uninformative.

Removed, dates (`2024-06`, `01.06.2024`), invoice/reference codes (`RE-2024-8096`,
`INV12345`), IBAN-like tokens, standalone digit runs ≥ 3.
Kept, vendor names, legal forms (`GmbH`), domain terms, short numbers (`Milestone 2`),
and protected alphanumerics (`B2B`, `365`, `Q1`, `24/7`). German banking abbreviations
are expanded (`mwst` → `umsatzsteuer`). Umlauts and `ß` are preserved. Empty results
become `nodescription` (0 rows on this data).

Effect: 1 427 distinct raw descriptions → 184 distinct cleaned; vocabulary
1 083 → 175. Full audit: `results/dataset/preprocessing_audit.json`.

---

## 6. Grouping

Two deterministic, label-free stages (`src/grouping.py`): a canonical sorted-token
key, then a union-find merge of keys at token-Jaccard ≥ 0.60.

The threshold was measured, not assumed: ≥ 0.7 merges nothing, ≥ 0.6 merges 61 pairs
with 0 crossing a category boundary, ≥ 0.5 begins fusing genuinely different
categories. 0.55 gives an identical grouping, so 0.60 sits on a plateau.

184 canonical keys → 152 groups, 0 label-impure. The target label is never used to
form groups; purity is verified afterwards and a unit test asserts that permuting the
labels leaves the grouping bit-identical.

---

## 7. Leakage-safe splitting

Groups are the atomic unit. A group lives in exactly one split, never divided.
Per category, over its distinct groups, seeded shuffle: ≥ 3 groups → 1 test,
1 validation, rest train; 2 groups → 1 test, 1 train; 1 group → train only.

| Split | Rows | Groups | Classes |
|---|---|---|---|
| train | 1 093 | 103 | 37 |
| validation | 228 | **16** | 16 |
| **test** | 647 | **33** | 33 |

Before splitting, 32 rows the source system itself flags as duplicates are removed:
26 of those 32 duplicate groups link *different* descriptions, a leak text grouping
alone cannot see.

The split is frozen to disk with a SHA-256 (`results/dataset/split_assignment.csv`,
`f715f040d8a5daa6…`). Training loads and verifies it rather than recomputing, which is
what guarantees both models see byte-identical data.

Verified: 0 shared groups, 0 shared cleaned texts, 0 same-label near-duplicate pairs
across all three split pairs.

A 4-way train/validation/calibration/test split is not affordable (only 13 of 37 categories have ≥ 4 groups), so validation doubles as the calibration set. Because no
model selection occurs, it is consumed exactly once.

---

## 8. Augmentation v2

No new generation is performed; the shipped LLM corpus is reused under three gates:

| Gate | Rejects | Dropped |
|---|---|---|
| **A** provenance | rows whose `source_text` resolves outside training | 2 483 |
| **B** collision | text equal to, or sharing a group key with, a held-out pattern | 0 |
| **C** near-duplicate | text reaching Jaccard ≥ 0.60 against a held-out pattern | **9** |
| | **7 033 → 4 541** | |

Gate C was added in Step 3A after verification found 9 rows that were near-copies of two
held-out *test* patterns, leakage by the project's own grouping definition.
`AUGMENTATION_DECISION.md` records the finding, the fix, and the honest outcome that
removing them changed no prediction.

The corpus is frozen with its own checksum (`4eb242d9f384041f…`) and loaded, not rebuilt.
Combined training corpus: 1 093 real + 4 541 synthetic = 5 634 rows, 29 of 37
categories with synthetic support.

---

## 9. Models, seeds, calibration and threshold

Identical treatment. Both models share the frozen split, corpus v2, label mapping,
loss, optimiser, schedule, seeds, metrics and threshold. Only the checkpoint and its
tokenizer differ. Verified mechanically across 12 hyperparameters before any comparison
table is produced.

Configuration (reference paper, Table 3): lr 2e-5, batch 16, max_len 128, 3 epochs,
AdamW (wd 0.01), linear warmup 10 % → decay, grad-clip 1.0, class-weighted focal loss
γ = 2, mixed precision, `model_selection = fixed_epochs` (no early stopping, so
validation stays pristine).

Five seeds (42, 43, 44, 45, 46) fixed before any test evaluation. No seed was
selected on test performance; all five are reported.

Temperature scaling is fitted per run on validation logits only and applied
unchanged to test. Test labels are never used to fit or select a temperature.
Result: validation NLL improved in 10/10 runs, but **test ECE worsened in 7/10 and test
NLL in 6/10**, and the operating point became unstable (coverage sd 5.67 → 19.24). It is
reported as an experiment that did not generalise, not as an improvement.

The 0.8 threshold is fixed and was never adjusted to improve any model's coverage.

Decision routing has two outcomes: predictions with confidence ≥ 0.8 are
`automatic`, everything else is `manual_review` (`src/evaluation.py`,
`src/inference.py`). As a sensitivity analysis, every run also records coverage,
accuracy among automatic decisions and manual-review share at thresholds 0.50, 0.60,
0.70, 0.75, 0.80, 0.85, 0.90 and 0.95, at row and group level
(`threshold_sweep` in `results/step4_comparison/<run>/test_metrics.json` and in
`results/step5_calibration/*_calibration.json`). The sweep is descriptive only and
was not used to choose the operating threshold.

---

## 10. Setup

Requires Python 3.10+. Verified on Python 3.13.2, Windows 11, NVIDIA RTX 3060
(12 GB), CUDA 12.6.

```bash
python -m venv .venv
.venv\Scripts\activate                        # Windows
# source .venv/bin/activate                   # Linux / macOS

# 1) PyTorch first — the CUDA build is not on PyPI
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0+cu126
#    CPU-only alternative (slower, pipeline unchanged):
#    pip install torch==2.13.0

# 2) everything else
pip install -r requirements.txt
```

Check the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU is optional. CUDA is used automatically when available and the code falls back
to CPU otherwise. On limited VRAM, batch size and gradient accumulation multiply to the
effective batch, so the paper's 16 is reachable on a smaller card:

```bash
python train_model.py --model germanfinbert --batch-size 8 --grad-accum 2
python train_model.py --model germanfinbert --batch-size 4 --grad-accum 4 --no-amp
```

At default settings a run peaks around 3.5 GB of VRAM. First run downloads ~880 MB of
pre-trained weights to `~/.cache/huggingface`; everything after that is offline.

No credentials are required. The LLM corpora were generated beforehand and ship as
CSV. The pipeline makes no API calls, and no key belongs anywhere in this repository.

---

## 11. Reproducing everything

```bash
# Step 2 — prepare data, group, split, freeze
python prepare_dataset.py
python verify_dataset.py                        # 19 checks

# Step 3A — build and freeze augmentation corpus v2
python prepare_augmentation.py --version v2
python verify_augmentation.py                   # 11 checks

# Steps 3/4 — train both models across five seeds
for SEED in 42 43 44 45 46; do
  for MODEL in germanfinbert finbert; do
    python train_model.py --model $MODEL --augment --augmentation-version v2 \
        --seed $SEED --output-dir results/step4_comparison \
        --run-name ${MODEL}_seed${SEED}
  done
done

# Step 4 — aggregate (re-checks fairness preconditions first)
python compare_models.py

# Step 5 — temperature scaling on saved validation logits (no GPU, seconds)
python calibrate_models.py

# Step 6 — final tables + traceability audit
python final_evaluation.py

# Test suite
python tests/test_pipeline.py                   # 66 checks
```

Runtime ≈ 2 minutes per training run on an RTX 3060, ≈ 25 minutes end to end.

Seeds are set for `random`, `numpy`, `torch` and CUDA, with a seeded `DataLoader`
generator and `cudnn.deterministic = True`. Both the split and the corpus are
checksum-verified on load, so a silently changed dataset aborts the run rather than
producing an incomparable result. The checksums were recorded on Windows; the
checksum helpers pin CRLF line endings so that the same recorded values verify on
Linux and macOS too (data and results are unchanged). Verified: an independent re-run reproduced accuracy,
macro-F1 and top-2 exactly.

### Where results are written

| Path | Contents |
|---|---|
| `results/dataset/` | frozen split, corpus v2, manifests, checksums, class distribution |
| `results/step4_comparison/<model>_seed<N>/` | per-run logits, predictions, metrics, confusion matrix, and weights in `model/` |
| `results/step5_calibration/` | fitted temperatures, calibrated probabilities, 20 reliability diagrams |
| `results/final_results.csv` / `.json` | 40-row final table + integrity, traceability, error analysis |
| `results/final_summary.csv` / `.xlsx` | mean ± sd summary |
| `results/final_comparison.md` | the written comparison |

### Inference on new descriptions

Model weights are not shipped (~420 MB per run - see `MANIFEST.md`), so train one
first. This writes `model/` into the run directory and takes about 2 minutes on the
RTX 3060:

```bash
python train_model.py --model germanfinbert --seed 42 --augment --run-name my_run
```

Then classify new descriptions:

```bash
python -m src.inference --run results/my_run \
    --text "SaaS Subscription 2025-04" --text "Gehaltslauf 2026-01"

python -m src.inference --run results/my_run \
    --csv reference/example_new_transactions.csv --column description --out predictions.csv
```

Each prediction carries a confidence and a decision of `automatic` (≥ 0.8) or
`manual_review`.

Confidences are uncalibrated by default, which is the primary operational result
here: temperature scaling improved validation likelihood but did not generalise to the
test set. To apply a fitted temperature anyway, pass it explicitly:

```bash
python -m src.inference --run results/my_run \
    --calibration results/step5_calibration/germanfinbert_seed42_calibration.json \
    --text "SaaS Subscription 2025-04"
```

---

## 12. Limitations

These bound what the results can support. They are properties of the provided data, not
defects in the implementation.

1. **n = 33.** The test set holds 33 independent groups, so one group is worth 3.03
   accuracy points and a 95 % CI on 0.65 spans roughly ±0.16: over four times the
   observed 3.6-point model gap.
2. **Every test category holds one group**, so per-class recall can only be 0.0 or 1.0
   and macro-F1 has high variance. 4 categories cannot appear in test at all.
3. **The calibration set has 16 groups covering 16 of 37 classes.** This is the direct
   cause of temperature scaling failing to generalise (§9).
4. **Seed variance is not test-set variance.** Five seeds establish robustness to
   initialisation only; all runs share one test set.
5. **The data is synthetic and template-generated**. 2 000 rows collapse to 184 distinct
   texts. Real German SME descriptions are noisier and less consistent; results do not
   transfer directly.
6. **8 categories have no synthetic support**, and the corpus *increases* class imbalance
   (67.5× → 89.7×), the opposite of the paper's inverse-frequency intent.
7. **All text is lower-cased** for consistency with the already-lower-cased corpus,
   discarding German noun capitalisation for the cased GermanFinBERT checkpoint.
8. **Metadata is unused.** `amount` would resolve 8 of 13 observed errors by its sign but
   is excluded by the agreed text-only scope.
9. **No hyperparameter search.** Configuration comes from the paper; nothing was tuned on
   validation or test.

### Not claimed

* **This project does not reproduce the paper.** Different language, 37 classes vs 11,
  synthetic vs real bank data, 33 test groups vs thousands of rows. See
  `FINAL_REPORT.md` §14.
* **Confidence ≥ 0.8 does not mean a prediction is correct**. About one in five
  automated decisions is wrong.
* **GermanFinBERT is not claimed to universally outperform FinBERT**, only that, under
  these conditions and on this test set, it did so on every metric and every seed.
* **Calibration is not presented as an improvement.**
* **No three-level decision scheme is implemented.** The pipeline routes to
  `automatic` or `manual_review` only. An intermediate "suggested category, user
  verifies" level is not part of the delivered code or results; the threshold sweep
  (§9) is the evidence such a design could build on.
* **Not a production banking system.** No production API, ERP integration, receipt
  processing or deployment is included.
