# Step 3, GermanFinBERT: Implementation, Training and First Test Evaluation

> Part of the final Step 2-6 pipeline. The augmented figures in §6 predate the Step-3A gate-C correction; the corrected values are in [`AUGMENTATION_DECISION.md`](AUGMENTATION_DECISION.md) §5.

Primary classification model for the German transaction classification task, trained
on the frozen Step-2 split. **No calibration was applied and no FinBERT comparison was
run**. Those are Steps 4 and 5.

```bash
python train_model.py --model germanfinbert --run-name step3_germanfinbert
```

---

## 1. Exact model used

`scherrmann/GermanFinBert_FP`. The checkpoint the client's own notebook loads.

Provenance of the choice, not an invention. It appears at cells 53 and 88 of
`notebooks/finbert_transaction_classification_sanitized.ipynb` and is the only German
BERT anywhere in the project. The notebook's other model references are
`ProsusAI/finbert` (the English comparison model, Step 4), plus
`seduerr/t5_base_paws_ger` and `google/flan-t5-small`, which are paraphrase/generation
experiments abandoned in the notebook and irrelevant here. `AUDIT.md` §1 records the
same identification.

| Property | Value |
|---|---|
| Checkpoint | `scherrmann/GermanFinBert_FP` |
| Base model | `deepset/gbert-base` (German BERT) |
| Pre-training head | `BertForMaskedLM`, further-pretrained on German financial text |
| Tokenizer | `BertTokenizer`, vocabulary **31 102**, **cased** (`do_lower_case=False`) |
| Parameters | 110.0 M |
| Max position embeddings | 512 |
| Classification head | freshly initialised, 37 outputs, `ignore_mismatched_sizes=True` |

### The ambiguity, stated

The `scherrmann` family publishes more than one GermanFinBERT artefact, notably an
`_FP` ("further pre-trained") variant and task-fine-tuned siblings such as `_SC`
(sentiment classification). They are not interchangeable: a sentiment-fine-tuned
checkpoint carries a 3-class head trained for a different objective.

`_FP` is both what the notebook specifies and the methodologically correct choice,
because it is a masked-language-model checkpoint with no task head: exactly the right
starting point for attaching a fresh 37-class classifier. It is used unchanged. Nothing
was substituted.

One friction worth recording: the checkpoint is cased, but the Step-2
preprocessing lower-cases all text (for consistency with the already-lower-cased
augmentation corpus). The model therefore never sees German noun capitalisation, a
signal its pre-training could have exploited. `preprocessing.lowercase = false` makes
this testable, at the cost of a train/test casing mismatch with the synthetic corpus.

---

## 2. Data: the frozen Step-2 split, loaded not recomputed

`train_model.py` loads `results/dataset/split_assignment.csv` and verifies its SHA-256
against `split_manifest.json` before doing anything else. If the split file ever
changes, training aborts. This is what will guarantee in Step 4 that FinBERT sees
byte-identical data.

```
Frozen split verified (sha256 f715f040d8a5daa6…)
1968 rows | 152 groups | train 1093r/103g | validation 228r/16g | test 647r/33g
```

| Split | Rows | Groups | Used for |
|---|---|---|---|
| train | 1 093 | 103 | gradient updates only |
| validation | 228 | 16 | **monitoring only**, see §4 |
| test | 647 | 33 | evaluated once, after training finished |

The label mapping is built by sorting the full 37-category inventory across all
splits, so integer *k* means the same category for every model and every run. Saved to
`label_mapping.json`.

---

## 3. Training configuration

Reference-paper Table 3 values, with the two corrections `AUDIT.md` justified (focal
loss form P10, warmup as a ratio P9).

| Item | Value |
|---|---|
| Model checkpoint | `scherrmann/GermanFinBert_FP` |
| Tokenizer | `BertTokenizer`, vocab 31 102, cased |
| Learning rate | 2e-5 |
| Batch size | 16 (configurable via `--batch-size` / `--grad-accum`) |
| Max sequence length | 128 |
| Epochs | 3 |
| Optimizer | AdamW, weight decay 0.01 |
| Scheduler | linear warmup → linear decay; warmup = 10 % of steps (**20** of 207) |
| Gradient clipping | max-norm 1.0 |
| Loss | class-weighted focal loss, γ = 2.0 |
| Class weights | balanced inverse-frequency over the training split |
| Random seed | 42 (`random`, `numpy`, `torch`, CUDA; seeded DataLoader; `cudnn.deterministic`) |
| Mixed precision | enabled (`torch.amp`, CUDA only), stable: no overflow observed |
| Device | `cuda:0` NVIDIA GeForce RTX 3060, 12.0 GB, torch 2.13.0+cu126 |
| Training size | 1 093 rows, 37 classes |
| Validation size | 228 rows / 16 groups / 16 classes |
| Total optimisation steps | 207 |
| Training time | 26 s |

`max_length=128` rather than the paper's 256: measured p95 token length is 15, so
128 truncates nothing (0 truncated, 0 `[UNK]`) and halves compute.

VRAM. `--batch-size` and `--grad-accum` multiply to the effective batch, so the
paper's 16 is reachable on a smaller card (`--batch-size 8 --grad-accum 2`, or
`--batch-size 4 --grad-accum 4 --no-amp` for very tight memory). At the default
settings the run peaks around 3.5 GB of the 12 GB available.

### Should the paper's weighted focal loss be reproduced exactly?: yes, and here is the evidence

The paper specifies weighted focal loss (γ = 2) for class imbalance. It is reproduced
exactly, in the mathematically correct form
`L = -α_t (1 - p_t)^γ log(p_t)`. Not the prototype's `p_t = exp(-CE_weighted)`, which
is only equal when the class weight is 1 (`AUDIT.md` P10).

To document how much that choice actually matters, a controlled variant was run with
class-weighted cross-entropy (identical in every other respect. Cross-entropy is
exactly focal loss with γ = 0, so the same implementation covers both):

| Loss | Group-level accuracy | Macro F1 | Mean confidence | Max confidence |
|---|---|---|---|---|
| **weighted focal, γ=2 (paper, primary)** | 0.3939 | 0.2924 | 0.1721 | 0.4210 |
| weighted cross-entropy (control) | 0.4242 | 0.3238 | 0.1763 | 0.4345 |

The difference is one test pattern (1/33 = 3.03 points), inside noise for this
sample size, and confidence behaviour is essentially identical.

Conclusion: reproduce the paper's loss. It costs nothing measurable, and fidelity
to the reference method is worth more than an unmeasurable difference. Importantly,
the choice was fixed *a priori* from the paper; the control run documents sensitivity
and was not used to select the configuration.

---

## 4. How the validation split was used

`model_selection = "fixed_epochs"`: exactly 3 epochs, **no checkpoint selection, no
early stopping**.

Validation metrics are computed each epoch and written to the training history, but
no configuration, checkpoint or hyperparameter depends on them. The epoch count
came from the paper before training started and was not revised after seeing
validation numbers.

This is a deliberate trade-off. Selecting the best epoch on validation would very
likely have helped, the augmented variant's validation macro-F1 peaks at epoch 2
(0.7778) and falls back at epoch 3 (0.6899). But it would consume the validation
split, which Step 5 needs as an untouched calibration set. With only 16 validation
groups covering 16 of 37 classes, spending it on epoch selection and then calibrating
on the same data would be the P4 mistake the audit flagged in the prototype.

`--early-stopping` implements the alternative if the client prefers it; the trade-off
is recorded here and in the run's `config.json`.

The test split was not touched until training was complete, and the 0.8 threshold
was applied once at reporting time. It influenced nothing.

---

## 5. Results: first test evaluation (uncalibrated)

Raw softmax probabilities. No temperature scaling.

### Headline

| Metric | Group level (n=33 independent groups) | Row level (n=647) |
|---|---|---|
| Accuracy | **0.3939** | 0.2535 |
| Macro F1 | **0.2924** | 0.2623 |
| Weighted F1 | 0.3101 | 0.2156 |
| Top-2 accuracy | 0.4242 | 0.2906 |

Group level is the honest sample size: 647 test rows collapse to 33 independent
groups. Both are reported throughout.

Validation, for reference (monitoring only): row-level accuracy 0.2500, macro-F1
0.4185; group-level accuracy 0.5625, macro-F1 0.4298 over 16 groups.

### Confidence statistics

| Statistic | Group level | Row level |
|---|---|---|
| Mean | 0.1721 | 0.1452 |
| Median | 0.1416 | 0.1137 |
| Std | 0.1101 | 0.0889 |
| Min | 0.0519 | 0.0519 |
| **Max** | **0.4210** | 0.4210 |

### The 0.8 confidence workflow

| Quantity | Group level | Row level |
|---|---|---|
| Predictions with confidence ≥ 0.8 | **0 (0.0 %)** | **0 (0.0 %)** |
| Accuracy among those | undefined | undefined |
| Routed to manual review | 33 (100 %) | 647 (100 %) |

**The model never reaches 0.8 confidence, its single most confident prediction scores
0.42.** At the requested operating point this configuration automates nothing; every
transaction would go to a human. The full sweep from 0.50 to 0.95 returns 0 % coverage
at every threshold.

This is reported as measured. Nothing was tuned to move it.

### Per-class behaviour

13 of 33 test categories are predicted correctly; the remaining 20 have recall 0.
Because every test category holds exactly one group, per-class recall can only be 0.0
or 1.0: the structural consequence documented in `DATA_PIPELINE.md` §7. Precision
varies widely (`Dienstleistungen` 0.019 with recall 1.0) because a few over-predicted
classes absorb many rows. Full table in `classification_report_test.txt`, confusion
matrix in `confusion_matrix_test.csv` / `.png`.

---

## 6. Documented variant: the paper's full pipeline including augmentation

Step 3 as specified trains on the Step-2 training split alone. But the reference
paper's pipeline has synthetic augmentation as a core component (`Generate(·)`,
paper §3.1.3), and reporting the un-augmented number alone would misrepresent what
this pipeline does. One additional run, using the Step-2 provenance gate so that only
synthetic examples derived from *training* patterns are admitted:

```
Gate A (provenance): kept 4550/7033 rows; 2483 dropped, source pattern held out
Gate B (collision):  kept 4550/4550; 0 colliding with a validation/test pattern
Training corpus: 1093 real + 4550 synthetic = 5643 rows
```

| Configuration | Group acc | Macro F1 | Top-2 | Mean conf | ≥0.8 | Acc ≥0.8 |
|---|---|---|---|---|---|---|
| **primary, no augmentation** | 0.3939 | 0.2924 | 0.4242 | 0.172 | **0.0 %** | – |
| variant, cross-entropy loss | 0.4242 | 0.3238 | 0.4242 | 0.176 | 0.0 % | – |
| **variant, with augmentation** | **0.6667** | **0.5638** | **0.7273** | **0.747** | **57.6 %** | **0.7895** |

Augmentation is worth +27 accuracy points at group level, and it is what makes the
0.8 workflow function at all: mean confidence rises from 0.17 to 0.75 and the
auto-processing rate from 0 % to 57.6 %, at 78.9 % accuracy among those.

Correction to an earlier hypothesis. The near-zero confidence initially looked like
the focal loss suppressing peak probabilities via its `(1-p_t)^γ` term. The
cross-entropy control disproves that: with γ = 0 the mean confidence is 0.176, all but
identical. The cause is the size of the training signal (1 093 rows and 207 optimisation steps across 37 classes) not the loss function.

The client should decide whether Steps 4 and 5 run on the augmented or un-augmented
corpus. Whichever is chosen must be applied identically to FinBERT, or the
comparison is void.

---

## 7. Saved artefacts

```
results/step3_germanfinbert/
├── model/
│   ├── model.safetensors          fine-tuned weights (best/final checkpoint)
│   ├── config.json                architecture + id2label / label2id
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── label_mapping.json             class_names, label2id, n_classes
├── model_meta.json                checkpoint, tokenizer, full training config,
│                                  per-epoch history, optimizer, scheduler, loss,
│                                  device, data sizes, class distribution,
│                                  frozen-split sha256, environment
├── validation_metrics.json        monitoring metrics (no decision taken from them)
├── test_metrics.json              all reported metrics, both levels
├── test_predictions.csv           per-example prediction, confidence, decision
├── confusion_matrix_test.csv/.png
├── classification_report_test.txt per-class precision / recall / F1
├── threshold_curve_test.png
├── config.json                    exact ExperimentConfig used
└── train.log
```

`model_meta.json` records `"calibration": "NOT APPLIED - Step 3 reports raw softmax
probabilities only"`.

Variant runs: `results/step3_variant_crossentropy/`, `results/step3_variant_augmented/`
(metrics only; weights not saved).

---

## 8. Limitations

1. **The 0.8 threshold is unusable in the primary configuration.** Peak confidence is
   0.42, so the workflow automates 0 %. Two things can change this and both belong to
   later steps: augmentation (§6, already measured) and temperature scaling with T < 1
   (Step 5), which sharpens probabilities. Neither has been applied here.
2. **n = 33.** One test group is worth 3.03 accuracy points. Differences smaller than
   ~6 points are not interpretable, which is exactly why the loss comparison in §3 is
   reported as "no material difference" rather than as a winner.
3. **Every test category holds one group**, so per-class recall is 0 or 1 and macro-F1
   has very high variance.
4. **4 categories are absent from the test set** and 21 from validation. A structural
   property of the corpus recorded in `DATA_PIPELINE.md` §7.
5. **Single seed.** These are seed-42 numbers. The earlier full-pipeline work found the
   model ordering on top-1 accuracy can flip between seeds at this sample size, so
   Step 4 should run several seeds before concluding anything about GermanFinBERT vs
   FinBERT.
6. **Cased checkpoint, lower-cased input** (§1). Noun capitalisation is discarded.
7. **Validation is small and unrepresentative** (16 groups, 16 of 37 classes), which
   limits how much Step 5's calibration can be trusted.
8. **No hyperparameter search was performed.** Values come from the paper. Nothing was
   tuned on validation or test.

---

## 9. Reproducibility

```bash
# 0. environment
python -m venv .venv && .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0+cu126
pip install -r requirements.txt

# 1. Step 2 - build and freeze the split (must run first)
python prepare_dataset.py
python verify_dataset.py            # 19 checks

# 2. Step 3 - train GermanFinBERT
python train_model.py --model germanfinbert --run-name step3_germanfinbert

# variants reported in section 6
python train_model.py --model germanfinbert --loss cross_entropy \
    --run-name step3_variant_crossentropy --no-save-weights
python train_model.py --model germanfinbert --augment \
    --run-name step3_variant_augmented --no-save-weights

# limited VRAM
python train_model.py --model germanfinbert --batch-size 8 --grad-accum 2
```

Seed 42 throughout. The frozen split is checksum-verified on load, so a silently
changed dataset aborts the run rather than producing an incomparable result. Expected
runtime on an RTX 3060: ~26 s primary, ~2 min augmented, plus first-time model download
(~440 MB).

`tests/test_pipeline.py` (66 checks) covers preprocessing, grouping, splitting,
augmentation gating and evaluation arithmetic.
