# Step 4: Standard FinBERT Comparison (Uncalibrated)

> Part of the final Step 2-6 pipeline. Index: [`PROJECT_INDEX.md`](PROJECT_INDEX.md).

Multi-seed comparison of the primary model (GermanFinBERT) against the reference
paper's model (standard FinBERT), on identical data under identical conditions.

No temperature scaling was applied. All results here are uncalibrated raw softmax.

```bash
python train_model.py --model finbert --augment --augmentation-version v2 \
    --seed 42 --output-dir results/step4_comparison --run-name finbert_seed42
python compare_models.py
```

---

## 1. Exact checkpoint

`ProsusAI/finbert`. The checkpoint used by the reference paper (Table 3, "Base
Model: ProsusAI/finbert") and by the client's notebook at nine separate cells
(5, 35, 45, 48, 52, 78, 85, 115, 138). Verified before training.

| Property | Standard FinBERT | GermanFinBERT (primary) |
|---|---|---|
| Checkpoint | `ProsusAI/finbert` | `scherrmann/GermanFinBert_FP` |
| Origin | `finbertTRC2` (English financial corpus) | `deepset/gbert-base` further-pretrained |
| Shipped head | `BertForSequenceClassification`, 3-class sentiment (positive/negative/neutral) | `BertForMaskedLM` (no task head) |
| Tokenizer | `BertTokenizer`, vocab **30 522**, **uncased** | `BertTokenizer`, vocab **31 102**, cased |
| Parameters | 109.5 M | 110.0 M |
| Head for this task | discarded, freshly initialised 37-class head (`ignore_mismatched_sizes=True`) | same |

Note the asymmetry, which is a property of the checkpoints rather than a choice: FinBERT
ships a 3-class sentiment head that must be thrown away, while GermanFinBERT ships no
task head at all. Both receive an identically-initialised fresh 37-class head, so
neither is advantaged by the replacement.

---

## 2. Fairness: verified mechanically, not asserted

`compare_models.py` refuses to produce a comparison table unless every run agrees on
split, corpus, labels, threshold and training configuration. All 10 runs passed.

| Shared component | Value |
|---|---|
| Frozen split | sha256 `f715f040d8a5daa621164ae8` (loaded, never recomputed) |
| Augmentation corpus | **v2**, sha256 `4eb242d9f384041fc5ec67ce` |
| Gates on that corpus | A provenance, B collision, **C near-duplicate** |
| Training corpus | 1 093 real + 4 541 synthetic = **5 634 rows** |
| Validation | 228 rows / 16 groups, monitoring only |
| Test | 647 rows / **33 groups** |
| Label mapping | identical 37-class mapping, byte-compared across runs |
| Confidence threshold | 0.80 |
| Training config | identical on epochs, lr, batch, max_length, loss, focal γ, class weights, warmup, model selection, gradient accumulation |

The test set was used only for the final evaluation. No model selection, hyperparameter
tuning or threshold choice touched it.

---

## 3. Training configuration

Identical for both models. The only difference is the checkpoint and its tokenizer.

| Item | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 (grad accumulation 1) |
| Max sequence length | 128 |
| Epochs | 3 |
| Optimizer | AdamW, weight decay 0.01 |
| Scheduler | linear warmup (10 % = 105 steps) → linear decay |
| Gradient clipping | max-norm 1.0 |
| Loss | class-weighted focal loss, γ = 2.0 |
| Class weights | balanced inverse-frequency over the training corpus |
| Model selection | `fixed_epochs`. None; validation stays pristine for Step 5 |
| Mixed precision | enabled (CUDA), stable |
| Device | `cuda:0` NVIDIA GeForce RTX 3060, 12.0 GB, torch 2.13.0+cu126 |
| Training steps | 353 per epoch × 3 = **1 059** |
| Seeds | **42, 43, 44, 45, 46** |

The focal-loss methodology is unchanged from GermanFinBERT. `AUDIT.md` (P10) and
`STEP3_GERMANFINBERT.md` §3 established it as the paper's specified loss, and the
cross-entropy control there showed the choice is not material on this corpus. No
model-specific tuning of any kind was introduced.

The seed list was fixed before any test evaluation and no seed was discarded.

---

## 4. Results per seed: group level (n = 33 independent groups)

| Model | Seed | Accuracy | Macro F1 | Weighted F1 | Top-2 | Mean conf | Max conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|
| finbert | 42 | 0.6061 | 0.5038 | 0.5343 | 0.6667 | 0.4581 | 0.7766 | **0.0** | – |
| finbert | 43 | 0.6364 | 0.5429 | 0.5758 | 0.6970 | 0.4514 | 0.7200 | **0.0** | – |
| finbert | 44 | 0.5758 | 0.4602 | 0.5020 | 0.6364 | 0.4726 | 0.7220 | **0.0** | – |
| finbert | 45 | 0.6061 | 0.5000 | 0.5455 | 0.6970 | 0.4789 | 0.6938 | **0.0** | – |
| finbert | 46 | 0.6364 | 0.5204 | 0.5677 | 0.6970 | 0.4734 | 0.7295 | **0.0** | – |
| germanfinbert | 42 | 0.6667 | 0.5638 | 0.5980 | 0.7576 | 0.7796 | 0.9313 | 63.6 | 0.8095 |
| germanfinbert | 43 | 0.6667 | 0.5857 | 0.6212 | 0.7576 | 0.7597 | 0.9254 | 66.7 | 0.7727 |
| germanfinbert | 44 | 0.6061 | 0.4857 | 0.5152 | 0.7576 | 0.7310 | 0.9217 | 63.6 | 0.7619 |
| germanfinbert | 45 | 0.6364 | 0.5190 | 0.5505 | 0.7273 | 0.7367 | 0.9219 | 54.5 | 0.8333 |
| germanfinbert | 46 | 0.6667 | 0.5667 | 0.6010 | 0.7879 | 0.7158 | 0.9285 | 54.5 | 0.8333 |

### Row level (n = 647)

| Model | Seed | Accuracy | Macro F1 | Weighted F1 | Top-2 | Mean conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|---|
| finbert | 42 | 0.4173 | 0.4808 | 0.3630 | 0.5580 | 0.4188 | 0.0 | – |
| finbert | 43 | 0.4266 | 0.5039 | 0.3585 | 0.5672 | 0.3737 | 0.0 | – |
| finbert | 44 | 0.4019 | 0.4255 | 0.3370 | 0.4992 | 0.3876 | 0.0 | – |
| finbert | 45 | 0.4606 | 0.4683 | 0.4116 | 0.5827 | 0.4031 | 0.0 | – |
| finbert | 46 | 0.4853 | 0.5028 | 0.4155 | 0.5827 | 0.3983 | 0.0 | – |
| germanfinbert | 42 | 0.4946 | 0.5375 | 0.4377 | 0.6291 | 0.6844 | 42.8 | 0.7437 |
| germanfinbert | 43 | 0.4281 | 0.5602 | 0.3760 | 0.6291 | 0.6345 | 45.1 | 0.6233 |
| germanfinbert | 44 | 0.4250 | 0.4414 | 0.3389 | 0.6291 | 0.6015 | 42.2 | 0.5458 |
| germanfinbert | 45 | 0.4343 | 0.4671 | 0.3569 | 0.5920 | 0.6117 | 32.6 | 0.6967 |
| germanfinbert | 46 | 0.4745 | 0.5508 | 0.3972 | 0.6677 | 0.5728 | 36.3 | 0.6255 |

---

## 5. Mean ± standard deviation across 5 seeds

Group level (headline):

| Model | Accuracy | Macro F1 | Weighted F1 | Top-2 | Mean conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|
| **germanfinbert** | **0.6485 ± 0.0271** | **0.5442 ± 0.0408** | **0.5772 ± 0.0433** | **0.7576 ± 0.0214** | 0.7446 ± 0.0252 | **60.61 ± 5.67** | 0.8022 ± 0.0335 |
| finbert | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.5451 ± 0.0292 | 0.6788 ± 0.0271 | 0.4669 ± 0.0116 | **0.00 ± 0.00** | – |

Row level:

| Model | Accuracy | Macro F1 | Weighted F1 | Top-2 | Mean conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|
| germanfinbert | 0.4513 ± 0.0313 | 0.5114 ± 0.0535 | 0.3813 ± 0.0383 | 0.6294 ± 0.0268 | 0.6210 ± 0.0418 | 39.81 ± 5.17 | 0.6470 ± 0.0760 |
| finbert | 0.4383 ± 0.0340 | 0.4763 ± 0.0321 | 0.3771 ± 0.0347 | 0.5580 ± 0.0345 | 0.3963 ± 0.0169 | 0.00 ± 0.00 | – |

Seed-42 comparability confirmed: the Step-4 GermanFinBERT seed-42 run reproduces the
Step-3A result exactly (0.66667 / 0.56381 / 0.59798 / 0.75758 on all four metrics),
so the two steps are directly comparable and training is deterministic across processes.

---

## 6. Interpretation: what can and cannot be claimed

### The confidence difference is categorical, not marginal

FinBERT never reaches 0.80 confidence on any seed. Its maximum across all five runs
is 0.7766; its per-seed maxima are 0.777, 0.720, 0.722, 0.694, 0.730. GermanFinBERT
automates 60.6 % ± 5.7 of test groups at 80.2 % accuracy.

This is not a noise question. At the client's requested operating point, FinBERT would
automate nothing while GermanFinBERT automates roughly three fifths of traffic. For
the human-in-the-loop workflow this is the difference between a usable system and one
that adds no value over full manual review. It is also the clearest single reason to
prefer the German model, and it is entirely invisible to an accuracy-only comparison.

### The accuracy difference is consistent but small: and must not be over-claimed

GermanFinBERT is ahead on 5 of 5 seeds on accuracy, macro-F1 and top-2, with a
consistent sign:

| Metric | Mean difference (German − FinBERT) | Seeds ahead |
|---|---|---|
| Accuracy | **+0.0364** (sd 0.0135) | 5/5 |
| Macro F1 | +0.0387 (sd 0.0165) | 5/5 |
| Top-2 | +0.0788 (sd 0.0345) | 5/5 |
| ≥0.8 % | +60.61 (sd 5.67) | 5/5 |

Paired tests over the 5 seeds: t = 6.00, p = 0.0039; Wilcoxon signed-rank p = 0.0625;
sign test p = 0.0625.

**This does not establish that GermanFinBERT is better in general, and the t-test result
should not be quoted as if it did.** The reason is that all ten runs share the *same*
33-group test set, so the paired test measures only robustness to training randomness:
it treats the test set as fixed and therefore models none of the sampling uncertainty
that actually dominates here. A 95 % confidence interval on a single group-level accuracy
of 0.65 with n = 33 spans roughly ±0.16, which is more than four times the 0.036 gap.

The defensible statement is:

> On this test set and under identical conditions, GermanFinBERT outperformed standard
> FinBERT on every metric and every seed, and the mean accuracy advantage of 3.6 points
> (1.2 of 33 test groups) was robust to training randomness. The test set is too small
> to establish that this advantage generalises.

The confidence and top-2 findings are stronger, because their effect sizes (60.6 points
and 7.9 points) are large relative to both seed variance and test-set uncertainty.

### Why the German model behaves differently: measured

| Model | Subtokens per word | Mean tokens | UNK |
|---|---|---|---|
| germanfinbert | 2.54 | 8.6 | 0 |
| finbert | 3.13 | 10.1 | 0 |

The English FinBERT vocabulary fragments German transaction text ≈23 % more heavily.
Neither tokenizer emits `[UNK]`, so the disadvantage is fragmentation rather than
out-of-vocabulary loss. This is consistent with FinBERT's lower confidence: more
fragmented inputs give the classifier a less concentrated signal.

---

## 7. Saved artefacts

`results/step4_comparison/<model>_seed<N>/` for each of the 10 runs:

```
logits.npz                  test + validation logits, probabilities, labels, group_ids
test_predictions.csv        per-example prediction, confidence, decision at 0.8
test_metrics.json           all metrics, group level and row level
validation_metrics.json     monitoring metrics
confusion_matrix_test.csv   full 37-class confusion matrix
confusion_matrix_test.png
classification_report_test.txt   per-class precision / recall / F1
threshold_curve_test.png
model_meta.json             checkpoint, tokenizer, config, history, checksums, device
label_mapping.json
config.json, train.log
model/                      weights + tokenizer (seed 42 only, to limit disk use)
```

Aggregates at `results/step4_comparison/`:
`per_seed_group.csv`, `per_seed_row.csv`, `summary_group.csv`, `summary_row.csv`,
`comparison.json` (includes the paired analysis and Markdown tables).

`logits.npz` exists specifically so that **Step 5 can fit temperature scaling on the
saved validation logits without retraining**.

---

## 8. Limitations

1. **n = 33.** One test group is worth 3.03 accuracy points. The 3.6-point mean accuracy
   gap is 1.2 groups. Test-set sampling uncertainty (±0.16) dwarfs it.
2. **Seed variance is not test-set variance.** Five seeds establish robustness to
   initialisation and data ordering only. All runs share one test set, so the paired
   p-value says nothing about generalisation to other SME transactions.
3. **Every test category holds one group**, so per-class recall is 0 or 1 and macro-F1
   has high variance: visible in the ±0.041 / ±0.054 standard deviations.
4. **4 categories are absent from the test set entirely**, and 8 categories have no
   synthetic augmentation, so the comparison covers 33 of 37 classes unevenly.
5. **Uncalibrated.** These are raw softmax outputs. Step 5 may change the confidence
   picture substantially, in particular, temperature scaling with T < 1 could lift
   FinBERT above the 0.8 threshold, so the 0 % automation figure is a property of the
   *uncalibrated* model and should not yet be reported as final.
6. **A single dataset**, synthetic and template-generated. Results do not transfer
   directly to real German SME bank data.
7. **No hyperparameter search for either model.** Configuration comes from the paper.
   It is possible that FinBERT would benefit from different settings; testing that would
   require a tuning budget spent equally on both, which is outside the agreed scope.

---

## 9. Reproducibility

```bash
python prepare_dataset.py          # Step 2, freezes the split
python prepare_augmentation.py --version v2   # Step 3A, freezes the corpus

for SEED in 42 43 44 45 46; do
  for MODEL in germanfinbert finbert; do
    python train_model.py --model $MODEL --augment --augmentation-version v2 \
        --seed $SEED --output-dir results/step4_comparison \
        --run-name ${MODEL}_seed${SEED}
  done
done

python compare_models.py           # aggregates + re-checks fairness preconditions
```

Both the split and the augmentation corpus are checksum-verified on load, so a silently
changed dataset aborts the run rather than producing an incomparable result. Runtime
≈ 2 minutes per run on an RTX 3060, ≈ 22 minutes total.

Not done in this step, by instruction: no temperature scaling, no calibration
metrics (ECE/NLL), no changes to the GermanFinBERT Step-3 or Step-3A results, no changes
to the frozen split or the augmentation corpus.
