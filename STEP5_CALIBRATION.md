# Step 5: Temperature Scaling and Confidence Calibration

> Part of the final Step 2-6 pipeline. Index: [`PROJECT_INDEX.md`](PROJECT_INDEX.md).

Temperature scaling fitted independently for each of the 10 Step-4 runs
(2 models × 5 seeds), on validation logits only, then applied unchanged to test.

```bash
python calibrate_models.py
```

**Headline: on this dataset temperature scaling does not improve calibration on the
test set, and on most seeds it makes it worse.** That is a negative result, reported as
measured. The cause is identified in §6 and it is a property of the calibration set,
not of the method.

The Step-4 uncalibrated results are unchanged and remain the reference.

---

## 1. Protocol

For each `(model, seed)` independently:

1. load the validation and test logits saved in Step 4 (`logits.npz`, no retraining);
2. fit a single scalar `T` by minimising **negative log-likelihood on the validation
   set** (LBFGS, strong-Wolfe line search, `log T` parameterised so `T > 0`);
3. apply that `T`, unchanged, to the test logits: `softmax(z / T)`;
4. recompute test metrics, calibration metrics and the 0.8 workflow.

Test labels were never used to fit or to select a temperature. No temperature was
chosen by comparing test performance. There is one `T` per run, fitted once.

This is only sound because Step 3 fixed `model_selection = "fixed_epochs"`: no
checkpoint or hyperparameter was ever selected on validation, so the validation split
arrives at Step 5 genuinely unused. Had we used early stopping, the same data would
have been consumed twice: the P4 mistake the audit found in the prototype.

Calibration set: 228 validation rows / 16 groups / 16 of 37 classes.

---

## 2. Fitted temperatures

| Model | T (mean ± sd) | Per seed (42 → 46) |
|---|---|---|
| germanfinbert | **1.1012 ± 0.0798** | 0.9863, 1.1115, 1.0628, 1.1872, 1.1584 |
| finbert | **1.0894 ± 0.1083** | 1.0961, 1.0318, 1.2689, 1.0641, 0.9861 |

Both sit just above 1, i.e. the fit judges both models mildly over-confident on
validation and softens their probabilities. Validation NLL (the quantity actually optimised) improved in 10 of 10 runs, so the optimiser did its job. The problem is
that the improvement does not transfer.

---

## 3. Test results, mean ± sd over 5 seeds: group level (n = 33)

| Model | Probabilities | Accuracy | Macro F1 | Top-2 | ECE | NLL | Brier | Mean conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| germanfinbert | uncalibrated | 0.6485 ± 0.0271 | 0.5442 ± 0.0408 | 0.7576 ± 0.0214 | **0.1754 ± 0.0366** | 1.4546 ± 0.0821 | 0.5064 ± 0.0211 | 0.7446 ± 0.0252 | **60.61 ± 5.67** | 0.8022 ± 0.0335 |
| germanfinbert | calibrated | 0.6485 ± 0.0271 | 0.5442 ± 0.0408 | 0.7576 ± 0.0214 | 0.2100 ± 0.0391 | **1.4488 ± 0.0698** | 0.5004 ± 0.0171 | 0.6883 ± 0.0653 | 44.24 ± 19.24 | 0.8634 ± 0.1049 |
| finbert | uncalibrated | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.6788 ± 0.0271 | **0.2138 ± 0.0321** | **1.9054 ± 0.0605** | 0.5715 ± 0.0156 | 0.4669 ± 0.0116 | 0.00 ± 0.00 | – |
| finbert | calibrated | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.6788 ± 0.0271 | 0.2497 ± 0.0451 | 1.9444 ± 0.0920 | 0.5949 ± 0.0356 | 0.4170 ± 0.0563 | 0.00 ± 0.00 | – |

### Row level (n = 647)

| Model | Probabilities | Accuracy | Macro F1 | Top-2 | ECE | NLL | Brier | Mean conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| germanfinbert | uncalibrated | 0.4513 ± 0.0313 | 0.5114 ± 0.0535 | 0.6294 ± 0.0268 | 0.2780 ± 0.0321 | 2.1015 ± 0.1398 | 0.7351 ± 0.0450 | 0.6210 ± 0.0418 | 39.81 ± 5.17 | 0.6470 ± 0.0760 |
| germanfinbert | calibrated | 0.4513 ± 0.0313 | 0.5114 ± 0.0535 | 0.6294 ± 0.0268 | 0.2782 ± 0.0302 | **2.0742 ± 0.1129** | 0.7195 ± 0.0356 | 0.5714 ± 0.0761 | 26.71 ± 15.20 | 0.7598 ± 0.1787 |
| finbert | uncalibrated | 0.4383 ± 0.0340 | 0.4763 ± 0.0321 | 0.5580 ± 0.0345 | 0.2429 ± 0.0811 | 2.4946 ± 0.1035 | 0.7401 ± 0.0088 | 0.3963 ± 0.0169 | 0.00 ± 0.00 | – |
| finbert | calibrated | 0.4383 ± 0.0340 | 0.4763 ± 0.0321 | 0.5580 ± 0.0345 | **0.2391 ± 0.0702** | 2.4976 ± 0.1017 | 0.7488 ± 0.0179 | 0.3545 ± 0.0488 | 0.00 ± 0.00 | – |

### Accuracy is identical before and after, as it must be

Accuracy, macro-F1, weighted-F1 and top-2 are bit-identical in every row above.
Scalar temperature scaling divides all logits by the same positive number, so it is
monotone and cannot reorder any prediction. `calibrate_models.py` asserts
`argmax(uncalibrated) == argmax(calibrated)` for all 10 runs; the assertion never fired.

This matters for reading the reference paper, which reports calibration *raising*
accuracy from 68.0 % to 73.4 %. A scalar temperature cannot do that. The paper's
`Calibrate(·)` is `z/T + b` with a learned per-class bias vector (§3.1.5). The bias is
what moves the argmax. That variant was examined earlier in this project and drove
calibration NLL to exactly 0.0000 by fitting 38 parameters to 16 independent points, so
it is not used here.

---

## 4. The 0.8 confidence workflow after calibration

| Model | ≥0.8 uncalibrated | ≥0.8 calibrated | Acc ≥0.8 uncal | Acc ≥0.8 cal |
|---|---|---|---|---|
| germanfinbert (group) | 60.61 ± 5.67 % | **44.24 ± 19.24 %** | 0.8022 ± 0.0335 | 0.8634 ± 0.1049 |
| finbert (group) | 0.00 % | **0.00 %** | – | – |

Two things to note.

FinBERT still automates nothing. Because its fitted `T > 1` on four of five seeds,
calibration *softens* its already-low confidence and moves it further from the
threshold. Its maximum calibrated confidence never approaches 0.8. The Step-4 finding
survives calibration.

GermanFinBERT's coverage drops and becomes unstable. Per-seed automation rate after
calibration: 63.6 %, 48.5 %, 57.6 %, 15.2 %, 36.4 %: a standard deviation of 19.2
points against 5.7 uncalibrated. Seed 45 collapses from 54.5 % to 15.2 % because its
fitted `T = 1.187` is the largest of the five. Accuracy among the retained predictions
rises (0.80 → 0.86), which is the expected trade (fewer, safer automatic decisions),
but the variance makes the operating point unpredictable, which is worse for a
deployment decision than a slightly miscalibrated but stable one.

---

## 5. Did calibration help? Per seed, on test

| Model | Seed | ΔECE | ΔNLL | Δ(≥0.8 %) |
|---|---|---|---|---|
| finbert | 42 | +0.0638 **worse** | +0.0377 worse | +0.0 |
| finbert | 43 | +0.0212 **worse** | +0.0140 worse | +0.0 |
| finbert | 44 | +0.1381 **worse** | +0.1183 worse | +0.0 |
| finbert | 45 | −0.0389 better | +0.0289 worse | +0.0 |
| finbert | 46 | −0.0048 better | −0.0036 better | +0.0 |
| germanfinbert | 42 | −0.0191 better | +0.0039 worse | +0.0 |
| germanfinbert | 43 | +0.0123 **worse** | −0.0262 better | −18.2 |
| germanfinbert | 44 | +0.0870 **worse** | −0.0108 better | −6.1 |
| germanfinbert | 45 | +0.0460 **worse** | −0.0090 better | −39.4 |
| germanfinbert | 46 | +0.0467 **worse** | +0.0136 worse | −18.2 |

Test ECE got worse on 7 of 10 runs; test NLL got worse on 6 of 10. Validation NLL
improved on 10 of 10. The fit is working and simply does not generalise from the
calibration split to the test split.

---

## 6. Why it fails: a diagnosable cause, not bad luck

The calibration set has 228 rows but only 16 independent groups, and group sizes are
very unequal. Fitting NLL over rows therefore weights a handful of large groups far more
than the 16 distinct patterns actually present.

Fitting the same temperature on one averaged observation per validation group instead of
per row gives a different answer. And in **8 of 10 runs the two fits disagree about the
direction of miscalibration**:

| Run | T fitted on rows | T fitted on groups | Agree? |
|---|---|---|---|
| finbert seed 42 | 1.0961 (soften) | 0.7714 (sharpen) | **disagree** |
| finbert seed 43 | 1.0318 (soften) | 0.7638 (sharpen) | **disagree** |
| finbert seed 44 | 1.2689 (soften) | 0.8277 (sharpen) | **disagree** |
| finbert seed 45 | 1.0641 (soften) | 0.7477 (sharpen) | **disagree** |
| finbert seed 46 | 0.9861 (sharpen) | 0.7563 (sharpen) | agree |
| germanfinbert seed 42 | 0.9863 (sharpen) | 0.8583 (sharpen) | agree |
| germanfinbert seed 43 | 1.1115 (soften) | 0.8930 (sharpen) | **disagree** |
| germanfinbert seed 44 | 1.0628 (soften) | 0.8944 (sharpen) | **disagree** |
| germanfinbert seed 45 | 1.1872 (soften) | 0.9488 (sharpen) | **disagree** |
| germanfinbert seed 46 | 1.1584 (soften) | 0.9341 (sharpen) | **disagree** |

Every group-level fit says *sharpen* (`T < 1`); most row-level fits say *soften*
(`T > 1`). A calibration set on which two defensible fitting choices disagree about
whether the model is over- or under-confident does not contain enough independent
information to calibrate anything. Add that it covers only 16 of 37 classes, and the
fitted temperature is essentially estimated from a handful of patterns and asked to
generalise to 33 unseen ones.

This is the same structural limitation that runs through the whole project (152 groups, median 2 per category) surfacing in the calibration stage.

---

## 7. What this means for the deliverable

* **Report the uncalibrated Step-4 numbers as the headline.** They are better calibrated
  on test (ECE 0.175 vs 0.210 for GermanFinBERT, 0.214 vs 0.250 for FinBERT) and their
  operating point is far more stable across seeds.
* **The calibration machinery is implemented, correct and reusable.** It fits on held-out
  data only, applies unchanged to test, and asserts its own invariants. On a corpus with
  a calibration set of realistic size it would be the right thing to run; here the data
  cannot support it.
* **Do not claim the paper's calibration result was reproduced.** The paper reports ECE
  0.1091 → 0.0048 and an accuracy gain. Neither is reproducible with a scalar temperature
  on a 16-group calibration set, and the accuracy gain is not achievable with a scalar
  temperature at all.
* **The model comparison is unaffected.** Calibration changes no prediction, so
  GermanFinBERT's advantage over FinBERT on accuracy, macro-F1 and top-2 is identical
  before and after, and FinBERT's 0 % automation rate survives calibration.

---

## 8. Saved artefacts

```
results/step5_calibration/
├── calibration_results.json                     everything, per run and aggregated
├── per_seed_calibration.csv                     40 rows: 2 models x 5 seeds x 2 levels x 2 variants
├── summary_calibration.csv                      mean ± sd tables
├── <model>_seed<N>_calibration.json             T, fit_info, validation and test metrics
├── <model>_seed<N>_calibrated_probs.npz         calibrated test probabilities + T
├── reliability_<model>_seed<N>_<level>.png      20 reliability diagrams, before vs after
└── calibrate.log
```

Step-4 artefacts were read only. Nothing in `results/step4_comparison/` was modified.

---

## 9. Limitations

1. **The calibration set is too small and unrepresentative**, 16 independent groups
   covering 16 of 37 classes. This is the direct cause of §5 and §6, and no choice of
   fitting procedure repairs it.
2. **Row-fitting versus group-fitting disagree on direction** in 8 of 10 runs. The
   default (`rows`) matches the deployment distribution (production sees rows), but the
   group fit is the honest effective sample size. Both are recorded.
3. **Test-set ECE is itself noisy.** With 33 groups and 15 bins most bins hold one or two
   observations, so ECE differences of ±0.03 should not be over-read.
4. **Only scalar temperature scaling was applied.** The paper's `z/T + b` vector variant
   was excluded on evidence of over-fitting (38 parameters, 16 independent points).
5. **Validation doubles as the calibration set.** A 4-way split was measured as
   unaffordable in Step 2 (only 13 of 37 categories have ≥ 4 groups). Because no model
   selection occurred, the split is used exactly once. But a genuinely separate
   calibration set was never available.
6. **Five seeds, one test set.** Seed variance is reported; test-set sampling uncertainty
   (±0.16 on accuracy at n = 33) is not, and dominates.

---

## 10. Reproducibility

```bash
python prepare_dataset.py                      # Step 2
python prepare_augmentation.py --version v2    # Step 3A
for SEED in 42 43 44 45 46; do
  for MODEL in germanfinbert finbert; do
    python train_model.py --model $MODEL --augment --augmentation-version v2 \
        --seed $SEED --output-dir results/step4_comparison \
        --run-name ${MODEL}_seed${SEED}
  done
done
python compare_models.py                       # Step 4, uncalibrated
python calibrate_models.py                     # Step 5, this document
```

Calibration itself needs no GPU and runs in a few seconds: it reads the saved logits
rather than re-running any model.
