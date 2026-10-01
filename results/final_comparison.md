# Final Evaluation: GermanFinBERT vs Standard FinBERT

Complete comparison from Steps 2–5. No model was retrained, no parameter tuned, and the
frozen test split was not modified. Every number below was **recomputed from saved
per-example logits** by `final_evaluation.py`; 120 of 120 recomputations matched the
values recorded in Steps 4 and 5.

```bash
python final_evaluation.py
```

---

## 1. Data integrity: 12 of 12 checks passed

| Check | Result |
|---|---|
| All 10 runs used one frozen split | **PASS** |
| That split matches the Step-2 manifest | **PASS**, `f715f040d8a5daa621164ae8…` |
| All 10 runs used one augmentation corpus | **PASS** |
| That corpus is v2 and matches the Step-3A manifest | **PASS**, `4eb242d9f384041fc5ec67ce…` |
| Corpus carries all three leakage gates | **PASS**, A provenance, B collision, C near-duplicate |
| All 10 runs used one identical label mapping | **PASS**, 37 classes |
| All 10 runs evaluated identical test groups, in identical order | **PASS** |
| All 10 runs used the same confidence threshold | **PASS**, 0.80 |
| Identical evaluation/training protocol across models | **PASS**, 12 hyperparameters compared |
| Each run used its declared seed | **PASS** |
| Temperature fitted on validation rows only, never on test | **PASS**, 228 vs 647 rows |
| Every recomputed metric matches its saved value | **PASS**, 120 comparisons |

| Experimental unit | Value |
|---|---|
| Train | 1 093 real rows + 4 541 gated synthetic = 5 634 |
| Validation / calibration | 228 rows, **16 groups**, 16 of 37 classes |
| **Test** | 647 rows, **33 groups**, 33 of 37 classes |
| Classes | 37 |
| Seeds | 42, 43, 44, 45, 46 |
| Threshold | 0.80 |

---

## 2. Final comparison table: group level (headline, n = 33 groups)

Mean ± standard deviation over 5 seeds.

| Model | Calibration | Accuracy | Macro F1 | Weighted F1 | Top-2 | ECE | NLL | ≥0.8 Coverage | ≥0.8 Accuracy | Manual Review Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **germanfinbert** | uncalibrated | **0.6485 ± 0.0271** | **0.5442 ± 0.0408** | **0.5772 ± 0.0433** | **0.7576 ± 0.0214** | **0.1754 ± 0.0366** | 1.4546 ± 0.0821 | **60.61 ± 5.67 %** | 0.8022 ± 0.0335 | 39.39 ± 5.67 % |
| **germanfinbert** | calibrated | 0.6485 ± 0.0271 | 0.5442 ± 0.0408 | 0.5772 ± 0.0433 | 0.7576 ± 0.0214 | 0.2100 ± 0.0391 | **1.4488 ± 0.0698** | 44.24 ± 19.24 % | **0.8634 ± 0.1049** | 55.76 ± 19.24 % |
| finbert | uncalibrated | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.5451 ± 0.0292 | 0.6788 ± 0.0271 | 0.2138 ± 0.0321 | 1.9054 ± 0.0605 | **0.00 %** | n/a | 100 % |
| finbert | calibrated | 0.6121 ± 0.0254 | 0.5054 ± 0.0304 | 0.5451 ± 0.0292 | 0.6788 ± 0.0271 | 0.2497 ± 0.0451 | 1.9444 ± 0.0920 | **0.00 %** | n/a | 100 % |

### Row level (n = 647 rows)

| Model | Calibration | Accuracy | Macro F1 | Weighted F1 | Top-2 | ECE | NLL | ≥0.8 Coverage | ≥0.8 Accuracy | Manual Review Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| germanfinbert | uncalibrated | 0.4513 ± 0.0313 | 0.5114 ± 0.0535 | 0.3813 ± 0.0383 | 0.6294 ± 0.0268 | 0.2780 ± 0.0321 | 2.1015 ± 0.1398 | 39.81 ± 5.17 % | 0.6470 ± 0.0760 | 60.19 ± 5.17 % |
| germanfinbert | calibrated | 0.4513 ± 0.0313 | 0.5114 ± 0.0535 | 0.3813 ± 0.0383 | 0.6294 ± 0.0268 | 0.2782 ± 0.0302 | 2.0742 ± 0.1129 | 26.71 ± 15.20 % | 0.7598 ± 0.1787 | 73.29 ± 15.20 % |
| finbert | uncalibrated | 0.4383 ± 0.0340 | 0.4763 ± 0.0321 | 0.3771 ± 0.0347 | 0.5580 ± 0.0345 | 0.2429 ± 0.0811 | 2.4946 ± 0.1035 | 0.00 % | n/a | 100 % |
| finbert | calibrated | 0.4383 ± 0.0340 | 0.4763 ± 0.0321 | 0.3771 ± 0.0347 | 0.5580 ± 0.0345 | 0.2391 ± 0.0702 | 2.4976 ± 0.1017 | 0.00 % | n/a | 100 % |

Group level is the honest sample size: 647 test rows collapse to 33 independent
patterns. Row level is reported for comparability with the reference paper.

**Accuracy, macro-F1, weighted-F1 and top-2 are bit-identical between the calibrated and
uncalibrated rows.** Scalar temperature scaling is monotone in the logits and cannot
reorder predictions; `calibrate_models.py` asserts this for all 10 runs.

---

## 3. Individual seed results: group level, uncalibrated

| Model | Seed | Accuracy | Macro F1 | Weighted F1 | Top-2 | Mean conf | Max conf | ≥0.8 % | Acc ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|
| germanfinbert | 42 | 0.6667 | 0.5638 | 0.5980 | 0.7576 | 0.7796 | 0.9313 | 63.6 | 0.8095 |
| germanfinbert | 43 | 0.6667 | 0.5857 | 0.6212 | 0.7576 | 0.7597 | 0.9254 | 66.7 | 0.7727 |
| germanfinbert | 44 | 0.6061 | 0.4857 | 0.5152 | 0.7576 | 0.7310 | 0.9217 | 63.6 | 0.7619 |
| germanfinbert | 45 | 0.6364 | 0.5190 | 0.5505 | 0.7273 | 0.7367 | 0.9219 | 54.5 | 0.8333 |
| germanfinbert | 46 | 0.6667 | 0.5667 | 0.6010 | 0.7879 | 0.7158 | 0.9285 | 54.5 | 0.8333 |
| finbert | 42 | 0.6061 | 0.5038 | 0.5343 | 0.6667 | 0.4581 | 0.7766 | 0.0 | – |
| finbert | 43 | 0.6364 | 0.5429 | 0.5758 | 0.6970 | 0.4514 | 0.7200 | 0.0 | – |
| finbert | 44 | 0.5758 | 0.4602 | 0.5020 | 0.6364 | 0.4726 | 0.7220 | 0.0 | – |
| finbert | 45 | 0.6061 | 0.5000 | 0.5455 | 0.6970 | 0.4789 | 0.6938 | 0.0 | – |
| finbert | 46 | 0.6364 | 0.5204 | 0.5677 | 0.6970 | 0.4734 | 0.7295 | 0.0 | – |

Per-seed calibrated results and the row-level equivalents are in `final_results.csv`
(40 rows: 2 models × 5 seeds × 2 calibration states × 2 levels). No seed was selected
on test performance; all five are reported.

---

## 4. The 0.8 confidence workflow

The threshold was held at 0.80 throughout and was not lowered to flatter FinBERT.

| Model | Calibration | Automatic | Manual review | Accuracy when automatic |
|---|---|---|---|---|
| germanfinbert | uncalibrated | **60.61 ± 5.67 %** | 39.39 % | 0.8022 ± 0.0335 |
| germanfinbert | calibrated | 44.24 ± 19.24 % | 55.76 % | 0.8634 ± 0.1049 |
| finbert | uncalibrated | **0.00 %** | **100 %** | undefined |
| finbert | calibrated | **0.00 %** | **100 %** | undefined |

FinBERT's maximum confidence over all five seeds is 0.7766 uncalibrated, and lower
after calibration (its fitted `T > 1` on four of five seeds softens it further). It never
reaches the threshold on any seed in either state, so it automates nothing.

Approximately 4 of every 5 automatic decisions by GermanFinBERT are correct, meaning
roughly one in five automated transactions is silently misfiled. That is the number a
deployment decision has to be made against, not the headline accuracy.

---

## 5. Calibration analysis: fitting performance vs test performance

This distinction matters and the two point in opposite directions.

| Quantity | Direction |
|---|---|
| **Validation NLL** (the fitted objective) | improved in **10 of 10 runs** |
| **Test NLL** | worsened in **6 of 10 runs** |
| **Test ECE** | worsened in **7 of 10 runs** |
| **Confidence stability** | worsened, GermanFinBERT ≥0.8 coverage sd rose 5.67 → 19.24 points |
| **≥0.8 coverage** | fell, 60.61 % → 44.24 % (GermanFinBERT); unchanged at 0 % (FinBERT) |
| **≥0.8 accuracy** | rose, 0.8022 → 0.8634, but on far fewer and less stable predictions |

Fitted temperatures: GermanFinBERT `T = 1.1012 ± 0.0798`, FinBERT `T = 1.0894 ± 0.1083`.

**Reported honestly as a generalisation limitation: temperature scaling optimised its
objective on the calibration set and failed to transfer to the test set.**

The cause is diagnosable rather than incidental. The calibration set holds 228 rows but
only 16 independent groups covering 16 of 37 classes, and group sizes are very
unequal. Fitting the same temperature on one averaged observation per validation group
instead of per row gives a different answer, and in **8 of 10 runs the two fits disagree
about the direction of miscalibration**: every group-level fit says sharpen (`T < 1`),
most row-level fits say soften (`T > 1`). A calibration set on which two defensible
fitting choices disagree about whether the model is over- or under-confident does not
carry enough independent information to calibrate anything.

Consequence for the deliverable: the uncalibrated results are the ones to report.
They have lower test ECE for both models and a far more stable operating point.

---

## 6. Error analysis

Across 33 test groups and 5 seeds, using each model's modal prediction:

| | Count |
|---|---|
| Both models always correct (5/5 seeds) | **17** |
| Both models always wrong (0/5 seeds) | **8** |
| GermanFinBERT better | 5 groups |
| FinBERT better | 3 groups |
| Tied | 25 groups |

### The eight groups both models always get wrong

| Text | True category | GermanFinBERT predicts | Train rows for true class |
|---|---|---|---|
| `ausschüttung gesellschafter` | Dividenden / Ausschüttungen | Gesellschaftereinlagen | 7 |
| `bonuszahlung mitarbeitende` | Boni / Prämien | Boni / Provisionen | 3 |
| `cloud hosting` | IT & Software | Abonnements | 135 |
| `refund vendor credit` | Rückvergütungen | Tilgungen | 18 |
| `gehaltslauf` | Löhne & Gehälter | Boni / Provisionen | 78 |
| `software license revenue` | Lizenzen | IT & Software | 34 |
| `lohnabgaben` | Sozialabgaben | Rohstoffe | 47 |
| `versicherungsprämie` | Versicherungen | Versicherungsleistungen | 27 |

Category confusion. Direction of money is the dominant failure mode.
86 % (12 of 14) of GermanFinBERT's errors cross a parent-category boundary, and most
are near-synonym pairs separated only by whether money flows in or out:

* `Versicherungen` (premium paid) ↔ `Versicherungsleistungen` (claim received)
* `Dividenden / Ausschüttungen` (paid out) ↔ `Gesellschaftereinlagen` (paid in)
* `Boni / Prämien` (staff bonus, expense) ↔ `Boni / Provisionen` (commission, income)
* `software license revenue`: licence revenue, classified as the licence expense
  class `IT & Software`

The `amount` column resolves the income/expense distinction by its sign and sits unused
in the source workbook, excluded by the agreed text-only scope. It is the single
cheapest available improvement.

Ambiguous descriptions. `cloud hosting` → `Abonnements` and
`serviceeinsatz technik` → `Wartungsverträge` are arguably defensible readings; a
service call and a maintenance contract are genuinely hard to separate from three words.
`rechnung produktverkauf bestellung` mixes sale and order vocabulary in one string.

Rare categories. `Boni / Prämien` has 3 training rows and is never recovered.
`Dividenden / Ausschüttungen` has 7. But scarcity is not the whole story: `cloud hosting`
fails with 135 training rows for its class, and `gehaltslauf` with 78. Those are
semantic confusions, not data starvation.

German-language effects. English FinBERT fragments German text ≈23 % more heavily
(3.13 vs 2.54 subtokens per word; neither model emits `[UNK]`). The clearest
German-specific wins for GermanFinBERT are `office supplies` (5/5 vs 1/5, FinBERT
predicts `Beratung & Dienstleistungen`) and `firmenwagen leasing anzahlung` (5/5 vs 1/5,
FinBERT predicts `Kapitalerhöhungen`, a compound it evidently cannot decompose).
FinBERT wins on `rechnung produktverkauf bestellung` (5/5 vs 2/5) and
`hardware einkauf maschine a12` (5/5 vs 4/5), both containing English or alphanumeric
tokens.

Grouping limitations. Every test category holds exactly one group, so per-class
recall can only be 0.0 or 1.0 and macro-F1 has high variance. Four categories
(`Darlehensaufnahmen`, `Gesellschaftereinlagen`, `Kapitalerhöhungen`,
`Versicherungsleistungen`) cannot appear in test at all. Yet two of them are
*predicted* as confusions, so they contribute false positives while being untestable
themselves. Lexical grouping also cannot merge `gehaltslauf` with
`payroll sammelüberweisung`, which mean the same thing but share no tokens.

Model-specific errors. The models disagree on only 8 of 33 groups, and neither
dominates the disagreements (5 vs 3). Their errors are largely the *same* errors.

---

## 7. Conclusions

Under the experimental conditions and test set used in this proof of concept:

1. Which model performed better? GermanFinBERT, on every aggregate metric: group
accuracy 0.6485 ± 0.0271 vs 0.6121 ± 0.0254, macro-F1 0.5442 vs 0.5054, top-2 0.7576 vs
0.6788.

2. How consistent was the difference? The sign was consistent on 5 of 5 seeds for
accuracy, macro-F1 and top-2 (paired t = 6.00, p = 0.0039 on accuracy). That establishes
robustness to training randomness only: all ten runs share one 33-group test set, so the
paired test treats the test set as fixed and models none of the sampling uncertainty that
actually dominates.

3. Did GermanFinBERT outperform on accuracy/F1/top-2? Yes on this test set. The
accuracy margin is +0.0364, 1.2 of 33 test groups. The top-2 margin (+0.0788) is
larger and more robust.

4. Which model provides useful ≥0.8 coverage? Only GermanFinBERT. It automates
60.6 ± 5.7 % of test groups at 80.2 % accuracy. FinBERT automates 0 % in every seed
and both calibration states; its maximum confidence anywhere is 0.7766. This is the
single largest and most decision-relevant difference, and it is invisible to an
accuracy-only comparison.

5. Did temperature scaling improve calibration on unseen data? No. Validation NLL
improved in 10/10 runs; test ECE worsened in 7/10 and test NLL in 6/10. It also made the
operating point unstable (coverage sd 5.67 → 19.24 points). Report the uncalibrated
results.

6. Does the group size limit confidence in generalisation? Severely. One test
group is worth 3.03 accuracy points; the model gap is 1.2 groups. A 95 % confidence
interval on a single group-level accuracy of 0.65 at n = 33 spans roughly ±0.16,
more than four times the observed gap. The calibration set is smaller still: 16 groups
covering 16 of 37 classes.

7. What can and cannot be concluded?

*Can:* under identical data, configuration and evaluation, GermanFinBERT outperformed
standard FinBERT on this test set on every aggregate metric and every seed; the gap in
usable high-confidence coverage (60.6 % vs 0 %) is categorical rather than marginal; LLM
augmentation is the dominant driver of performance (+27 accuracy points over no
augmentation, `STEP3_GERMANFINBERT.md` §6); and the pipeline is leakage-free by
construction and verified.

*Cannot:* that GermanFinBERT universally outperforms FinBERT; that the 3.6-point accuracy
gap generalises beyond this test set; that any per-class result is reliable (each test
class is one item); that temperature scaling is useless in general. Only that it fails
on a 16-group calibration set; or that these results transfer to real German SME bank
data, since the corpus is synthetic and template-generated.

---

## 8. Comparison with the reference paper

These are different experiments. No claim of reproduction is made.

| | Paper (Aluffi et al., 2025) | This project |
|---|---|---|
| Language | English | **German** |
| Categories | 11 | **37** |
| Data | real Open-Banking transactions, 9 SMEs | **synthetic, template-generated** |
| Independent test items | thousands of rows | **33 groups** |
| Split unit | held-out company (SME 3) | **held-out text group** |
| Primary model | FinBERT | **GermanFinBERT**; FinBERT as comparison |
| Training set | balanced synthetic only | real + gated synthetic |
| Calibration | `z/T + b`, per-class bias | scalar `T` only |
| **Standard accuracy** | 73.49 % ± 5.09 | **64.85 % ± 2.71** (group), 45.13 % (row) |
| **High-confidence accuracy (>0.8)** | 90.36 % ± 6.52 | **80.22 % ± 3.35** |
| **Top-2 accuracy** | 89.63 % ± 4.51 | **75.76 % ± 2.14** |
| **ECE after calibration** | 0.0048 (from 0.1091) | **0.2100** (from 0.1754. Worsened) |
| Calibration effect on accuracy | +5.4 points | **0.0 by construction** |

Two differences deserve explicit comment.

The paper's calibration raises accuracy; ours cannot. A scalar temperature is monotone
and cannot reorder predictions. The paper's `Calibrate(·)` is `z/T + b` with a learned
per-class bias vector (§3.1.5); the bias is what moves the argmax. That variant was tested
here and drove calibration NLL to exactly 0.0000 by fitting 38 parameters to 16
independent points, so it was excluded as over-fitting.

The paper's ECE improvement is not reachable at this calibration-set size. 0.1091 →
0.0048 on a calibration set drawn from two full companies is a different regime from 16
groups covering 16 of 37 classes.

The pipeline follows the paper's *method* (Clean, Group, Generate, Finetune with weighted focal loss, Calibrate) and the shortfall against its numbers is explained by the data,
not by a departure from the methodology.

---

## 9. Reproducibility and traceability

Every number in this document is regenerated by:

```bash
python prepare_dataset.py                      # Step 2  - freeze split
python verify_dataset.py                       # 19 checks
python prepare_augmentation.py --version v2    # Step 3A - freeze corpus
python verify_augmentation.py                  # 11 checks
for SEED in 42 43 44 45 46; do
  for MODEL in germanfinbert finbert; do
    python train_model.py --model $MODEL --augment --augmentation-version v2 \
        --seed $SEED --output-dir results/step4_comparison \
        --run-name ${MODEL}_seed${SEED}
  done
done
python compare_models.py                       # Step 4  - uncalibrated
python calibrate_models.py                     # Step 5  - temperature scaling
python final_evaluation.py                     # Step 6  - this document
python tests/test_pipeline.py                  # 66 checks
```

| Artefact | Contents |
|---|---|
| `results/final_results.csv` | 40 rows, every (model, seed, calibration, level) |
| `results/final_results.json` | the above plus integrity, traceability and error analysis |
| `results/final_summary.csv` | mean ± sd tables |
| `results/final_summary.xlsx` | Summary / PerSeed / Integrity / Errors / Provenance sheets |
| `results/final_error_analysis.csv` | per-group correctness for both models across all seeds |
| `results/step4_comparison/*/logits.npz` | raw logits, the ultimate source of every metric |

Traceability was verified rather than assumed: all 120 metric recomputations from raw
logits matched the values recorded in Steps 4 and 5.
