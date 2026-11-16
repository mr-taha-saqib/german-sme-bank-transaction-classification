# Model card: German SME transaction classifier

| | |
|---|---|
| Served version | `germanfinbert-seed42-40de860119` (weights SHA-256 prefix `40de860119`) |
| Architecture | `BertForSequenceClassification`, 110 M parameters, 37-class head |
| Base checkpoint | `scherrmann/GermanFinBert_FP`, Hugging Face revision `0b3622dd589d2fc2f518b4be12a9dc306b903df2` |
| Training | research configuration unchanged: corpus v2 (1,093 real + 4,541 synthetic rows), lr 2e-5, batch 16, max_length 128, 3 fixed epochs, class-weighted focal loss γ = 2, AdamW, seed 42 |
| Hardware | CPU (2 cores, fp32, torch 2.13.0+cpu), 6,818 s of training (≈ 1 h 54 min). The published runs used an RTX 3060 with mixed precision |
| Calibration | none: uncalibrated softmax (temperature scaling did not generalise in the research) |
| Decision rule | confidence ≥ 0.80 → `automatic`, else `manual_review` (research operating point, unchanged) |
| MLflow | experiment `german-sme-transaction-classification`, model version 1, alias `champion`; run `6e5f29aafc344bc5bf2386b2cb2c0de2` |

## Intended use

Pre-classifying short German B2B bank-transaction descriptions into the 37-category
bookkeeping taxonomy, with a human reviewing everything below 0.80 confidence. A
reference implementation for evaluation with the client's own data; **not** an
autonomous accounting system.

## Performance

Frozen test split (33 independent groups, 647 rows), evaluated once by `train_model.py`
after fixed-epoch training. The published GPU run with the same seed is shown for
comparison; the published five-seed mean is the authoritative research result.

| Metric (test) | This CPU model | Published GPU seed 42 | Published mean ± sd (5 seeds) |
|---|---|---|---|
| Group accuracy | 0.6970 (23/33) | 0.6667 (22/33) | 0.6485 ± 0.0271 |
| Group macro-F1 | 0.5880 | 0.5638 | 0.5442 ± 0.0408 |
| Group top-2 | 0.7576 | 0.7576 | 0.7576 ± 0.0214 |
| Group automatic share (≥ 0.80) | 60.6 % | 63.6 % | 60.61 ± 5.67 % |
| Group accuracy among automatic | 0.800 | 0.810 | 0.8022 ± 0.0335 |
| Row accuracy | 0.5317 | 0.4946 | – |
| Row automatic share / accuracy | 41.9 % / 0.738 | 42.8 % / 0.744 | – |

Validation split metrics are identical to the GPU run to five decimals (row accuracy
0.49123, macro-F1 0.68992). On test, 557 of 647 row predictions (86 %) match the GPU
run; the remainder differ because fp32 CPU and mixed-precision GPU arithmetic diverge
during fine-tuning. The one-group difference in accuracy is within the research's
seed-to-seed spread and must not be read as an improvement.

## Limitations

* Trained on **synthetic**, template-generated data (2,000 rows → 184 distinct texts).
  Accuracy on real bank descriptions is unknown and likely lower.
* 33 test groups: one group = 3.03 accuracy points; 95 % CI ≈ ±0.16.
* About one in five automatic decisions was wrong on the test set. Confidence is not a
  guarantee of correctness.
* Lower-cased input; text only (amount, date and counterparty metadata unused).
* Example of failure: "Gehaltslauf 2026-01" (payroll) is predicted as
  "Boni / Provisionen" with confidence 0.19 and therefore routed to manual review.

## Ethical and privacy considerations

The API never logs or stores descriptions; monitoring keeps numeric features only.
Automatic decisions should be sampled and audited; delayed labels (`POST /feedback`)
make that measurable.

## Research context

Comparison model FinBERT (`ProsusAI/finbert`) never reached 0.80 confidence and
automates nothing; see `FINAL_REPORT.md`. All research numbers are unchanged.
