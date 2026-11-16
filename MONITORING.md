# Monitoring

Implemented in `src/monitoring.py`; exposed by the API (`GET /metrics`,
`GET /monitoring/summary`); evidence in `docs/evidence/`.

## Privacy first

Descriptions are processed in memory only. The monitor records per prediction:
a random `prediction_id`, the predicted category, confidence, decision and four
**bucketed numeric features** (character length, token count, share of tokens unseen
in training, share of digits). No text, no hash of text. Validation errors never echo
the input. Request logs contain method, endpoint, status, latency and request id only.

## Signals

| Signal | Source | Reference |
|---|---|---|
| Requests, latency, errors | `/metrics` (`txclf_requests_total`, `txclf_request_latency_seconds`, `txclf_errors_total`) | – |
| Prediction count by decision, by category; confidence histogram | `/metrics`, summary | – |
| Manual-review rate | summary `decisions` | validation auto-rate of the approved model |
| **Input drift** | PSI of the four text-feature distributions | held-out real descriptions (validation + test, 875 rows) |
| **Prediction drift** | PSI of predicted-category and confidence distributions | this model's predictions on the held-out rows (predictions vs predictions) |
| **Observed performance** | accuracy (overall, among automatic) from `POST /feedback` labels | shown only with ≥ 30 labels |
| Data quality | empty-after-cleaning inputs, validation errors, batch item errors | – |

PSI status: `ok` < 0.10 ≤ `warning` < 0.25 ≤ `alert`; `insufficient_data` below 50
predictions (configurable in `configs/mlops.yaml`). Drift is never reported as a
performance change; accuracy is reported only from delayed ground truth.

## Delayed ground truth

Each prediction returns `prediction_id`. When a person confirms the category
(e.g. after manual review), send `POST /feedback {"prediction_id", "true_category"}`.
Labels for automatic decisions are needed too, otherwise the accuracy estimate is
biased towards hard cases; the summary reports both counts.

## Event log

`monitoring/events.jsonl` (prediction and label events, numeric fields only).
Metrics are in process memory and reset on restart; scrape `/metrics` for history.

## Reference choice (and a correction made during acceptance)

New transactions are patterns the model has not trained on, so the reference is the
held-out population (validation + test rows). Two earlier choices produced false
alarms on realistic traffic and were replaced: the validation split alone (16 of 37
classes) and true-label category frequencies (prediction drift must compare
predictions with predictions; model errors are not drift). The reference is a
monitoring baseline only, never used for model selection.

## Controlled test (executed, `docs/evidence/04_monitoring_drift_test.log`)

| Phase (fresh API process) | Input drift (4 signals) | Category / confidence drift | Manual review | Observed performance |
|---|---|---|---|---|
| 200 real held-out descriptions | all `ok` (PSI 0.007–0.018) | 0.156 `warning` / 0.013 `ok` | 64.5 % | insufficient labels |
| + 100 more with delayed labels | all `ok` (≤ 0.011) | 0.055 `ok` / 0.014 `ok` | 62.7 % | **available**: accuracy 0.54, among automatic 0.83 |
| 200 English free-text descriptions | all `alert` (PSI 6.8–17.1) | 7.51 `alert` / 4.25 `alert` | 98 % | insufficient labels |

PSI over 37 categories is noisy for a few hundred requests (the single `warning` at 200
requests cleared at 300); raise `min_requests_for_drift` for production alerting.
Privacy check over the 600 logged events: 0 fields outside the allow-list, no
description words present.
