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
| **Input drift** | PSI of the four text-feature distributions | approved model's validation split |
| **Prediction drift** | PSI of predicted-category and confidence distributions | approved model's validation predictions |
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
