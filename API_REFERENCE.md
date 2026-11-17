# API reference

Implemented in `src/api.py` (API version `1.0.0`, release `v1.1.0-mlops`). Machine-readable
schema: [`docs/openapi.json`](docs/openapi.json) (exported from the running service); interactive
docs at `/docs` (Swagger UI) and `/redoc`. All examples below are **real responses** captured from
the service serving `germanfinbert-seed42-40de860119`, with synthetic inputs
(`examples/demo/demo_output_example.txt`). Earlier, longer transcript:
`docs/evidence/03_api_curl_examples.txt`.

Base URL: `http://127.0.0.1:8000` (local or Docker default). Content type: `application/json`.

## Authentication

**None on the prediction, feedback and monitoring endpoints.** Only `POST /model/reload` is
protected by the `X-Admin-Token` header and is disabled unless `TX_ADMIN_TOKEN` is set. Put the API
behind a gateway that provides TLS, authentication and rate limiting before exposing it
([SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md)).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model loaded and checksum verified |
| GET | `/model/info` | served version, checkpoint, categories, threshold, limits, metrics |
| POST | `/predict` | classify one description |
| POST | `/predict/batch` | classify up to 256 descriptions |
| POST | `/feedback` | confirmed category for an earlier prediction (delayed ground truth) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/monitoring/summary` | monitoring snapshot (decisions, drift, observed accuracy) |
| POST | `/model/reload` | reload the approved model (admin) |

## Error format

Every non-2xx response: `{"error": "<code>", "detail": <string or list>, "request_id": "<id>"}`.
Submitted descriptions are never echoed. Codes: `validation_error` (422), `unknown_prediction`
(404), `not_found` (404), `forbidden` (403), `model_unavailable` (503), `internal_error` (500).
Send `X-Request-ID` to set your own request id.

---

### GET /health

Response `HealthResponse`: `status` (`ok` | `unavailable`), `model_loaded` (bool),
`model_version`, `api_version`, `detail` (reason when unavailable). HTTP 200 when a model is
served, 503 otherwise.

```text
$ curl -s http://127.0.0.1:8000/health
{"status": "ok", "model_loaded": true, "model_version": "germanfinbert-seed42-40de860119", "api_version": "1.0.0", "detail": null}
```

### GET /model/info

Returns `model_version`, `model_name`, `base_checkpoint`, `base_revision`, `n_classes`,
`categories`, `confidence_threshold`, `calibrated`, `confidence_note`, `max_length`, `seed`,
`training_device`, `created_at`, `git_commit`, `data` (checksums), `metrics` (validation/test),
`limits` (`max_text_length`, `max_batch_size`), `status`. Excerpt:

```json
{"model_version": "germanfinbert-seed42-40de860119", "base_checkpoint": "scherrmann/GermanFinBert_FP",
 "n_classes": 37, "confidence_threshold": 0.8, "calibrated": false,
 "limits": {"max_text_length": 512, "max_batch_size": 256}}
```

### POST /predict

Request `PredictRequest` (unknown fields rejected):

| Field | Type | Rules |
|---|---|---|
| `description` | string | required, 1–512 characters (`api.max_text_length`), not whitespace-only |
| `transaction_id` | string or null | optional, ≤ 128 characters, echoed back |

Response `PredictionResult`:

| Field | Type | Meaning |
|---|---|---|
| `prediction_id` | string | random id; use with `/feedback` |
| `transaction_id` | string or null | echoed |
| `category`, `parent_category` | string | predicted category (37 classes) and its parent (11) |
| `confidence` | float | uncalibrated softmax probability of `category` (routing signal, not a guarantee) |
| `decision` | string | `automatic` if `confidence ≥ threshold`, else `manual_review` |
| `threshold` | float | 0.80 |
| `alternatives` | list of `{category, probability}` | next most probable categories (2) |
| `calibrated` | bool | `false` |
| `model_version` | string | served version |

```text
$ curl -s -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
    -d '{"description": "Betriebshaftpflicht 2026-07", "transaction_id": "demo-single"}'
{"prediction_id": "35bf6f7a92244f80b163753c3aad1a56", "transaction_id": "demo-single",
 "category": "Versicherungen", "parent_category": "Betriebliche Aufwendungen", "confidence": 0.92332,
 "decision": "automatic", "threshold": 0.8,
 "alternatives": [{"category": "Versicherungsleistungen", "probability": 0.00452},
                  {"category": "Miete & Nebenkosten", "probability": 0.00341}],
 "calibrated": false, "model_version": "germanfinbert-seed42-40de860119"}
```

Errors (captured):

```text
{"description": ""}                -> 422 {"error":"validation_error","detail":[{"loc":["body","description"],"msg":"String should have at least 1 character","type":"string_too_short"}],"request_id":"..."}
{"description":"x","amount":5}     -> 422 {"error":"validation_error","detail":[{"loc":["body","amount"],"msg":"Extra inputs are not permitted","type":"extra_forbidden"}],...}
513 characters                     -> 422 {"error":"validation_error","detail":"description longer than 512 characters",...}
malformed JSON                     -> 422 {"error":"validation_error","detail":[{"loc":["body",16],"msg":"JSON decode error","type":"json_invalid"}],...}
no model loaded                    -> 503 {"error":"model_unavailable","detail":"<reason>",...}
```

### POST /predict/batch

Request `BatchRequest`: `{"items": [{"description": ..., "transaction_id": ...}, ...]}`,
1–256 items (`api.max_batch_size`). Each item is validated separately.

Response `BatchResponse`: `model_version`, `n_items`, `n_ok`, `n_errors`, `n_automatic`,
`n_manual_review`, `results` (list of `{index, status: "ok"|"error", transaction_id, result:
PredictionResult|null, error: string|null}`).

Captured with `examples/demo/synthetic_batch.json` (summary of the response):

```text
items 6  ok 5  errors 1  automatic 0  manual_review 5
  demo-001                 Miete & Nebenkosten                    0.779  manual_review
  demo-002                 Boni / Provisionen                     0.188  manual_review
  demo-003                 Versicherungen                         0.604  manual_review
  demo-004                 Abonnements                            0.306  manual_review
  demo-005                 Boni / Provisionen                     0.384  manual_review
  demo-006-invalid-empty   ERROR: description is empty or whitespace
```

257 items → 422 `{"error":"validation_error","detail":"batch larger than 256 items",...}`.

### POST /feedback

Request `FeedbackRequest`: `prediction_id` (1–64 chars), `true_category` (a category of the
taxonomy, 1–200 chars). Response: `{"accepted": true, "model_version": "..."}`.

```text
$ curl -s -X POST http://127.0.0.1:8000/feedback -H 'content-type: application/json' \
    -d '{"prediction_id": "35bf6f7a92244f80b163753c3aad1a56", "true_category": "Versicherungen"}'
{"accepted": true, "model_version": "germanfinbert-seed42-40de860119"}
```

Errors: unknown id → 404 `{"error":"unknown_prediction","detail":"prediction_id not found in the
monitoring window",...}`; unknown category → 422 `{"error":"validation_error","detail":"true_category
is not in the label taxonomy",...}`. The window holds the last 5,000 predictions of the running process.

### GET /metrics

Prometheus text format. Families: `txclf_requests_total{endpoint,status}`,
`txclf_request_latency_seconds`, `txclf_predictions_total{decision}`,
`txclf_predicted_category_total`, `txclf_confidence`, `txclf_errors_total`,
`txclf_feedback_labels_total`, `txclf_drift_psi`, `txclf_model_info`.

```text
txclf_requests_total{endpoint="/predict",status="200"} 2.0
txclf_requests_total{endpoint="/predict",status="422"} 1.0
txclf_predictions_total{decision="automatic"} 2.0
```

### GET /monitoring/summary

Keys: `model_version`, `uptime_seconds`, `predictions_in_window`, `totals`, `decisions`
(`automatic`, `manual_review`, `manual_review_rate`, `reference_auto_rate`, `threshold`),
`confidence`, `category_distribution_top10`, `input_drift` (PSI per feature with status),
`prediction_drift`, `observed_performance` (from feedback; ≥ 30 labels), `latency_ms`,
`data_quality`, `thresholds`, `privacy`.

```json
{"model_version": "germanfinbert-seed42-40de860119", "predictions_in_window": 12,
 "decisions": {"threshold": 0.8, "automatic": 2, "manual_review": 10, "manual_review_rate": 0.8333,
               "reference_auto_rate": 0.384}}
```

### POST /model/reload

Header `X-Admin-Token: <TX_ADMIN_TOKEN>`. Reloads the approved registry version (after a
promotion or rollback). Disabled → 404 `{"error":"not_found","detail":"reload endpoint disabled"}`;
wrong or missing token → 403 `{"error":"forbidden","detail":"invalid admin token"}`.
