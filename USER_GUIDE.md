# User guide

How to use the transaction classifier day to day. All examples use **synthetic** descriptions.
Endpoint details: [API_REFERENCE.md](API_REFERENCE.md). Setup: [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md).

## 1. Typical workflow

1. Export new bank-transaction descriptions from your banking or bookkeeping system.
2. Send them to `POST /predict/batch` (up to 256 per call) or `POST /predict`.
3. Book results with `decision = "automatic"` directly (subject to your own spot checks); put
   `manual_review` results in a review queue for a bookkeeper.
4. When a bookkeeper confirms or corrects a category, send it to `POST /feedback` with the
   `prediction_id`; this measures real accuracy over time.
5. Watch `GET /monitoring/summary` (or scrape `GET /metrics`) for drift and review rates.
6. Periodically collect confirmed labels into a CSV and run gated retraining
   (`python -m src.retrain`); promote only after reviewing the report.

## 2. Input requirements

| Field | Rule |
|---|---|
| `description` | German booking text, 1–512 characters, not only whitespace. Dates, reference numbers and IBAN-like tokens are removed server-side, so they may be included |
| `transaction_id` | optional, ≤ 128 characters, echoed back for matching |
| batch | `{"items": [...]}`, 1–256 items; an invalid item is reported, the rest are classified |

No other fields are accepted (unknown fields → HTTP 422). Do not send names of private persons or
account numbers unless your data-protection basis allows it; the service does not store
descriptions, but your network path and gateway may log requests.

## 3. Expected output

```json
{
  "prediction_id": "35bf6f7a92244f80b163753c3aad1a56",
  "transaction_id": "demo-single",
  "category": "Versicherungen",
  "parent_category": "Betriebliche Aufwendungen",
  "confidence": 0.92332,
  "decision": "automatic",
  "threshold": 0.8,
  "alternatives": [{"category": "Versicherungsleistungen", "probability": 0.00452},
                   {"category": "Miete & Nebenkosten", "probability": 0.00341}],
  "calibrated": false,
  "model_version": "germanfinbert-seed42-40de860119"
}
```

`confidence` is an uncalibrated softmax probability. It ranks how sure the model is; it is
**not** the probability of being correct. About one in five automatic decisions was wrong on the
research test set.

## 4. Practical examples

```bash
# one transaction
curl -s -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
     -d '{"description": "Miete Buero Juli 2026", "transaction_id": "tx-42"}'

# a batch from a file
curl -s -X POST http://127.0.0.1:8000/predict/batch -H 'content-type: application/json' \
     --data @examples/demo/synthetic_batch.json

# confirm the category of an earlier prediction
curl -s -X POST http://127.0.0.1:8000/feedback -H 'content-type: application/json' \
     -d '{"prediction_id": "<id from a prediction>", "true_category": "Miete & Nebenkosten"}'

# everything at once, with readable output
./examples/demo/run_demo.sh
```

A captured run against the real model is in `examples/demo/demo_output_example.txt`. In that run
every batch item went to manual review, and one (`Gehaltslauf 2026-07`) was misclassified with
low confidence (0.188): this is the intended safety behaviour of the 0.80 threshold, and a reminder
that the model needs validation on your data.

Python client:

```python
import httpx
r = httpx.post("http://127.0.0.1:8000/predict", json={"description": "Telekom Rechnung Juli"})
r.raise_for_status()
p = r.json()
queue = "auto" if p["decision"] == "automatic" else "review"
```

## 5. Error handling

| HTTP | `error` | Typical cause | What to do |
|---|---|---|---|
| 422 | `validation_error` | empty / too long description, unknown field, malformed JSON, batch > 256, unknown category in feedback | fix the request; the message names the field |
| 404 | `unknown_prediction` | feedback for a `prediction_id` that is no longer in the monitoring window (5,000 predictions) or after a restart | send feedback sooner |
| 404 | `not_found` | `/model/reload` while `TX_ADMIN_TOKEN` is unset | set the token to enable it |
| 403 | `forbidden` | wrong `X-Admin-Token` | |
| 503 | `model_unavailable` | no approved model, or checksum mismatch | see `/health` `detail` |
| 500 | `internal_error` | unexpected | logged with the `request_id` |

Every error body contains `request_id`; quote it when reporting a problem.

## 6. Model management

The registry is `model_registry/` (`registry.json` + `versions/<id>/`). Inspect it:

```bash
python -c "from src.registry import ModelRegistry as R; r=R('model_registry'); print('approved:', r.approved_version()); print(r.list_versions())"
curl -s http://127.0.0.1:8000/model/info | python -m json.tool     # what the API serves
```

* Register (and optionally approve) a trained run: `python -m src.model_artifact --run <run dir> [--approve --reason "..."]`.
* Roll back to the previous approved version: `python -m src.retrain --rollback --reason "..."`.
* Make a running API serve the new approved version: restart it, or set `TX_ADMIN_TOKEN` and call
  `curl -X POST -H "X-Admin-Token: $TX_ADMIN_TOKEN" http://127.0.0.1:8000/model/reload`.

Every approval, rejection and rollback is recorded with a reason in `registry.json`; MLflow's
`champion` alias follows the approved version.

## 7. Monitoring

* `GET /monitoring/summary`: decision counts and manual-review rate (compared with the model's
  validation rate), confidence statistics, input drift and prediction drift (PSI with
  `ok` / `warning` / `alert` / `insufficient_data`), observed accuracy from feedback (shown from
  30 labels), latency, data-quality counters.
* `GET /metrics`: Prometheus format (`txclf_requests_total`, `txclf_predictions_total`,
  `txclf_confidence`, `txclf_request_latency_seconds`, ...).
* Simulate traffic to see the signals: `python scripts/simulate_traffic.py --mode normal --n 200`
  then `--mode drift` (synthetic).

A drift alert means the inputs look different from the validation data; it does **not** by itself
mean accuracy dropped. Only feedback labels measure accuracy.

## 8. Retraining

Prepare a UTF-8 CSV with columns `description,categoryName` containing **new, human-confirmed**
transactions (labels from the 37-category taxonomy in `reference/`). Then:

```bash
python -m src.retrain --feedback confirmed.csv --validate-only   # checks only, seconds
python -m src.retrain --feedback confirmed.csv --dry-run         # trains + evaluates, changes nothing
python -m src.retrain --feedback confirmed.csv                   # promotes only if the gate passes
```

Rows that duplicate validation/test patterns are removed (leakage protection); at least 30 rows
and 3 classes are needed. A candidate trains for ≈ 2 h on CPU. Read the report in
`reports/retraining/` before relying on a promotion; with the default gate a candidate that only
ties the current model is promoted, so set `retraining.promotion.min_improvement` above 0 for
production use. To practise the workflow without real data:
`python scripts/generate_synthetic_feedback.py --mode train_variants --mode heldout_leak --mode invalid --n 200 --out data/feedback/synthetic_feedback.csv`.
