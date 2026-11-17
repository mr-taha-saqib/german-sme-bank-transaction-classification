# API documentation: German SME transaction classification

Base URL (local): `http://localhost:8000`. Swagger UI `/docs`, ReDoc `/redoc`, schema
`/openapi.json`. Examples below were **captured from the running service** serving
`germanfinbert-seed42-40de860119` (full transcript: `docs/evidence/03_api_curl_examples.txt`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | 200 when the approved model is loaded and its weights checksum verified; 503 otherwise |
| GET | `/model/info` | version, base checkpoint + revision, categories, threshold, data checksums, validation/test metrics |
| POST | `/predict` | classify one description |
| POST | `/predict/batch` | classify up to 256 descriptions; invalid items reported individually |
| POST | `/feedback` | confirmed category for an earlier `prediction_id` (delayed ground truth) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/monitoring/summary` | decisions, confidence, drift, observed performance |
| POST | `/model/reload` | reload the approved model; disabled unless `TX_ADMIN_TOKEN` is set |

## Request rules

* `description`: string, 1 to 512 characters (configurable), not whitespace-only. The
  research preprocessing is applied server-side (dates, references, IBAN-like tokens
  removed; abbreviations expanded; umlauts kept).
* `transaction_id` (optional, ≤ 128 chars): echoed back for correlation.
* Unknown fields are rejected. Batch: 1–256 items; each item is validated separately.
* Errors never echo the submitted description.

## Response fields

| Field | Meaning |
|---|---|
| `category` / `parent_category` | predicted category (37-class taxonomy) and its parent |
| `confidence` | uncalibrated softmax probability of `category`; **a routing signal, not a guarantee** |
| `decision` | `automatic` if `confidence ≥ threshold` (0.80), else `manual_review` |
| `alternatives` | next most probable categories |
| `prediction_id` | random id; use it with `POST /feedback` |
| `calibrated` | `false` (temperature scaling is not applied; see STEP5_CALIBRATION.md) |

Status codes: 200 success · 404 unknown `prediction_id` / reload disabled · 403 wrong
admin token · 422 invalid input · 503 model unavailable · 500 unexpected error (logged
with request id).

## Examples (real responses)

```text
$ curl -s http://localhost:8000/health
{"status":"ok","model_loaded":true,"model_version":"germanfinbert-seed42-40de860119","api_version":"1.0.0","detail":null}
```

```text
$ curl -s -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"description": "Gehaltslauf 2026-01", "transaction_id": "tx-1001"}' | python -m json.tool --no-ensure-ascii
{
    "prediction_id": "7b40c566d49a4b2d85a14e9f7cf43b88",
    "transaction_id": "tx-1001",
    "category": "Boni / Provisionen",
    "parent_category": "Sonstige operative Einnahmen",
    "confidence": 0.18807,
    "decision": "manual_review",
    "threshold": 0.8,
    "alternatives": [
        {
            "category": "Zinsen",
            "probability": 0.09958
        },
        {
            "category": "Kapitalerhöhungen",
            "probability": 0.04625
        }
    ],
    "calibrated": false,
    "model_version": "germanfinbert-seed42-40de860119"
}
```

```text
$ curl -s -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"description": "Miete Büro Juni 2026"}' | python -m json.tool --no-ensure-ascii
{
    "prediction_id": "670ced1d12494e08b9ef27703ba87984",
    "transaction_id": null,
    "category": "Miete & Nebenkosten",
    "parent_category": "Betriebliche Aufwendungen",
    "confidence": 0.85904,
    "decision": "automatic",
    "threshold": 0.8,
    "alternatives": [
        {
            "category": "Immobilien / Umbauten",
            "probability": 0.01808
        },
        {
            "category": "Instandhaltung",
            "probability": 0.00988
        }
    ],
    "calibrated": false,
    "model_version": "germanfinbert-seed42-40de860119"
}
```

```text
$ curl -s -X POST http://localhost:8000/predict/batch -H "content-type: application/json" -d '{"items": [{"description": "SaaS Subscription 2025-04", "transaction_id": "a"}, {"description": "   ", "transaction_id": "b"}, {"description": "Stromabschlag Mai", "transaction_id": "c"}]}' | python -m json.tool --no-ensure-ascii
{
    "model_version": "germanfinbert-seed42-40de860119",
    "n_items": 3,
    "n_ok": 2,
    "n_errors": 1,
    "n_automatic": 1,
    "n_manual_review": 1,
    "results": [
        {
            "index": 0,
            "status": "ok",
            "transaction_id": "a",
            "result": {
                "prediction_id": "74dd77a935844988bc89ebc8d6205272",
                "transaction_id": "a",
                "category": "IT & Software",
                "parent_category": "Betriebliche Aufwendungen",
                "confidence": 0.86525,
                "decision": "automatic",
                "threshold": 0.8,
                "alternatives": [
                    {
                        "category": "Abonnements",
                        "probability": 0.07955
                    },
                    {
                        "category": "Software (CapEx)",
                        "probability": 0.00364
                    }
                ],
                "calibrated": false,
                "model_version": "germanfinbert-seed42-40de860119"
            },
            "error": null
        },
        {
            "index": 1,
            "status": "error",
            "transaction_id": "b",
            "result": null,
            "error": "description is empty or whitespace"
        },
        {
            "index": 2,
            "status": "ok",
            "transaction_id": "c",
            "result": {
                "prediction_id": "7a36958bb300413fad4d7202cfaa26dd",
                "transaction_id": "c",
                "category": "Energie",
                "parent_category": "Betriebliche Aufwendungen",
                "confidence": 0.66134,
                "decision": "manual_review",
                "threshold": 0.8,
                "alternatives": [
                    {
                        "category": "Rohstoffe",
                        "probability": 0.05374
                    },
                    {
                        "category": "Verkauf von Anlagevermögen",
                        "probability": 0.01911
                    }
                ],
                "calibrated": false,
                "model_version": "germanfinbert-seed42-40de860119"
            },
            "error": null
        }
    ]
}
```

Invalid input:

```text
$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"description": ""}'
{"error":"validation_error","detail":[{"loc":["body","description"],"msg":"String should have at least 1 character","type":"string_too_short"}],"request_id":"8486ac395c45423e"}
HTTP 422

$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"text": "Gehalt"}'
{"error":"validation_error","detail":[{"loc":["body","description"],"msg":"Field required","type":"missing"},{"loc":["body","text"],"msg":"Extra inputs are not permitted","type":"extra_forbidden"}],"request_id":"014c399766c3486a"}
HTTP 422

$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{not json'
{"error":"validation_error","detail":[{"loc":["body",1],"msg":"JSON decode error","type":"json_invalid"}],"request_id":"b365ce8e55364e44"}
HTTP 422
```

Delayed ground truth:

```text
$ PID_=$(curl -s -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{"description": "Gehaltslauf 2026-02"}' | python -c "import json,sys; print(json.load(sys.stdin)[\"prediction_id\"])"); curl -s -X POST http://localhost:8000/feedback -H "content-type: application/json" -d "{\"prediction_id\": \"$PID_\", \"true_category\": \"Löhne & Gehälter\"}"
{"accepted":true,"model_version":"germanfinbert-seed42-40de860119"}

$ curl -s -w "\nHTTP %{http_code}" -X POST http://localhost:8000/feedback -H "content-type: application/json" -d '{"prediction_id": "does-not-exist", "true_category": "Löhne & Gehälter"}'
{"error":"unknown_prediction","detail":"prediction_id not found in the monitoring window","request_id":"c6b75d48787845a5"}
HTTP 404
```

Prometheus excerpt:

```text
$ curl -s http://localhost:8000/metrics | grep -E "^txclf_(predictions_total|requests_total|model_info)"
txclf_requests_total{endpoint="/health",status="200"} 2.0
txclf_requests_total{endpoint="/model/info",status="200"} 1.0
txclf_requests_total{endpoint="/predict",status="200"} 3.0
txclf_requests_total{endpoint="/predict/batch",status="200"} 1.0
txclf_requests_total{endpoint="/predict",status="422"} 3.0
txclf_requests_total{endpoint="/feedback",status="200"} 1.0
txclf_requests_total{endpoint="/feedback",status="404"} 1.0
txclf_predictions_total{decision="manual_review"} 3.0
txclf_predictions_total{decision="automatic"} 2.0
txclf_model_info{version="germanfinbert-seed42-40de860119"} 1.0
```

Batch CLI (research inference module, same model) on the shipped examples:

```text
$ python -m src.inference --run model_registry/versions/$(python -c "import json; print(json.load(open(\"model_registry/registry.json\"))[\"approved\"])") --csv reference/example_new_transactions.csv --column description 2>/dev/null | tail -16
                                          input                             cleaned          predicted_category  confidence              second_choice  second_confidence  confidence_uncalibrated      decision
         SaaS Subscription 2026-07 RE-2026-3311                   saas subscription               IT & Software     0.86525                Abonnements            0.07955                  0.86525     automatic
                            Gehaltslauf 2026-07                         gehaltslauf          Boni / Provisionen     0.18807                     Zinsen            0.09958                  0.18807 manual_review
           Sozialversicherung Beitraege 2026-07        sozialversicherung beitraege               Sozialabgaben     0.90046                    Energie            0.00572                  0.90046     automatic
               LinkedIn Ads B2B Leadgen 2026-07            linkedin ads b2b leadgen        Marketing & Vertrieb     0.45807              IT & Software            0.17143                  0.45807 manual_review
                    Betriebshaftpflicht 2026-07                 betriebshaftpflicht              Versicherungen     0.92332    Versicherungsleistungen            0.00452                  0.92332     automatic
                        KFZ Leasingrate 2026-07                     kfz leasingrate                Leasingraten     0.90869                  Tilgungen            0.00656                  0.90869     automatic
Rechnung RE-2026-8842 Beratungsleistung 2026-07          rechnung beratungsleistung Beratung & Dienstleistungen     0.44602           Dienstleistungen            0.38059                  0.44602 manual_review
          Stromabschlag Betriebsstaette 2026-07       stromabschlag betriebsstaette                     Energie     0.54093 Verkauf von Anlagevermögen            0.05920                  0.54093 manual_review
             Zinsabschluss Kontokorrent 2026-07          zinsabschluss kontokorrent                      Zinsen     0.91530              IT / Hardware            0.00504                  0.91530     automatic
            Unklare Sammelbuchung Empfaenger XY unklare sammelbuchung empfaenger xy                Büromaterial     0.25950               Handelswaren            0.17186                  0.25950 manual_review

Threshold 0.8: 5/10 automatic, 5 routed to manual review.
Note: confidence >= threshold is a routing heuristic, not a correctness guarantee.
Confidences are uncalibrated (the primary operational result). Pass --calibration to apply a fitted temperature.
```

`/model/info` (excerpt): see the transcript; it returns the metrics shown in MODEL_CARD.md.

## Python client

```python
import httpx
with httpx.Client(base_url="http://localhost:8000", timeout=30) as c:
    r = c.post("/predict/batch", json={"items": [{"description": d} for d in ["Miete Büro Juni", "Stromabschlag Mai"]]}).json()
    for item in r["results"]:
        if item["status"] == "ok":
            res = item["result"]
            print(res["category"], res["confidence"], res["decision"])
        else:
            print("invalid item", item["index"], item["error"])
```
