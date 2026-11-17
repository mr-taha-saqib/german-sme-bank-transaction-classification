#!/usr/bin/env bash
# Synthetic end-to-end demo of the classification API. Every description below is
# INVENTED for this demo; no real banking data is used.
#
#   ./examples/demo/run_demo.sh                    # API at http://localhost:8000
#   API_URL=http://host:port ./examples/demo/run_demo.sh
#
# Requires: curl, python3. The API must be running with an approved model
# (real GermanFinBERT model, or the tiny demo model from scripts/build_demo_model.py,
# whose predictions are meaningless and only show the mechanics).
set -euo pipefail
API_URL="${API_URL:-http://localhost:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pp() { python3 -m json.tool --no-ensure-ascii; }
step() { printf '\n=== %s\n$ %s\n' "$1" "$2"; }

step "1. Health" "GET /health"
curl -sS "$API_URL/health" | pp

step "2. Model information (version, threshold, limits)" "GET /model/info"
curl -sS "$API_URL/model/info" | python3 "$HERE/summarise.py" info

step "3. Classify one synthetic transaction" "POST /predict"
PRED=$(curl -sS -X POST "$API_URL/predict" -H 'content-type: application/json' \
  -d '{"description": "Betriebshaftpflicht 2026-07", "transaction_id": "demo-single"}')
echo "$PRED" | pp
PID=$(echo "$PRED" | python3 "$HERE/summarise.py" id)

step "4. Classify a synthetic batch (one item is deliberately invalid)" "POST /predict/batch"
curl -sS -X POST "$API_URL/predict/batch" -H 'content-type: application/json' \
  --data @"$HERE/synthetic_batch.json" | python3 "$HERE/summarise.py" batch

step "5. Invalid input is rejected with HTTP 422 (input is never echoed)" "POST /predict {\"description\": \"\"}"
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' -X POST "$API_URL/predict" -H 'content-type: application/json' -d '{"description": ""}'

step "6. Delayed ground truth: a reviewer confirms the category" "POST /feedback"
curl -sS -X POST "$API_URL/feedback" -H 'content-type: application/json' \
  -d "{\"prediction_id\": \"$PID\", \"true_category\": \"Versicherungen\"}" | pp

step "7. Monitoring summary (decision rates, drift status)" "GET /monitoring/summary"
curl -sS "$API_URL/monitoring/summary" | python3 "$HERE/summarise.py" summary

step "8. Prometheus metrics (first lines)" "GET /metrics"
curl -sS "$API_URL/metrics" | grep -E '^txclf_(requests_total|predictions_total|model_info)' | head -8
printf '\nDemo finished.\n'
