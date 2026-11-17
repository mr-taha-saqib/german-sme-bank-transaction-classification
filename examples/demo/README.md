# Synthetic API demo

`run_demo.sh` calls every public endpoint of a running API with **invented** German
transaction descriptions (`synthetic_batch.json`): health, model information, single and
batch classification (one deliberately invalid item), input validation (HTTP 422),
delayed-label feedback, monitoring summary and Prometheus metrics.

```bash
docker compose up -d api              # or: uvicorn src.api:app --port 8000
./examples/demo/run_demo.sh           # API_URL=http://host:port to target another address
```

With the real GermanFinBERT model the categories are meaningful (but trained on synthetic
data only, see MODEL_CARD.md). With the tiny demo model from
`scripts/build_demo_model.py` the predictions are meaningless; it only demonstrates the
mechanics. A captured run is in `demo_output_example.txt`.
