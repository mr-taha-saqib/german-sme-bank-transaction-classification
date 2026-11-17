# Technical documentation

Release `v1.1.0-mlops`. This document describes what is implemented, with file references.
Deeper material: [ARCHITECTURE.md](ARCHITECTURE.md), [MONITORING.md](MONITORING.md),
[RETRAINING.md](RETRAINING.md), [DEPLOYMENT.md](DEPLOYMENT.md), [MODEL_CARD.md](MODEL_CARD.md),
[DATA_CARD.md](DATA_CARD.md), research: [README.md](README.md), [FINAL_REPORT.md](FINAL_REPORT.md).

## 1. Overall architecture

Two layers share one code base:

1. **Research layer** (unchanged since the research delivery): data preparation, label-free
   grouping, frozen leakage-safe split, gated augmentation (corpus v2), fine-tuning of
   GermanFinBERT and FinBERT over five seeds, temperature-scaling experiment, final evaluation.
2. **MLOps layer**: model packaging and registry, MLflow tracking, FastAPI inference,
   monitoring, gated retraining, Docker and CI. It reuses the research code (preprocessing,
   split loader, training loop, inference) instead of re-implementing it.

## 2. Data flow

1. `data/synthetic_b2b_transactions_2000.xlsx` (2,000 synthetic, template-generated rows) →
   `prepare_dataset.py`: cleaning (`src/preprocessing.py`, 14 deterministic rules), label-free
   grouping (token-Jaccard ≥ 0.60), frozen split `results/dataset/split_assignment.csv`
   (train 1,093 / validation 228 / test 647 rows; 103 / 16 / 33 groups; SHA-256 `f715f040…`).
2. `data/final_synthetic_clean_7033.csv` (LLM-generated) → `prepare_augmentation.py`: gates A
   (provenance), B (collision with validation/test), C (near-duplicate) → corpus v2
   (4,541 rows, SHA-256 `4eb242d9…`).
3. `train_model.py`: fine-tunes on train + corpus v2; writes a run directory (weights, logits,
   metrics, config).
4. `src/model_artifact.py`: packages a run into `model_registry/versions/<id>/` with checksums,
   a monitoring reference profile and metadata; optional explicit approval.
5. API serves the **approved** version; monitoring records numeric features only.
6. Confirmed labels from the client's bookkeeping feed `src/retrain.py`.

No real banking data enters this flow.

## 3. Model architecture and inference

* `BertForSequenceClassification`, ≈ 110 M parameters, 37-class head, base checkpoint
  `scherrmann/GermanFinBert_FP` at pinned revision `0b3622dd589d2fc2f518b4be12a9dc306b903df2`.
* Training configuration (research, unchanged): corpus v2, learning rate 2e-5, batch 16,
  `max_length` 128, 3 fixed epochs, class-weighted focal loss (γ = 2), AdamW, seed 42.
* Inference (`src/inference.py`, `TransactionClassifier`): the run's preprocessing → tokenizer
  → softmax over 37 classes → top category, parent category, top-k alternatives (k = 3, the top
  category plus two alternatives in the response) → routing `automatic` if confidence ≥ 0.80
  else `manual_review`. Probabilities are **uncalibrated** (temperature scaling did not
  generalise in the research, see `STEP5_CALIBRATION.md`).
* Delivered version: `germanfinbert-seed42-40de860119`, trained on CPU (fp32, ≈ 1 h 54 min).

## 4. Training and evaluation workflow

| Step | Command | Output |
|---|---|---|
| Dataset checks | `python verify_dataset.py` | 19 checks |
| Augmentation checks | `python verify_augmentation.py` | 11 checks |
| Train one model | `python train_model.py --model germanfinbert --augment --augmentation-version v2 --seed 42 --output-dir artifacts/runs --run-name <name>` | run directory |
| Final evaluation (research) | `python final_evaluation.py` | 12 integrity checks + 120 metric recomputations from saved logits |
| Package + approve | `python -m src.model_artifact --run artifacts/runs/<name> --approve --reason "..."` | registry version |

Evaluation uses the frozen test split once; group-level metrics (33 groups) are primary because
rows within a group are near-duplicates. Published research result (5 seeds, GPU): group
accuracy 0.6485 ± 0.0271. Delivered CPU model: 0.6970 (23/33), within the seed spread.

## 5. API architecture

`src/api.py` builds a FastAPI app (`create_app`) around a `ServiceState` that loads the approved
registry version at start-up (verifying the weights SHA-256) and owns the monitor.

* Endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch`,
  `POST /feedback`, `GET /metrics`, `GET /monitoring/summary`, `POST /model/reload`
  ([API_REFERENCE.md](API_REFERENCE.md), schema `docs/openapi.json`).
* Validation: Pydantic models with `extra="forbid"`; description 1–512 characters
  (`api.max_text_length`, schema hard cap 2,000), not whitespace-only; batch 1–256 items
  (`api.max_batch_size`), invalid batch items reported per item.
* Errors: uniform JSON `{"error", "detail", "request_id"}`; submitted text is never echoed.
* Middleware: request id (`X-Request-ID` honoured or generated), latency, structured access log
  (method, endpoint, status, latency, request id; never bodies).
* `/model/reload` is disabled (404) unless `TX_ADMIN_TOKEN` is set; token compared in constant time.

## 6. MLflow integration

`src/tracking.py` (`Tracker`). Default URI `sqlite:///mlflow.db`, artifacts `mlartifacts/`,
experiment `german-sme-transaction-classification`. Logged: parameters, per-epoch metrics,
validation/test metrics, data checksums, git commit. Registered model
`german-sme-transaction-classifier`; each version is tagged `registry_version=<registry id>`;
alias `champion` mirrors the approved registry version (also after a rollback, since the
post-release fix in this delivery), `rejected-latest` marks the last rejected candidate.
Weights are logged to MLflow only if `mlflow.log_weights: true` (default false, ≈ 420 MB).
The 10 published research runs were imported read-only (`scripts/import_research_runs_to_mlflow.py`).

## 7. Model registry

`src/registry.py`: `model_registry/registry.json` (approved pointer, history, event log) and
`versions/<id>/` (weights + tokenizer, config, label mapping, metrics, reference profile,
metadata with checksums). Writes are atomic and guarded by a lock file. Registration never
approves; approval and rollback require a reason and are recorded. The API refuses a version
whose weights checksum does not match. Weights are never committed to git.

## 8. Monitoring and drift detection

`src/monitoring.py`. Per prediction it stores a random `prediction_id`, category, confidence,
decision and four bucketed numeric features (character length, token count, unseen-token share,
digit share), **never the text**. Signals: request/latency/error counters, decision and
category counts, confidence histogram (Prometheus `txclf_*`); input drift (PSI of the four
features vs the held-out validation + test reference); prediction drift (PSI vs the model's
predictions on the reference); observed accuracy from `POST /feedback` (only with ≥ 30 labels).
PSI status: ok < 0.10 ≤ warning < 0.25 ≤ alert; `insufficient_data` below 50 predictions.
Event log: `monitoring/events.jsonl`. In-memory metrics reset on restart.

## 9. Retraining workflow

`python -m src.retrain --feedback confirmed.csv [--validate-only | --dry-run]`. Input: CSV
`description,categoryName` of new, human-confirmed transactions. Steps: load approved version and
verify the frozen split; validate rows; leakage protection with research gates B and C against
validation/test; minimum rows (30) and classes (3); train a candidate with the approved
configuration on train + corpus v2 + accepted rows; evaluate on **validation only** (test is never
read); apply the promotion gate; register; promote only if every criterion passes; JSON + Markdown
report in `reports/retraining/`.

## 10. Promotion and rollback

Gate (`configs/mlops.yaml` → `retraining.promotion`): row macro-F1 ≥ approved + `min_improvement`
(0.0), row accuracy drop ≤ 0.01, accuracy among automatic decisions drop ≤ 0.02, automatic
coverage drop ≤ 5 percentage points. Promotion is automatic when the gate passes (not with
`--dry-run`); the previous version remains for rollback: `python -m src.retrain --rollback
--reason "..."`. A running API picks up a new approved version via `POST /model/reload` with
`X-Admin-Token`, or on restart. **Production note:** with `min_improvement` 0.0 a candidate that
merely ties the approved model passes; set a positive margin and/or a human approval step.

## 11. Configuration and environment variables

`configs/mlops.yaml` (or `$TX_CONFIG`) holds all settings; environment variables override:

| Variable | Default | Purpose |
|---|---|---|
| `TX_REGISTRY_DIR` | `model_registry` | registry location |
| `TX_MONITORING_DIR` | `monitoring` | event log directory |
| `TX_RETRAINING_REPORT_DIR` | `reports/retraining` | retraining reports |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend (or `http://host:5000`) |
| `TX_MLFLOW_ENABLED` | `true` | disable tracking |
| `TX_MLFLOW_ARTIFACTS` | `mlartifacts` | MLflow artifact location |
| `TX_MAX_BATCH_SIZE` | `256` | batch limit |
| `TX_ADMIN_TOKEN` | unset | enables `/model/reload` (secret) |
| `TX_LOG_LEVEL`, `LOG_FORMAT` | `INFO`, `json` (API) | logging |
| `OMP_NUM_THREADS` | `2` | PyTorch CPU threads |
| `API_BIND`, `API_PORT` | `127.0.0.1`, `8000` | Compose port publishing |
| `HF_HOME`, `HF_TOKEN` | – | Hugging Face cache / optional token (rate limits only) |
| `TX_API_NO_AUTOCREATE` | – | `1` stops the module from building an app at import (tests) |
| `GIT_COMMIT` | – | recorded commit when `.git` is absent (e.g. in images) |

## 12. Docker architecture

`Dockerfile` targets: `base` (API; CPU PyTorch + `requirements-api.txt`; user `app` uid 10001;
health check on `/health`; `CMD uvicorn src.api:app`) and `tools` (adds training/research
dependencies). The base image's pip/setuptools are upgraded; an optional BuildKit secret
`pip_ca` supports TLS-inspecting proxies without storing the CA in a layer.
`docker-compose.yml`: service `api` (port `127.0.0.1:8000`, `model_registry` bind-mounted
**read-only**, `monitoring` volume, `cap_drop: ALL`, `no-new-privileges`, 3 GB memory limit;
the API image sets `HF_HUB_OFFLINE=1`, so serving never contacts Hugging Face), profile `tools`, profile `mlflow` (official `ghcr.io/mlflow/mlflow:v3.18.0`
server on `127.0.0.1:5000`).

## 13. CI/CD workflow

`.github/workflows/ci.yml` runs on pull requests, pushes to `main` (and the historical delivery
branch name) and manual dispatch:

1. **Research integrity** (no deep-learning stack): frozen split (19 checks), augmentation
   corpus (11), final results integrity and recomputation from saved logits.
2. **Lint + tests**: ruff (MLOps layer), pytest with a tiny BERT built offline (no model download).
3. **Docker**: build the API image, build a tiny demo model, start the container, smoke-test
   `/health` and `/predict`.

There is no automatic deployment (CD); releases are tagged manually.

## 14. Security considerations

Summarised in [SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md); review details in
[SECURITY.md](SECURITY.md). Key points: no authentication/TLS/rate limiting in the application
(gateway prerequisites); localhost-only Compose defaults; non-root, capability-dropped container
with read-only registry; descriptions never logged or stored; pinned dependencies audited.

## 15. Known technical limitations

* Synthetic training and evaluation data; no real-data validation (client data required).
* 33 independent test groups → wide uncertainty; uncalibrated confidences.
* Monitoring state is per process (in-memory metrics reset on restart; no horizontal scaling).
* CPU training ≈ 2 h per run; GPU path not re-verified in the delivery environment.
* `final_evaluation.py` breaks ties in the descriptive `*_modal_prediction` column of
  `results/final_error_analysis.csv` by set order, so two tied cells can differ between runs
  (`PYTHONHASHSEED`); metrics are unaffected (FINAL_E2E_ACCEPTANCE.md §10).
* Permissive default promotion gate (see §10).
