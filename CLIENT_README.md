# German SME Bank Transaction Classification: client README

**Start here.** This page explains what the project does, how to install and run it, and where
to find the details. Release `v1.1.0-mlops`; repository
https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification.

> **Important:** the model was trained and evaluated on **synthetic** German transactions only.
> It has not been validated on real bank data. It is a reference implementation for evaluation,
> not a certified banking or accounting decision system. See [Limitations](#11-important-limitations).

## 1. Business problem

Small and medium-sized German businesses receive bank statements whose booking texts
("Verwendungszweck", for example `Miete Büro Juli`, `Gehaltslauf 2026-07`) must be assigned to
bookkeeping categories. Doing this by hand is slow and repetitive. This project classifies such
descriptions automatically and sends uncertain cases to a person.

## 2. What the system does

* Classifies a German transaction description into one of **37 bookkeeping categories**
  (grouped into 11 parent categories) with a fine-tuned **GermanFinBERT** model.
* Routes every prediction: confidence ≥ **0.80** → `automatic`, otherwise `manual_review`.
* Serves predictions over a **FastAPI** REST API (single and batch).
* Records experiments in **MLflow**, keeps approved model versions in a **model registry**,
  monitors traffic and **drift** without storing any description text, and supports
  **gated retraining** with **promotion** and **rollback**.
* Ships **Docker** images, a Compose file and **GitHub Actions** CI.

## 3. Main capabilities

| Capability | Where |
|---|---|
| Research pipeline (data preparation, leakage-safe split, augmentation, training, evaluation) | `prepare_dataset.py`, `prepare_augmentation.py`, `train_model.py`, `final_evaluation.py` |
| Inference API | `src/api.py` (8 endpoints), [API_REFERENCE.md](API_REFERENCE.md) |
| Model packaging + registry (checksums, explicit approval, rollback) | `src/model_artifact.py`, `src/registry.py` |
| Experiment tracking | `src/tracking.py` (MLflow, local SQLite) |
| Monitoring (Prometheus metrics, PSI drift, delayed-label accuracy) | `src/monitoring.py` |
| Gated retraining | `src/retrain.py` |
| Containers, CI | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| Synthetic demo | `examples/demo/run_demo.sh` |

## 4. Technology stack

Python 3.12, PyTorch 2.13 (CPU or CUDA), Hugging Face Transformers 5.16, FastAPI 0.163,
Uvicorn, Pydantic 2, MLflow 3.18, prometheus-client, pandas, scikit-learn, Docker 24+ with
Compose v2, GitHub Actions, pytest, ruff.

## 5. Architecture overview

```
labelled CSV / research data ──> train_model.py ──> run directory ──> src/model_artifact.py
                                                                         │ package + checksum
                                                                         ▼
                      MLflow (runs, model versions, alias "champion") <── model_registry/ (approved pointer)
                                                                         │ approved version
                                                                         ▼
client ──HTTP──> FastAPI (src/api.py) ──> TransactionClassifier ──> category + confidence + decision
                     │
                     └──> monitoring (src/monitoring.py): metrics, drift, feedback accuracy
confirmed labels ──> src/retrain.py ──> candidate ──> promotion gate (validation only) ──> registry
```

Details: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## 6. Repository structure (main parts)

```
src/                 research modules + MLOps layer (api, registry, monitoring, retrain, tracking)
tests/               pytest suites (MLOps layer, research checks) + tests/test_pipeline.py
scripts/             demo model builder, traffic simulator, synthetic feedback, MLflow import
configs/mlops.yaml   threshold, registry, MLflow, monitoring, retraining and API settings
data/                synthetic research data (no real banking data)
reference/           taxonomy workbook, example new transactions
results/             frozen split, published research results (unchanged)
examples/demo/       synthetic API demo
docs/                evidence logs, release notes, OpenAPI schema, dataset assessment
Dockerfile, docker-compose.yml, requirements*.txt, pyproject.toml, .env.example
```

## 7. Requirements

* Python **3.12** (3.11 is not supported by the pinned numpy). Verified on Linux (Ubuntu 24.04
  base, x86-64); Docker images are Debian-based `python:3.12-slim`.
* 2+ CPU cores, ≈ 5 GB RAM for CPU training, ≈ 1 GB for the API.
* Docker 24+ with Compose v2 for the container route.
* Internet on the first training run (Hugging Face base checkpoint). No API keys are needed.
* Optional NVIDIA GPU for training (not re-verified in the delivery environment).

## 8. Installation and start (short version)

Full, tested instructions: [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md).

```bash
git clone https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification.git
cd german-sme-bank-transaction-classification
git checkout client-delivery     # = the delivered ZIP (main once PR #4 is merged)
python3.12 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
pip install -r requirements-dev.txt -r requirements.txt
```

The model weights (≈ 420 MB) are **not** in the repository. Either:

* **Quick look (minutes, offline):** `python scripts/build_demo_model.py --registry model_registry`
  builds a tiny test model. Its predictions are meaningless; it only demonstrates the mechanics.
* **Real model (≈ 2 h on CPU, minutes on a GPU):** train and register GermanFinBERT as described in
  [INSTALLATION_AND_SETUP.md §5](INSTALLATION_AND_SETUP.md#5-model-weights).

Then start the API:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000          # local
# or
docker compose up -d api                                  # container, http://127.0.0.1:8000
curl http://127.0.0.1:8000/health
```

## 9. Using the API

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' \
     -d '{"description": "Betriebshaftpflicht 2026-07", "transaction_id": "tx-1"}'
./examples/demo/run_demo.sh            # full synthetic walk-through of every endpoint
```

Interactive documentation: http://127.0.0.1:8000/docs. Reference: [API_REFERENCE.md](API_REFERENCE.md).
Typical workflows: [USER_GUIDE.md](USER_GUIDE.md).

## 10. Tests and the MLOps workflow

```bash
ruff check .                     # lint (configured scope: MLOps layer + tests)
pytest -q                        # 60 tests, CPU, uses a tiny offline test model
python tests/test_pipeline.py    # 66 research pipeline checks
python -m src.retrain --feedback new_labelled.csv --dry-run    # gated retraining, changes nothing
python -m src.retrain --rollback --reason "..."                # re-approve the previous model
```

Results of the final audit: [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md),
[FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md).

## 11. Important limitations

* **Synthetic data only.** Training and evaluation use template-generated and LLM-generated
  transactions. Research test results are **not** evidence of accuracy on real bank data.
  **Authorised, labelled real transaction data from the client is required** before any business
  validation ([docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md](docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md):
  no suitable public dataset exists).
* The test set has only 33 independent groups; about one in five automatic decisions was wrong
  on it. **Confidence values are uncalibrated** and are a routing signal, not a guarantee.
* GPU training was **not re-verified** in the delivery environment; the delivered model was
  trained on CPU.
* The API has **no authentication, TLS or rate limiting**; these must be provided by your
  deployment (reverse proxy / API gateway). See [SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md).
* Model weights are not distributed; they are reproduced from the pinned public checkpoint.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health` returns 503 "no approved model" | no approved version in `model_registry/`: build the demo model or train + register the real one |
| `/health` 503 "checksum mismatch" | weights changed after packaging: re-package with `python -m src.model_artifact` |
| `pip install` fails on numpy | Python older than 3.12 |
| Docker build fails behind a corporate proxy | build with `--secret id=pip_ca,src=<proxy-ca.pem>` ([INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md#7-docker)) |
| API not reachable from another machine | by design Compose binds to 127.0.0.1; expose only behind a TLS proxy (`API_BIND`) |
| Training is very slow | CPU training takes ≈ 2 h; use a CUDA GPU if available |

## 13. Documentation index

| Document | Purpose |
|---|---|
| [CLIENT_HANDOVER.md](CLIENT_HANDOVER.md) | what was delivered, release, next steps |
| [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md) | exact setup, Docker, MLflow, shutdown |
| [USER_GUIDE.md](USER_GUIDE.md) | workflows, examples, model management, monitoring, retraining |
| [API_REFERENCE.md](API_REFERENCE.md) | every endpoint with schemas and real responses |
| [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | architecture and implementation details |
| [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md) | tests, CI, evaluation, final audit results |
| [SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md) | security status, privacy, licensing, production prerequisites |
| [LICENSE_AND_DATA_ATTRIBUTION.md](LICENSE_AND_DATA_ATTRIBUTION.md) | software, model and data licences |
| [FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md) | final QA checks with status |
| [CHANGELOG.md](CHANGELOG.md) | release history |
| [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md) | model and data facts |
| [README.md](README.md), [FINAL_REPORT.md](FINAL_REPORT.md) | full research documentation |
