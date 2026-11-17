# Client handover: German SME Bank Transaction Classification

## 1. What has been delivered

A German bank-transaction classifier (GermanFinBERT, 37 bookkeeping categories, 0.80 confidence
routing to automatic vs manual review) with a complete MLOps layer: FastAPI inference service,
model packaging and registry with explicit approval and rollback, MLflow experiment tracking,
privacy-preserving monitoring with drift detection and delayed-label accuracy, leakage-gated
retraining with a validation-only promotion gate, Docker images and Compose, GitHub Actions CI,
tests, a synthetic demo and full documentation. The original research pipeline and its published
results are unchanged.

| | |
|---|---|
| Release | **v1.1.0-mlops** |
| Repository | https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification |
| GitHub release | https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/releases/tag/v1.1.0-mlops |
| Release commit | `27d27019dfc78839805fa71ed7830040215042cb` |
| Delivery package | release + post-release delivery commits on branch `client-delivery` (PR #4): client documentation, synthetic demo, OpenAPI export, MLflow-alias rollback fix; the exact commit is in `PACKAGE_MANIFEST.md` |
| Served model | `germanfinbert-seed42-40de860119` (weights not distributed; reproducible, see below) |

## 2. Main capabilities

* `POST /predict`, `POST /predict/batch` (≤ 256): category, parent category, confidence,
  alternatives, `automatic` / `manual_review` decision.
* `POST /feedback`, `GET /monitoring/summary`, `GET /metrics`: delayed ground truth, drift, Prometheus.
* `python -m src.retrain`: validate, dry-run or promote new labelled data; rollback.
* MLflow runs and model versions (`champion` alias follows the approved version).
* Docker: hardened API container (localhost-only, non-root, read-only registry), tools image, MLflow.

## 3. How to install and run

Short version in [CLIENT_README.md](CLIENT_README.md) §8; exact, tested steps in
[INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md). In short: Python 3.12 venv → CPU PyTorch +
requirements → model (tiny demo model in minutes, or the real GermanFinBERT model in ≈ 2 h on CPU) →
`uvicorn src.api:app` or `docker compose up -d api` → `./examples/demo/run_demo.sh`.

**Model weights** (≈ 420 MB) are not in the repository or the ZIP. They are reproduced from the
public, pinned checkpoint `scherrmann/GermanFinBert_FP@0b3622dd…` (Apache-2.0) with
`train_model.py` + `python -m src.model_artifact` (INSTALLATION_AND_SETUP.md §5b). Requires internet
access to Hugging Face on the first run.

## 4. Documentation index

| Start | [CLIENT_README.md](CLIENT_README.md) |
|---|---|
| Setup | [INSTALLATION_AND_SETUP.md](INSTALLATION_AND_SETUP.md) |
| Daily use | [USER_GUIDE.md](USER_GUIDE.md), [API_REFERENCE.md](API_REFERENCE.md), `docs/openapi.json` |
| Internals | [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), ARCHITECTURE.md, MONITORING.md, RETRAINING.md, DEPLOYMENT.md |
| Quality | [TESTING_AND_VALIDATION.md](TESTING_AND_VALIDATION.md), [FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md), FINAL_E2E_ACCEPTANCE.md |
| Risk and legal | [SECURITY_AND_LIMITATIONS.md](SECURITY_AND_LIMITATIONS.md), SECURITY.md, [LICENSE_AND_DATA_ATTRIBUTION.md](LICENSE_AND_DATA_ATTRIBUTION.md), docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md |
| Model and data | MODEL_CARD.md, DATA_CARD.md |
| Research | README.md, FINAL_REPORT.md |
| History | [CHANGELOG.md](CHANGELOG.md), `docs/releases/v1.1.0-mlops.md` |

## 5. Testing evidence

Final delivery QA from a fresh clone: 60 pytest tests passed (79 % coverage), 66 + 19 + 11 + 12
research checks and 120 metric recomputations passed, lint clean, no known dependency
vulnerabilities, API and Docker checks with the real model passed, MLflow server and container
started, GitHub Actions green on the release and delivery commits. Real-model training was
executed during the delivery (≈ 2 h CPU) and not repeated in the final QA; GPU training could not
be checked (no GPU available). Details: FINAL_DELIVERY_QA_REPORT.md, `docs/evidence/`.

## 6. Known limitations

* Trained and evaluated on **synthetic** data only; research test results are not evidence of
  production accuracy. Real, authorised, labelled client data is required before business use.
* Small test set (33 groups); uncalibrated confidences; about one in five automatic decisions was
  wrong in the research test set.
* No authentication, TLS or rate limiting in the application.
* GPU training not re-verified; CPU training ≈ 2 h.
* Default promotion gate accepts a tie (`min_improvement` 0.0) and promotes automatically.

## 7. Client-side prerequisites

* Python 3.12 or Docker 24+; ≈ 5 GB RAM for CPU training; internet access to Hugging Face for the
  first training run.
* For any business use: authorised, labelled real transaction data (see
  docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md for the requested format and governance), and a
  gateway providing TLS, authentication and rate limiting.

## 8. Recommended next steps

1. Install with the tiny demo model and run `examples/demo/run_demo.sh` to see the workflow.
2. Reproduce the real model (GPU recommended) and compare with MODEL_CARD.md.
3. Provide a representative labelled sample of real transactions; evaluate the model on it
   without tuning; decide on retraining.
4. Set the promotion policy (`retraining.promotion.min_improvement` > 0, human approval).
5. Deploy behind a TLS gateway with authentication and rate limiting; integrate `/metrics` with
   your monitoring.
6. Review and accept FINAL_DELIVERY_QA_REPORT.md; agree licence/ownership terms for the code
   (LICENSE_AND_DATA_ATTRIBUTION.md §1).
