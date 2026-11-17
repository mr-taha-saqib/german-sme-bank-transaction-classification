# Client delivery guide

Two independent Data Science / MLOps projects, each in its own GitHub repository.
The MLOps delivery work has been merged into **`main`** in both repositories. The pull requests are closed as merged.

| | Project 1: Transaction classification | Project 2: Movie recommendation |
|---|---|---|
| Repository | https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification | https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering |
| Delivered branch | `main` (merged from `mlops-production-delivery`) | `main` (merged from `mlops-production-delivery`) |
| Model | GermanFinBERT (`scherrmann/GermanFinBert_FP`), 37 categories, 0.80 routing threshold | truncated SVD, k = 20 (chosen on validation) |
| Acceptance report | `FINAL_E2E_ACCEPTANCE.md` | `FINAL_E2E_ACCEPTANCE.md` |
| One-page status | `DELIVERY_SUMMARY.md` | `DELIVERY_SUMMARY.md` |
| Exact delivery status | All 25 mandated E2E steps PASS on CPU; research results unchanged; GPU re-verification not possible (no NVIDIA GPU in the delivery environment) | Acceptance PASS, including the real-data CI job on GitHub Actions |
| Pull request | [PR #1](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/pull/1), **merged** | [PR #1](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/pull/1), **merged** |
| Release tag | `v1.1.0-mlops`: notes ready, **not yet published** (section 15) | `v1.0.0-mlops`: notes ready, **not yet published** (section 15) |

## 1. What each project does

**Project 1** classifies short German bank-transaction descriptions into 37 bookkeeping
categories and routes each prediction: confidence ≥ 0.80 → automatic, otherwise
manual review. The research (leakage-safe grouping and split, augmentation, five-seed
comparison with FinBERT, calibration experiment) is preserved unchanged; an inference
API and MLOps tooling were added around it.

**Project 2** recommends movies from MovieLens ratings with a truncated SVD and predicts
ratings. A test-set leakage defect in the original evaluation (rank chosen on the test
set) was corrected; historical results are kept and labelled.

## 2. Features delivered (both projects)

FastAPI service with Swagger docs · input validation and structured errors · health,
model-info, prediction and monitoring endpoints · versioned model registry with
explicit approval, rejection records and rollback · MLflow experiment tracking (local,
no credentials) · Prometheus metrics, drift checks (PSI) and delayed-label performance ·
gated retraining with dry-run · Docker images and docker-compose · GitHub Actions CI ·
unit, API and integration tests · full documentation set.

## 3. Technology stack

Python 3.12+, NumPy, pandas, scikit-learn, PyTorch 2.13 (CPU or CUDA) and Hugging Face
Transformers 5.16 (Project 1), FastAPI + Uvicorn + Pydantic 2, Prometheus client,
MLflow 3.18 (SQLite), Docker / Compose, GitHub Actions, pytest, ruff.

## 4. Installation

```bash
git clone <repository> && cd <repository> && git checkout mlops-production-delivery
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Project 1:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0   # or the CUDA build
pip install -r requirements-dev.txt
```

Project 2:
```bash
pip install -r requirements-dev.txt
```

## 5. Models

Project 1 weights (≈ 420 MB) are not in git. Train and register (GPU ≈ 2 min, CPU ≈ 2 h):
```bash
python train_model.py --model germanfinbert --augment --augmentation-version v2 --seed 42 \
    --output-dir artifacts/runs --run-name germanfinbert_seed42
python -m src.model_artifact --run artifacts/runs/germanfinbert_seed42 --approve --reason "<your review>"
```

Project 2 (downloads MovieLens, ≈ 1 min on CPU):
```bash
python src/main.py --approve --approval-reason "<your review>"
```

## 6. Start the APIs

| | Command | Swagger |
|---|---|---|
| Project 1 | `uvicorn src.api:app --port 8000` | http://localhost:8000/docs |
| Project 2 | `uvicorn api:app --app-dir src --port 8001` | http://localhost:8001/docs |

## 7. Docker

Project 1 (mounts `./model_registry` read-only):
```bash
docker compose build api && docker compose up -d api && docker compose logs -f api
docker compose down
```

Project 2 (all state in named volumes):
```bash
docker compose --profile tools run --rm trainer   # download, train, approve
docker compose up -d api
docker compose down
```

Change `API_PORT` in `.env` if both run on one machine.

## 8. MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # http://localhost:5000
```
Project 1: `python scripts/import_research_runs_to_mlflow.py` adds the 10 published
research runs for comparison. Both: `docker compose --profile mlflow up -d mlflow`.

## 9. Tests

```bash
pytest                       # both projects; offline; Project 1 uses a tiny test BERT
ruff check .                 # Project 1   |   ruff check src tests scripts   (Project 2)
python tests/test_pipeline.py && python verify_dataset.py && python verify_augmentation.py   # Project 1 research checks
```

## 10. Retraining

Project 1 (CSV `description,categoryName`, human-confirmed labels):
```bash
python -m src.retrain --feedback confirmed.csv --dry-run
python -m src.retrain --feedback confirmed.csv
python -m src.retrain --rollback --reason "..."
```

Project 2 (CSV `userId,movieId,rating[,timestamp]`, e.g. from `POST /feedback`):
```bash
python src/retrain.py --feedback data/feedback/api_feedback.csv --dry-run
python src/retrain.py --feedback data/feedback/api_feedback.csv
python src/retrain.py --rollback --reason "..."
```

A candidate is promoted only if it passes explicit criteria on the **validation**
data; the test data is never used. The previous model always remains available.

## 11. Where to find things

| | Project 1 | Project 2 |
|---|---|---|
| Acceptance report | `FINAL_E2E_ACCEPTANCE.md` (+ research acceptance `FINAL_E2E_ACCEPTANCE_RESEARCH.md`) | `FINAL_E2E_ACCEPTANCE.md` |
| Evidence logs | `docs/evidence/` | `docs/evidence/` |
| API reference | `API_DOCUMENTATION.md` | `API_DOCUMENTATION.md` |
| Model / data cards | `MODEL_CARD.md`, `DATA_CARD.md` | `MODEL_CARD.md`, `DATA_CARD.md` |
| Operations | `DEPLOYMENT.md`, `MONITORING.md`, `RETRAINING.md` | same |

## 12. Known limitations

Project 1
* Trained and evaluated on **synthetic** German transactions only (no real banking data was
  used; "real" rows in the research documents means the original template-generated
  workbook rows). The test set has only 33 independent groups (one group = 3 accuracy
  points). Real-world accuracy is unknown: **authorised, labelled real transaction data is
  required before any business deployment.**
* About one in five automatic (≥ 0.80) decisions was wrong in the research test set;
  confidence is not a guarantee.
* Temperature scaling did not generalise; served confidences are uncalibrated.
* CPU training takes ≈ 2 h per run; a GPU is recommended for retraining.
* Not a production banking system: no authentication, ERP integration or cloud deployment.

Project 2
* **MovieLens licence: research use only. Commercial use requires written permission from
  GroupLens, or a commercially licensed / client-owned replacement dataset.**
* Offline top-10 quality of the SVD model is far below a most-popular list
  (NDCG@10 0.011 vs 0.147); its strength is rating prediction.
* Random per-user split (not temporal); popularity-only cold start.

Both
* Retraining was demonstrated with **synthetic** data; real retraining needs real labels.
* No API authentication, TLS, rate limiting or horizontal scaling; monitoring state is per process.
* No public cloud deployment was performed.

## 13. External prerequisites

* Internet access on first run: Hugging Face (Project 1 base checkpoint, pinned revision
  `0b3622dd…`) and files.grouplens.org (Project 2 dataset, checksum-verified).
* Docker 24+ with Compose v2 for container use.
* Optional GPU (CUDA 12.x) for Project 1 training.
* No API keys, accounts or paid services are required.

## 14. Merge status

Both PR #1s have been squash-merged into `main`.

* Project 1 merge commit: `18539bd97207ae8031f04b157652f05beccb9b1e`
* Project 2 merge commit: `12a8100bdbebdeb598d093db4069d37ebcc7cdfc`

A new GitHub Actions run is triggered on each merge commit. Check the `main` branch workflow status before creating the releases.

## 15. Remaining release and client requirements

| Item | Project | Status / requirement |
|---|---|---|
| Publish Git tags and GitHub releases | both | Release notes are committed under `docs/releases/`; tags/releases are not yet published. Use a machine with GitHub tag/release permissions and tag the current approved `main` commit after its merge-triggered CI passes. |
| GPU re-verification | 1 | Optional. The delivered CPU training path passed; no NVIDIA GPU/CUDA was available in the verification environment. |
| Real-data validation | 1 | Required before business use: obtain authorised, labelled real transaction data from the client and evaluate/retrain with appropriate governance. |
| Dataset licence | 2 | Required before commercial use: obtain written GroupLens permission or replace MovieLens with a commercially licensed/client-owned dataset. |
| Client acceptance | both | Obtain client sign-off on the acceptance reports and agreed scope. |

The release notes are in `docs/releases/`. The approved squash-merged implementation and updated guide are on `main`. After the current `main` workflow is green, publish the release using Git and GitHub CLI (`gh`) from a session with tag/release write permissions. The pinned commit below is the verified current `main` commit at handover.

```bash
# Project 1
cd german-sme-bank-transaction-classification
git fetch origin && git checkout main && git pull --ff-only origin main
git tag -a v1.1.0-mlops fa15261e575a7998aa7480fc602952d22ee03a3e --cleanup=verbatim -F docs/releases/v1.1.0-mlops.md
git push origin v1.1.0-mlops
gh release create v1.1.0-mlops --verify-tag --title "v1.1.0-mlops" --notes-file docs/releases/v1.1.0-mlops.md

# Project 2
cd ../SVD-Based-Collaborative-Filtering
git fetch origin && git checkout main && git pull --ff-only origin main
git tag -a v1.0.0-mlops e72c0dafa87c988b57160db695bc39a099443a73 --cleanup=verbatim -F docs/releases/v1.0.0-mlops.md
git push origin v1.0.0-mlops
gh release create v1.0.0-mlops --verify-tag --title "v1.0.0-mlops" --notes-file docs/releases/v1.0.0-mlops.md
```

Neither project is commercially production-ready (no authentication, TLS or rate
limiting; plus the data and licence items above).
