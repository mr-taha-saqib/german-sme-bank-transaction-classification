# Client delivery guide

Covers both projects. For a single project, start with that repository's `CLIENT_README.md`
and `CLIENT_HANDOVER.md`.

Two independent Data Science / MLOps projects, each in its own GitHub repository.
All delivery work is merged into **`main`** in both repositories (PR #1 MLOps layer, PR #2 release
instructions, PR #3 security review and dataset assessments).

| | Project 1: Transaction classification | Project 2: Movie recommendation |
|---|---|---|
| Repository | https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification | https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering |
| Delivered branch | `main` (merged from `mlops-production-delivery`) | `main` (merged from `mlops-production-delivery`) |
| Model | GermanFinBERT (`scherrmann/GermanFinBert_FP`), 37 categories, 0.80 routing threshold | truncated SVD, k = 20 (chosen on validation); separate MovieTweetings model k = 5 (not cleared for commercial use) |
| Acceptance report | `FINAL_E2E_ACCEPTANCE.md` | `FINAL_E2E_ACCEPTANCE.md` |
| One-page status | `DELIVERY_SUMMARY.md` | `DELIVERY_SUMMARY.md` |
| Exact delivery status | All 25 mandated E2E steps PASS on CPU; research results unchanged; GPU re-verification not possible (no NVIDIA GPU in the delivery environment); no suitable public real dataset for external validation | Acceptance PASS, including the real-data CI job on GitHub Actions; MovieLens research-only, MovieTweetings needs legal review |
| Security | `SECURITY.md`: fixes applied; TLS, authentication, rate limiting are deployment prerequisites | `SECURITY.md`: fixes applied; TLS, authentication, rate limiting are deployment prerequisites |
| Pull requests | [#1](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/pull/1), [#2](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/pull/2), [#3](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/pull/3): **merged** | [#1](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/pull/1), [#2](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/pull/2), [#3](https://github.com/mr-taha-saqib/SVD-Based-Collaborative-Filtering/pull/3): **merged** |
| Release tag | `v1.1.0-mlops` on the final `main` commit (notes `docs/releases/v1.1.0-mlops.md`; publishing: section 15) | `v1.0.0-mlops` on the final `main` commit (notes `docs/releases/v1.0.0-mlops.md`; publishing: section 15) |

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
git clone <repository> && cd <repository> && git checkout <release tag>   # v1.1.0-mlops / v1.0.0-mlops, or stay on main
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
| Security review | `SECURITY.md` | `SECURITY.md` |
| Dataset assessment | `docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md` | `docs/DATASET_LICENSING.md`, `docs/MOVIETWEETINGS_EVALUATION.md` |
| Release notes | `docs/releases/v1.1.0-mlops.md` | `docs/releases/v1.0.0-mlops.md` |

## 12. Known limitations

Project 1
* Trained and evaluated on **synthetic** German transactions only (no real banking data was
  used; "real" rows in the research documents means the original template-generated
  workbook rows). The test set has only 33 independent groups (one group = 3 accuracy
  points). Real-world accuracy is unknown: **authorised, labelled real transaction data is
  required before any business deployment.** No suitable public German SME dataset exists
  (8 candidates checked: `docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md`).
* About one in five automatic (≥ 0.80) decisions was wrong in the research test set;
  confidence is not a guarantee.
* Temperature scaling did not generalise; served confidences are uncalibrated.
* CPU training takes ≈ 2 h per run; a GPU is recommended for retraining.
* Not a production banking system: no authentication, ERP integration or cloud deployment.

Project 2
* **MovieLens licence: research use only. Commercial use requires written permission from
  GroupLens (the University of Minnesota offers a negotiated commercial licence, Technology ID
  z05173), or a commercially licensed / client-owned replacement dataset.** The MovieTweetings
  alternative is MIT-licensed by its author but derived from tweets and linkable to Twitter
  accounts: legal review required (`docs/DATASET_LICENSING.md`).
* The retraining gate is permissive (minimum improvement 0.0, automatic promotion): on
  MovieTweetings a noise-trained candidate was promoted. Set a stricter policy for production.
* Offline top-10 quality of the SVD model is far below a most-popular list
  (NDCG@10 0.011 vs 0.147); its strength is rating prediction.
* Random per-user split (not temporal); popularity-only cold start.

Both
* Retraining was demonstrated with **synthetic** data; real retraining needs real labels.
* No API authentication, TLS, rate limiting or horizontal scaling; monitoring state is per process.
  Compose publishes the APIs on 127.0.0.1 only; put a TLS gateway with authentication and rate
  limiting in front before exposing them (`SECURITY.md` in each repository).
* No public cloud deployment was performed.

## 13. External prerequisites

* Internet access on first run: Hugging Face (Project 1 base checkpoint, pinned revision
  `0b3622dd…`) and files.grouplens.org (Project 2 dataset, checksum-verified);
  raw.githubusercontent.com for the optional MovieTweetings pipeline (checksum-pinned).
* Docker 24+ with Compose v2 for container use.
* Optional GPU (CUDA 12.x) for Project 1 training.
* No API keys, accounts or paid services are required.

## 14. Merge status

PR #1 (MLOps layer), PR #2 (release instructions) and PR #3 (security review, dataset
assessments, MovieTweetings pipeline) are squash-merged into `main` in both repositories. The
release tags point at the `main` commit created by merging PR #3; its CI run must be green.

## 15. Remaining release and client requirements

| Item | Project | Status / requirement |
|---|---|---|
| Publish Git tags and GitHub releases | both | Release notes are committed under `docs/releases/`. Tag the final `main` commit after its CI is green (commands below). See the GitHub Releases page for the published state |
| GPU re-verification | 1 | Optional. The delivered CPU training path passed; no NVIDIA GPU/CUDA was available in the verification environment |
| Real-data validation | 1 | Required before business use: authorised, labelled real transaction data from the client (`docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md`) |
| Dataset licence | 2 | Required before commercial use: a MovieLens commercial licence, legal clearance of MovieTweetings, or client-owned data (`docs/DATASET_LICENSING.md`) |
| Deployment security | both | TLS, authentication/authorisation, rate limiting, log retention (`SECURITY.md`) |
| Retraining policy | 2 | Positive minimum improvement and/or human approval before promotion |
| Client acceptance | both | Client sign-off on the acceptance reports and agreed scope |

Publishing a tag (from a machine where Git and `gh` are authenticated with tag and release
rights). A file cannot contain the hash of the commit it is part of, so the commit is resolved
and checked at publishing time instead of being pinned here:

```bash
# Project 1 (Project 2: repository SVD-Based-Collaborative-Filtering, tag v1.0.0-mlops)
git clone https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification.git && cd german-sme-bank-transaction-classification
REL=$(git rev-parse origin/main) && echo "$REL"
gh run list --commit "$REL" --json name,conclusion      # every run must show "success"
git ls-remote --tags origin v1.1.0-mlops                # must print nothing (do not overwrite a tag)
git tag -a v1.1.0-mlops "$REL" --cleanup=verbatim -F docs/releases/v1.1.0-mlops.md
git push origin v1.1.0-mlops
gh release create v1.1.0-mlops --verify-tag --title "v1.1.0-mlops" --notes-file docs/releases/v1.1.0-mlops.md
git ls-remote origin "refs/tags/v1.1.0-mlops^{}"        # must print $REL
```

Neither project is commercially production-ready (no authentication, TLS or rate
limiting; plus the data and licence items above).
