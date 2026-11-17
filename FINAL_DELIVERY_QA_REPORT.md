# Final delivery QA report: German SME Bank Transaction Classification

| | |
|---|---|
| Release | `v1.1.0-mlops` → commit `27d27019dfc78839805fa71ed7830040215042cb` (annotated tag, GitHub release published) |
| QA'd commit | `4b1f66d45cd5a4832b7191e424f2210120065741` (branch `client-delivery`, PR #4): the release plus a post-release delivery commit (documentation, synthetic demo, OpenAPI export, MLflow-alias rollback fix + test). The QA report, testing and handover documents were added afterwards in a documentation-only commit |
| Date | 2026-11-17/18 (UTC) |
| Environment | Ubuntu 24.04.5, x86-64, Python 3.12.3, Docker (BuildKit), CPU only; outbound traffic via a TLS-inspecting proxy |
| Method | fresh `git clone` of the QA'd commit, new virtual environment, documented commands; every command and its output: [`docs/evidence/16_final_delivery_qa.log`](docs/evidence/16_final_delivery_qa.log) |

Statuses: **PASS**, **FAIL**, **BLOCKED**, **NOT RUN**, **NOT APPLICABLE**. No check failed.

## 1. Installation and static checks

| # | Check | Command (abridged) | Result | Status |
|---|---|---|---|---|
| 1 | Source baseline | `git describe`, `git diff --stat v1.1.0-mlops HEAD` | tag reachable; 20 files changed vs the release (docs, demo, `src/retrain.py`, `src/tracking.py`, one test) | PASS |
| 2 | Clean environment | `python3.12 -m venv .venv`; `pip install --upgrade pip` | Python 3.12.3, pip 26.3 | PASS |
| 3 | Dependency installation | `pip install --index-url …/whl/cpu torch==2.13.0`; `pip install -r requirements-dev.txt -r requirements.txt` | torch 2.13.0+cpu, 121 distributions | PASS |
| 4 | Dependency consistency | `pip check` | no broken requirements | PASS |
| 5 | Dependency vulnerabilities | `pip-audit --path .venv/…/site-packages`; torch audited as `torch==2.13.0` | no known vulnerabilities (torch: none for the PyPI version). Note: Ubuntu's venv ships pip 24.0 with advisories; the documented `pip install --upgrade pip` step removes it | PASS |
| 6 | Lint | `ruff check .` | all checks passed | PASS |
| 7 | Formatting | – | no formatter configured in `pyproject.toml` or CI | NOT APPLICABLE |
| 8 | Compose configuration | `docker compose config -q` | valid; services `api`, `tools`, `mlflow` | PASS |

## 2. Automated tests

| # | Check | Command | Result | Status |
|---|---|---|---|---|
| 9 | Unit tests (components) | `pytest tests/test_mlops_components.py` | 10 passed | PASS |
| 10 | API tests | `pytest tests/test_mlops_api.py` | 21 passed | PASS |
| 11 | Retraining / integration tests (tiny BERT, real MLflow SQLite) | `pytest tests/test_mlops_retrain.py` | 19 passed (incl. new rollback-alias test) | PASS |
| 12 | Full suite + coverage | `pytest --cov=src` | **60 passed** (10 + 21 + 19 + 10 research checks collected from `tests/test_pipeline.py`), coverage 79 % of `src/` | PASS |
| 13 | Research pipeline checks | `python tests/test_pipeline.py` | 66 passed, 0 failed | PASS |
| 14 | Frozen dataset | `python verify_dataset.py` | 19 checks passed | PASS |
| 15 | Augmentation corpus v2 | `python verify_augmentation.py` | 11 checks passed, 0 findings | PASS |
| 16 | Final evaluation | `python final_evaluation.py` | 12/12 integrity checks, 120 metric recomputations matched | PASS |
| 17 | Regenerated results vs committed | content comparison (line endings ignored, NaN-aware) | `final_results.csv`, `final_summary.csv`, `augmentation_verification.json`: identical; `final_results.json`: identical in one run, in another only the known tie-break field of group 132; `final_error_analysis.csv`: one tie-break cell (known, `PYTHONHASHSEED`-dependent, metrics unaffected; FINAL_E2E_ACCEPTANCE.md §10) | PASS (documented exception) |

## 3. End-to-end, API and MLOps workflow

| # | Check | Result | Status |
|---|---|---|---|
| 18 | Quick path: tiny demo model | `scripts/build_demo_model.py` built and approved `tinybert-demo-seed42-80ba659431` offline | PASS |
| 19 | Local API start + health | `uvicorn src.api:app` → `/health` HTTP 200 | PASS |
| 20 | API example requests (demo script, demo model) | all 8 steps completed; invalid item reported; HTTP 422 for empty input | PASS |
| 21 | Retraining data validation | `python -m src.retrain --feedback … --validate-only`: leakage gate B dropped 21 rows colliding with validation/test patterns; decision `validated_only` | PASS |
| 22 | Rollback without history | clean error "No previous approved version available for rollback" (no traceback) | PASS |
| 23 | Promotion + rollback + MLflow alias | covered by tests 11 (tiny model, real MLflow); end-to-end with the real registry in `docs/evidence/05_*`, `06_*` | PASS |
| 24 | Real GermanFinBERT training (≈ 2 h CPU) | not repeated; executed during delivery (`docs/evidence/02_model_training_and_registration.log`) | NOT RUN (time) |
| 25 | Real-model retraining dry run (≈ 2 h) | not repeated; mechanics covered by 11 and 21 | NOT RUN (time) |
| 26 | GPU training | no NVIDIA GPU / CUDA available | BLOCKED (hardware) |
| 27 | MLflow server startup + connectivity | `mlflow server` on the delivered tracking DB: `/health` OK; experiment `german-sme-transaction-classification`; 11 runs; alias `champion` → version 1 = `germanfinbert-seed42-40de860119` | PASS |

## 4. Docker (real GermanFinBERT model)

| # | Check | Result | Status |
|---|---|---|---|
| 28 | Build API image | `german-sme-txclf-api:1.0.0` `7bb271b14f00`, 1.78 GB (built with the documented `--secret id=pip_ca` because of the proxy) | PASS |
| 29 | Start + health | `docker compose up -d api` → healthy; `/health` 200, model `germanfinbert-seed42-40de860119` | PASS |
| 30 | Model loading / inference | `Betriebshaftpflicht 2026-07` → Versicherungen, 0.92332, automatic | PASS |
| 31 | Demo script against the container | all 8 steps; batch 5 ok / 1 error, all manual review | PASS |
| 32 | Invalid inputs | empty, 513 characters, 257 batch items, unknown field → 422; reload without token → 404 | PASS |
| 33 | Hardening | published on `127.0.0.1:8000` only; user `app` (uid 10001); `CapDrop [ALL]`; `no-new-privileges`; memory limit 3 GiB; registry mount read-only (write refused); code not writable | PASS |
| 34 | Persistence | healthy after `docker compose restart api`; 7 monitoring events kept | PASS |
| 35 | Privacy | 0 occurrences of submitted description text in the event log (only category labels and numeric buckets) | PASS |
| 36 | Drift detection | drift traffic: all four input signals `alert`; normal traffic `ok` (same code, `docs/evidence/14_session3_monitoring_drift.log`) | PASS |
| 37 | Tools image + research check in a container | `german-sme-txclf-tools:1.0.0` (2.69 GB) built; `verify_dataset.py` in the container: 19 checks passed | PASS |
| 38 | MLflow container (`--profile mlflow`) | `ghcr.io/mlflow/mlflow:v3.18.0` pulled and started; `/health` OK on 127.0.0.1:5000 | PASS |
| 39 | Clean-up | `docker compose down -v`: 0 containers, 0 volumes left | PASS |

## 5. Documentation and CI

| # | Check | Result | Status |
|---|---|---|---|
| 40 | Relative documentation links | 42 Markdown files, 97 relative links checked (files and anchors); the only broken targets at QA time were the three documents written after the QA run (this report, TESTING_AND_VALIDATION.md, CLIENT_HANDOVER.md); re-checked at packaging | PASS (re-checked at packaging) |
| 41 | External links | GitHub PR links HTTP 200; GitHub Actions run links verified through the GitHub API (pages return 403 to automated clients) | PASS |
| 42 | Documentation vs behaviour | API_REFERENCE examples are captured responses; INSTALLATION commands marked [Tested] were executed in this QA; corrected: SECURITY.md input limit (512), health-check intervals, pip upgrade step, tools-container status | PASS |
| 43 | GitHub Actions on the QA'd commit | run [44153461847](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/actions/runs/44153461847): research integrity, lint + tests, Docker smoke test: all success | PASS |
| 44 | GitHub Actions on the release commit | run 44088387748 on `27d2701`: success | PASS |
| 45 | ZIP integrity | performed after packaging; results in the delivery folder copy of this report and in `PACKAGE_MANIFEST.md` | see package verification |

## 6. Environment notes (not product defects)

* Docker builds used `--secret id=pip_ca,...` and the local run used the proxy CA because all
  traffic passed through a TLS-inspecting proxy; both are documented options.
* Docker Compose was used with a separate project name (`COMPOSE_PROJECT_NAME=txqa`) so that QA
  volumes did not touch other state.

## 7. Conclusion

All executable checks pass. Outstanding items are external: real-data validation (client data),
GPU re-verification (hardware), and production controls (TLS, authentication, rate limiting). The
package is **delivery-ready**; it is **not production-ready**.
