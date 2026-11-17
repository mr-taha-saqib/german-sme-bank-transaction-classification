# Changelog

## 1.1.0 (2026-11-17), branch `mlops-production-delivery`

MLOps layer added around the unchanged research pipeline. No research result,
dataset, frozen split, augmentation corpus, seed, threshold or published metric was
modified.

### Added
* `src/api.py`: FastAPI service (`/health`, `/model/info`, `/predict`,
  `/predict/batch`, `/feedback`, `/metrics`, `/monitoring/summary`, `/model/reload`),
  0.8 routing, top-3 alternatives, parent category, per-item batch errors, structured
  JSON logs without request bodies.
* `src/model_artifact.py` + `src/registry.py`: packaging of training runs into a
  versioned registry (safetensors weights, checksums, metadata, monitoring reference),
  explicit approval, rejection records, rollback.
* `src/tracking.py`, `scripts/import_research_runs_to_mlflow.py`: MLflow tracking on
  local SQLite; the 10 published research runs imported read-only for comparison.
* `src/monitoring.py`: Prometheus metrics, PSI input and prediction drift,
  delayed-label accuracy, privacy-preserving event log.
* `src/candidate_training.py`, `src/retrain.py`: gated retraining that reuses the
  research training and evaluation code, applies the research leakage gates B and C to
  new data, compares on the validation split only, supports dry-run, validate-only,
  idempotence and rollback.
* `configs/mlops.yaml`, `.env.example`, `Dockerfile` (API + tools targets),
  `docker-compose.yml`, `.github/workflows/ci.yml`, `pyproject.toml`.
* Tests: `tests/test_mlops_*.py` (API, components, retraining) with a tiny offline BERT.
* Documents: ARCHITECTURE, API_DOCUMENTATION, MODEL_CARD, DATA_CARD, DEPLOYMENT,
  MONITORING, RETRAINING, TESTING, CONTRIBUTING, DELIVERY_SUMMARY, FINAL_E2E_ACCEPTANCE.

### Changed
* `src/inference.py`: new `predict_proba` method; `predict` now calls it (same output).
* `FINAL_E2E_ACCEPTANCE.md` (research acceptance) renamed to
  `FINAL_E2E_ACCEPTANCE_RESEARCH.md`, content unchanged.
* README: Python requirement corrected to 3.12+ (the pinned `numpy==2.5.2` does not
  install on 3.10/3.11); MLOps sections added; "no API" statements updated.

### Fixed
* `tests/test_pipeline.py`: test functions only *recorded* failed checks, so under
  `pytest` every function passed even when a check failed (demonstrated by breaking a
  check: pytest reported "10 passed"). Each function now fails on its own failed checks;
  the standalone runner still reports all 66 checks.

## 1.0.0
Academic proof of concept: see `FINAL_REPORT.md` and `FINAL_E2E_ACCEPTANCE_RESEARCH.md`.
