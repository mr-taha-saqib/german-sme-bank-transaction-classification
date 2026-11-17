# Changelog

## Post-release delivery additions to v1.1.0-mlops (2026-11-17, branch `client-delivery`)

Not a new model or API version; the release tag `v1.1.0-mlops` is unchanged.

### Fixed
* `python -m src.retrain --rollback` now moves the MLflow `champion` alias back to the
  rolled-back registry version (`Tracker.mirror_alias`); previously the alias kept pointing at the
  rolled-back-from version while the file registry was correct. Test added (60 tests).
* `SECURITY.md`: input limit stated as 1–512 characters (`api.max_text_length`); the schema's
  2,000 is only an outer cap.
* `CLIENT_DELIVERY_GUIDE.md`: install command referred to the merged delivery branch.

### Added (documentation and demo only)
* `CLIENT_README.md`, `TECHNICAL_DOCUMENTATION.md`, `INSTALLATION_AND_SETUP.md`, `USER_GUIDE.md`,
  `API_REFERENCE.md`, `TESTING_AND_VALIDATION.md`, `SECURITY_AND_LIMITATIONS.md`,
  `CLIENT_HANDOVER.md`, `LICENSE_AND_DATA_ATTRIBUTION.md`, `FINAL_DELIVERY_QA_REPORT.md`.
* `examples/demo/` synthetic API demo with a captured run; `docs/openapi.json`.

### Unchanged
* Research pipeline, data, split, corpus, seeds, threshold, model, API behaviour and all results.

## 1.1.0 additions (2026-11-17, session 3, PR #3)

### Security (see SECURITY.md)
* Constant-time admin-token comparison for `/model/reload`.
* API image upgrades pip 26.3 / setuptools 84.2.1 (base-image versions had advisories).
* `requirements.txt`: pillow 12.3.0 → 13.0.0 (advisories; plotting only; research checks re-run,
  results unchanged).
* Compose: API and MLflow published on 127.0.0.1 by default (`API_BIND`), `cap_drop: ALL`,
  `no-new-privileges`.

### Added
* `SECURITY.md`, `docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md`.

### Unchanged
* Research pipeline, data, frozen split, corpus v2, seeds, threshold, published results, model.

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
