# Testing and validation

Release `v1.1.0-mlops`. What is tested, how to run it, and the results of the final delivery audit
([FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md)). Detailed suite inventory:
[TESTING.md](TESTING.md).

## 1. Environment requirements

Python 3.12, CPU PyTorch (`--index-url https://download.pytorch.org/whl/cpu torch==2.13.0`),
`pip install -r requirements-dev.txt -r requirements.txt`. No internet access, GPU or model download
is needed for the tests: they build a tiny BERT offline. Docker checks need Docker 24+.

## 2. Commands

```bash
ruff check .                                     # lint (scope in pyproject.toml: MLOps layer, tests, scripts)
pytest -q                                        # 60 tests
pytest -q --cov=src                              # with coverage
pytest -q tests/test_mlops_api.py                # API only
python tests/test_pipeline.py                    # 66 research pipeline checks
python verify_dataset.py                         # 19 frozen-split checks
python verify_augmentation.py                    # 11 augmentation-corpus checks
python final_evaluation.py                       # 12 integrity checks + 120 recomputations from saved logits
./examples/demo/run_demo.sh                      # end-to-end API walk-through (running API needed)
```

`final_evaluation.py` and the verify scripts rewrite files under `results/`; restore them with
`git checkout -- results/` (only line endings and one known tie-break cell change).

## 3. Test suites

| Suite | Tests | Covers |
|---|---|---|
| `tests/test_mlops_components.py` (unit) | 10 | packaging (contents, metadata, incomplete runs, idempotence), registry rollback and rejection, monitoring (numeric-only features, PSI, drift, delayed labels), config overrides, reference profile |
| `tests/test_mlops_api.py` (API) | 21 | every endpoint, validation and error format, batch per-item errors, no input echo, reload auth (missing / wrong / near-miss token), 503 without model |
| `tests/test_mlops_retrain.py` (integration) | 19 | retraining with a tiny BERT: data validation, leakage gates, validation-only gate, promotion, rejection, dry run, rollback CLI, MLflow tracking, MLflow alias after rollback |
| `tests/test_pipeline.py` (research) | 66 checks as a script; 10 grouped tests under pytest | preprocessing, grouping, split, augmentation gates, training/evaluation utilities |
| research verification scripts | 19 + 11 + 12 + 120 | frozen split, corpus v2, results integrity, metric recomputation |

Coverage (measured in the final QA): **79 %** of `src/` lines with `pytest --cov=src`. The research
modules exercised only by the research scripts lower this number; the MLOps modules are covered
by the pytest suites.

## 4. Continuous integration

`.github/workflows/ci.yml` (pull requests, pushes to `main`, manual): research integrity (no
deep-learning stack), lint + tests (CPU, tiny model), Docker build + container smoke test.
Final runs: delivery commit `4b1f66d` → run 44153461847 (all three jobs success); release commit
`27d2701` → run 44088387748 (success).

## 5. Docker validation (final QA)

API image built (1.78 GB); container healthy with the real model; inference correct; invalid inputs
422; non-root, capabilities dropped, read-only registry, localhost-only port; state persisted across
restart; no description text in the event log; drift alert on drifted traffic; tools image runs the
dataset checks; MLflow container starts.

## 6. Model evaluation

Frozen test split (synthetic data; 33 independent groups, 647 rows), evaluated once:

| Metric | Delivered CPU model | Published research (5 seeds, GPU) |
|---|---|---|
| Group accuracy | 0.6970 (23/33) | 0.6485 ± 0.0271 |
| Group macro-F1 | 0.5880 | 0.5442 ± 0.0408 |
| Automatic share (≥ 0.80) / accuracy among automatic (groups) | 60.6 % / 0.800 | 60.61 % / 0.8022 |
| Row accuracy | 0.5317 | – |

These are research results on synthetic data and **not evidence of production accuracy**.
Confidence values are uncalibrated. Full analysis: MODEL_CARD.md, FINAL_REPORT.md.

## 7. Actual results of the final audit

| Area | Result |
|---|---|
| Install (clean venv, documented commands) | PASS: 121 distributions, `pip check` clean |
| Lint | PASS |
| pytest | **60 passed** (10 + 21 + 19 + 10), coverage 79 % |
| Research checks | 66/66, 19/19, 11/11, 12/12 + 120/120: PASS |
| Results reproducibility | identical except the documented tie-break field |
| Dependency audit | no known vulnerabilities |
| API (local, demo model; Docker, real model) | PASS |
| Retraining validation, rollback error handling, MLflow server | PASS |
| Docker build/run/hardening/persistence/privacy/drift/tools/MLflow | PASS |
| Real-model training and retraining (≈ 2 h each) | NOT RUN in the final audit (executed earlier, evidence 02, 05, 06) |
| GPU | BLOCKED (no hardware) |

## 8. Known test limitations

* Tests use a tiny BERT, not GermanFinBERT; real-model behaviour is checked through the Docker run
  and the recorded training run.
* No test data from real banking transactions exists; all evaluation is on synthetic data.
* No load or performance testing; latency figures in the monitoring summary are indicative only.
* No security penetration testing; security checks are configuration and behaviour checks.
* The modal-prediction tie-break in the research error analysis is not deterministic across
  `PYTHONHASHSEED` values (documented; metrics unaffected).
