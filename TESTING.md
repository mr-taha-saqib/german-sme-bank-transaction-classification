# Testing

## Suites

| Suite | Command | Count | Needs |
|---|---|---|---|
| Frozen split verification | `python verify_dataset.py` | 19 checks | numpy, pandas |
| Augmentation corpus verification | `python verify_augmentation.py` | 11 checks | numpy, pandas |
| Final results integrity + traceability | `python final_evaluation.py` | 12 checks + 120 recomputations from saved logits | numpy, pandas, scikit-learn |
| Research unit checks | `python tests/test_pipeline.py` | 66 checks (10 pytest functions) | + CPU torch |
| MLOps tests | `pytest` | 49 tests in `tests/test_mlops_*.py` | + transformers, FastAPI, MLflow (no download) |
| **pytest total** | `pytest` | **59** (78 % line coverage of `src/`) | |
| Lint (MLOps layer) | `ruff check .` | – | ruff |

`verify_augmentation.py` and `final_evaluation.py` rewrite tracked result files in
place. On Linux the rewrite only changes line endings (CRLF → LF; content identical, as
verified with `git diff --ignore-cr-at-eol`). Run them in a scratch copy, or
`git checkout -- results/` afterwards, to keep the tree clean.

## The tiny test model

MLOps tests never download GermanFinBERT. `tests/conftest.py` builds a 1-layer BERT
(hidden size 32) with a WordPiece vocabulary from the cleaned training texts and trains
it for 4 epochs on the frozen train split (~20 s on CPU), once per session. It exercises
the real code paths: research preprocessing, tokenisation, `TransactionClassifier`,
packaging, checksums, the API, monitoring and retraining. Its predictions are
meaningless and it is never presented as the production model.

## What the MLOps tests cover

* **API** (`test_mlops_api.py`, 21): health/info, prediction schema, routing rule,
  API = direct classifier output, 7 invalid-input cases, no echo of rejected input,
  malformed JSON, batch with per-item errors and order preserved, batch limits, feedback
  flow (unknown category 422, unknown id 404), metrics, monitoring log contains no
  description text, 503 without a model, corrupted weights refused, reload auth, OpenAPI.
* **Components** (`test_mlops_components.py`, 10): package contents and metadata,
  incomplete runs rejected, idempotent packaging, registry rollback/rejection, numeric-only
  text features, PSI, controlled input drift detected and not reported as performance,
  delayed labels → accuracy, config overrides, reference profile.
* **Retraining** (`test_mlops_retrain.py`, 18): every gate criterion can block; feedback
  validation incl. leakage gate (no accepted text equals a held-out text); conflicting
  labels; missing columns; data-requirement rejection; validate-only; live retraining
  with **`evaluate_test=False` asserted**; previous version preserved; dry run changes
  nothing; MLflow run contents; back-to-back runs keep separate reports; rollback CLI
  exits cleanly without history.

## Weak-test fix in the research suite

The research `test_*` functions only appended to a list of failures, so under pytest
they always passed. Demonstration before the fix: forcing one check to fail gave
"10 passed". After the fix the same change gives "1 failed, 9 passed", and the
standalone runner still lists every check. No check was removed or loosened.

## CI

`.github/workflows/ci.yml` runs three jobs: research integrity without a deep-learning
stack, lint + pytest on CPU, Docker build + container inference with the tiny model.

## Session-2 audit (2026-11-17)

Re-run of all suites on the final branch: `docs/evidence/10_session2_acceptance_audit.log`
(19 + 11 + 66 research checks, 12/12 integrity, 120/120 recomputations, regenerated
results content-identical to the committed files, 59 pytest, ruff clean, CLI 5/10
automatic) and `docs/evidence/11_session2_docker_audit.log` (container health, inference,
422 on empty input, drift alert, restart, uid 10001).
