# Final end-to-end acceptance report: MLOps layer

| | |
|---|---|
| Repository | https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification |
| Branch | `mlops-production-delivery` (from `main` @ `0f95563`) |
| Environment | Linux x86_64, 2 CPU cores, 8 GB RAM, **no GPU**, Python 3.12.3, torch 2.13.0+cpu, transformers 5.16.1, Docker 29.9.3 |
| Date | 2026-11-17 |
| Approved model | `germanfinbert-seed42-40de860119` (MLflow run `6e5f29aafc344bc5bf2386b2cb2c0de2`, model version 1, alias `champion`) |
| Research acceptance | `FINAL_E2E_ACCEPTANCE_RESEARCH.md` (original report, unchanged) |

Every row was executed and its result observed; evidence is in `docs/evidence/`.
The research data, frozen split, corpus v2, seeds, 0.8 threshold and every published
result are unchanged (verified at the end of the run, section 3).

## 1. Baseline before changes

| Item | Observed |
|---|---|
| Commit | `0f95563` on `main` |
| Research suites | `verify_dataset.py` 19/19, `verify_augmentation.py` 11/11, `tests/test_pipeline.py` 66/66, `final_evaluation.py` 12/12 integrity + 120/120 traceability; regenerated finals content-identical to committed ones |
| Defects found | (1) research `test_*` functions never fail under pytest (a forced failing check still gave "10 passed"); (2) README "Python 3.10+" but pinned `numpy==2.5.2` needs ≥ 3.12; (3) `verify_augmentation.py` / `final_evaluation.py` rewrite tracked files with LF on Linux (content identical; documented); (4) no API, registry, MLflow, monitoring, retraining, Docker or CI |
| Model weights | not in the repository (by design, ≈ 420 MB per run) |

## 2. Acceptance table (25 mandated steps)

| # | Requirement | Test executed | Result | Evidence | Status |
|---|---|---|---|---|---|
| 1 | Dataset loading | `prepare_dataset.py --out /tmp/tx_regen` from the raw workbook | 2,000 rows loaded, 32 source-flagged duplicates removed → 1,968 | `01_data_pipeline_regeneration.log` | PASS |
| 2 | Dataset validation | `verify_dataset.py --dataset /tmp/tx_regen` | 19/19 checks | same | PASS |
| 3 | Preprocessing | same regeneration | 14 transformations; output identical to the frozen artefact | same | PASS |
| 4 | Leakage + split integrity | regeneration + checksum comparison | split SHA-256 `f715f040…` and corpus v2 `4eb242d9…` regenerated **bit-identically**; 0 group/text/near-duplicate overlap | same | PASS |
| 5 | Model training | `train_model.py --model germanfinbert --augment --augmentation-version v2 --seed 42` (unmodified research script) on CPU | 3 epochs, 6,818 s; test group acc 0.6970, macro-F1 0.5880, ≥0.8 coverage 60.6 % (acc 0.800); validation identical to the published GPU run | `02_model_training_and_registration.log` | PASS |
| 6 | Artifact saving | `python -m src.model_artifact --run … --approve --reason …` | registry version with safetensors weights, metadata, checksums, reference profile | same | PASS |
| 7 | Artifact reloading | API startup, `TransactionClassifier`, checksum verification | loads; corrupted weights refused (503, test) | `03_api_curl_examples.txt`, `test_corrupted_weights_are_not_served` | PASS |
| 8 | Inference | `/predict`, research CLI on `reference/example_new_transactions.csv` | real predictions; CLI: 5/10 automatic (same count as the research acceptance with GPU weights) | `03_api_curl_examples.txt` | PASS |
| 9 | Confidence routing | API responses | ≥ 0.80 → `automatic` (e.g. "Miete Büro Juni 2026" 0.859), else `manual_review` (e.g. "Gehaltslauf 2026-01" 0.188) | same | PASS |
| 10 | Batch inference | `/predict/batch` with an invalid item | 3 items → 2 ok, 1 error ("description is empty or whitespace"), order preserved | same | PASS |
| 11 | MLflow logging | `scripts/import_research_runs_to_mlflow.py` + run logging | 10 research runs imported read-only; delivered model run `6e5f29aa…` with params, metrics, per-epoch history, checksums; model version 1 alias `champion` | `02_…log`, `mlflow.db` | PASS |
| 12 | FastAPI startup | `uvicorn src.api:app` | healthy, model loaded | `03_api_curl_examples.txt` | PASS |
| 13 | Valid API prediction | curl `/predict` | 200, full schema | same | PASS |
| 14 | Invalid API request | empty, wrong field, malformed JSON, unknown prediction id | 422 / 422 / 422 / 404, input never echoed | same + 21 API tests | PASS |
| 15 | Prediction logging | event log inspection | 600 events, 0 fields outside the allow-list, no description words | `04_monitoring_drift_test.log` | PASS |
| 16 | Monitoring execution | live API + `simulate_traffic.py` | decisions, confidence, drift, delayed-label accuracy (0.54 overall, 0.83 automatic, 100 labels) | same | PASS |
| 17 | Controlled drift test | 200 held-out vs 200 English free-text descriptions | input PSI 0.007–0.018 `ok` → 6.8–17.1 `alert`; manual review 64.5 % → 98 % | same | PASS |
| 18 | Retraining trigger | `python -m src.retrain` on a registry copy, synthetic feedback | data-requirement rejection, validate-only, live run, idempotent re-run | `05_retraining_e2e.log` | PASS |
| 19 | Candidate evaluation | live GermanFinBERT candidate (1 epoch, max_length 32) | evaluated on validation only (`test_split_used: false`) | `retraining_reports/` | PASS |
| 20 | Promotion or rejection | gate | real candidate **rejected** (0 % automation vs 28.5 %); tiny-model scenario **promoted** | `05_…`, `06_retraining_promotion_tiny_model.log` | PASS |
| 21 | Previous-model preservation | registry state + rollback | approved model unchanged after rejection; after promotion the previous version was kept and restored by rollback | same | PASS |
| 22 | Docker image build | `docker build --target base` | `german-sme-txclf-api:1.0.0`, 1.77 GB (CPU PyTorch) | `07_docker_api_session.log` | PASS |
| 23 | Docker container startup | `docker compose up -d api` | `(healthy)` after ≈ 15 s, uid 10001, registry read-only, 296 MiB | same | PASS |
| 24 | Inference inside Docker | curl to the container | real predictions, batch per-item errors, 422; state kept across restart | same | PASS |
| 25 | GitHub Actions CI | push to `mlops-production-delivery` | run [43969620094](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/actions/runs/43969620094) and final run [43981230434](https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification/actions/runs/43981230434): research integrity ✔ (19/11 checks, 12/12 + 120/120), lint + pytest ✔ (all passed: 57 tests in the first run, 59 in the final run), Docker build + in-container inference (tiny model) ✔ | Actions | PASS |

Additional checks:

| Requirement | Result | Status |
|---|---|---|
| Automated tests | `pytest`: **59 passed** (10 research functions = 66 checks, 49 MLOps); coverage 78 % | PASS |
| Lint | `ruff check .` clean | PASS |
| Security | `git grep` for keys/tokens/private keys: 0; no `.env`, weights, caches or local DBs committed; descriptions never logged | PASS |
| Documentation | README + 12 documents; commands in README executed in this run | PASS |
| GPU training | not available in this environment; CPU training verified instead (≈ 2 h/run) | BLOCKED (hardware), CPU path PASS |

## 3. Research results unchanged (verified after all changes)

`verify_dataset.py` 19/19 · `verify_augmentation.py` 11/11 · `final_evaluation.py`
12/12 integrity, 120/120 recomputations · regenerated `final_results.csv`,
`final_summary.csv` and `augmentation_verification.json` content-identical to the
committed files · `tests/test_pipeline.py` 66/66 (`docs/evidence/08_final_verification.log`).

## 4. Reproduction honesty

The delivered model is a CPU (fp32) re-run of the published seed-42 configuration.
Validation metrics match the GPU run exactly; on test, 557 of 647 row predictions (86 %)
are identical, and group accuracy is 23/33 vs 22/33. Mixed-precision GPU and fp32 CPU
arithmetic diverge during fine-tuning. The published five-seed result
(0.6485 ± 0.0271) remains the authoritative research figure.

## 5. Problems found during acceptance and how they were resolved

1. Research tests could not fail under pytest → `@collect_checks`; demonstrated before and after.
2. Monitoring reference initially the validation split (16 classes), then true labels →
   false drift alarms on realistic traffic; replaced by held-out predictions/texts; re-run clean.
3. My own first privacy check flagged category names (logged by design) → replaced by a
   strict field allow-list check: 0 violations.
4. Retraining reports of two runs in the same second overwrote each other → microsecond
   timestamps + test; rollback without history raised a traceback → clean exit code 1 + test.
5. Ruff reorganised imports and CRLF bytes of the research test file → reverted so the
   diff contains only the assertion fix.
6. Docker builds behind the sandbox's TLS-inspecting proxy → optional `pip_ca` build secret.

## 6. Locations

Model: `model_registry/versions/germanfinbert-seed42-40de860119/` (git-ignored; reproduce
per DEPLOYMENT.md) · run directory: `artifacts/runs/germanfinbert_seed42_cpu/` · MLflow:
`mlflow.db` + `mlartifacts/` · monitoring: `monitoring/events.jsonl` · evidence:
`docs/evidence/` · image: `german-sme-txclf-api:1.0.0`.

## 7. Known limitations

Synthetic training data; 33 test groups; ~1 in 5 automatic decisions wrong; uncalibrated
confidences; small validation split for promotion decisions; CPU training ≈ 2 h; no
API authentication or cloud deployment; retraining demonstrated with synthetic data only.

## 8. Decision

All 25 mandated end-to-end steps passed in this environment (GPU training not
available; the CPU path was verified). The project is ready for client **hand-over as a
reproducible, tested, containerised reference implementation**. It is not a production
banking system and must be validated on real transaction data before any business use.

## 9. Follow-up verification, session 2 (2026-11-17)

Re-inspected the remote state first (no assumptions carried over from session 1).

| Item | Result | Evidence | Status |
|---|---|---|---|
| GPU training (GermanFinBERT seed 42) | Not run. The delivery container has no GPU (`nvidia-smi` absent, `torch.cuda.is_available()` = False, CPU-only PyTorch wheel). The only other reachable machine is the owner's Apple Silicon MacBook Pro (darwin/arm64): no NVIDIA GPU, no CUDA. The CPU-trained model and the published five-seed GPU result are unchanged | this section | BLOCKED (hardware) |
| Research integrity re-run | 19/19 dataset, 11/11 augmentation, 12/12 integrity, 120/120 recomputations, 66/66 `test_pipeline.py`; regenerated split, corpus and result files content-identical to the committed ones | `docs/evidence/10_session2_acceptance_audit.log` | PASS |
| pytest / ruff / CLI | 59 passed; ruff clean; CLI 5/10 automatic at 0.80 | same | PASS |
| Docker | rebuilt image `ac088501aaae` (1.77 GB); healthy; `Betriebshaftpflicht 2026-07` → Versicherungen 0.92332 automatic; empty input → 422; normal traffic input drift ok, drift traffic alert; healthy after restart; uid 10001 | `docs/evidence/11_session2_docker_audit.log` | PASS |
| Documentation defect | MLOps documents and one monitoring source string used "real" for the original template-generated workbook rows, which could be read as real banking data. Reworded to "original"; terminology note added to README and DATA_CARD; research documents unchanged | git diff | FIXED |
| Evidence logs missing from git | `.gitignore` excluded `*.log`, so the session-1 evidence logs cited in this report (`docs/evidence/01`–`08_*.log`) were never committed. Added `!docs/evidence/*.log`; logs checked for credentials (none) and committed unchanged | `.gitignore`, `docs/evidence/` | FIXED |
| Release tag + GitHub release | Tag push refused by the delivery session's Git proxy with the exact message `ERR push contains a ref outside refs/heads/*; only branch updates are permitted.` (diagnosed with `GIT_TRACE_CURL`; one attempt, not retried). The `gh` CLI has no usable token in this session. Creating the tag through the REST API was **not** attempted because it would bypass that policy. Release notes are ready in `docs/releases/v1.1.0-mlops.md`; publish commands are in `CLIENT_DELIVERY_GUIDE.md` §15 | this section | BLOCKED (permission: the session may update branches only, not tags or releases) |
| PR #1 review | diff vs `main` re-reviewed: no secrets, no weights, no databases, no virtual environments or caches; original research / historical files byte-identical where they must be | `git diff origin/main` | PASS; PR left **open and unmerged** pending owner approval |
| Commercial production readiness | not claimed | – | n/a |

Outstanding external decisions (not completable by the delivery team): owner approval to
merge PR #1; publishing the tag and release; GPU re-verification (optional); authorised, labelled real transaction data for validation before business use; client acceptance.
