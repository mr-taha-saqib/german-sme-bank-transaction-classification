# Retraining and model promotion

`src/retrain.py` (with `src/candidate_training.py`). Executed scenarios and reports:
`docs/evidence/05_retraining_e2e.log`, `docs/evidence/06_retraining_promotion_tiny_model.log`,
`docs/evidence/retraining_reports/`.

## Interface

CSV, UTF-8, columns `description,categoryName`: **new, human-confirmed** transactions
(for example the outcome of manual reviews). Labels must belong to the 37-class
taxonomy of the approved model.

```bash
python -m src.retrain --feedback confirmed.csv --validate-only     # data checks only, seconds
python -m src.retrain --feedback confirmed.csv --dry-run           # train + evaluate, registry untouched
python -m src.retrain --feedback confirmed.csv                     # promote only if the gate passes
python -m src.retrain --feedback confirmed.csv --force             # re-run already-processed feedback
python -m src.retrain --rollback --reason "..."                    # re-approve the previous model
# optional: --base-model <hf-id or local dir>, --epochs N, --max-length L (recorded as overrides)
```

## Workflow

1. Load the approved version; abort if none.
2. Load the frozen split (checksum verified) and check it equals the one the approved
   model was trained on.
3. Validate rows: required columns; missing, non-string, empty, over-long (> 512 chars)
   text; unknown labels; empty after the research preprocessing; duplicates; the same
   cleaned text with conflicting labels. Every drop is counted in the report.
4. **Leakage protection** with the research gates: rows whose cleaned text equals, shares
   a group key with, or reaches token-Jaccard ≥ 0.60 against any **validation or test**
   pattern are dropped (`src/splitting.filter_synthetic_for_train`, gates B and C).
5. Minimum requirements: ≥ 30 accepted rows covering ≥ 3 classes (`configs/mlops.yaml`).
6. Candidate: fine-tune the base checkpoint with the approved run's configuration on
   frozen train + corpus v2 + accepted rows (`train_model.py` code: same loss, schedule,
   seeding). The **test split is never evaluated** (`evaluate_test=False`; asserted in
   `tests/test_mlops_retrain.py`).
7. Gate on the frozen **validation** split, against the approved model's validation
   metrics produced by the same code:
   * row-level macro-F1 ≥ approved + `min_improvement` (0.0);
   * row-level accuracy not lower by more than 0.01;
   * share of automatic decisions (≥ 0.80) not lower by more than 5 pp;
   * accuracy among automatic decisions not lower by more than 0.02 (when the approved
     model automates anything).
8. Candidate registered either way, MLflow run logged (params, metrics, approved
   metrics, gate result), decision recorded in `registry.json` with its evidence.
9. Promotion only if every criterion passes and not `--dry-run`. The previous version
   stays available; `--rollback` restores it.
10. Report `reports/retraining/retraining-<timestamp>.{md,json}`.

Idempotence: (approved version, accepted-data fingerprint, overrides, base model) is
recorded; the same input is not retrained twice unless `--force`.

## Caveats

* The validation split has 16 independent groups covering 16 of 37 classes, and it was
  also used to fit the temperature-scaling experiment. Promotion decisions on it are
  noisy; treat them as a safety gate, and re-run the full research evaluation
  (`final_evaluation.py`) before relying on a promoted model's reported quality.
* CPU training of GermanFinBERT takes ≈ 2 h (3 epochs, max_length 128, 2 cores); a GPU is
  strongly recommended. `--max-length 32` is lossless for this data (longest text 19
  tokens) and ≈ 4× faster.
* **All demonstrations used synthetic feedback.** Production retraining needs real,
  correctly labelled transactions.

## Executed scenarios

On an **isolated copy** of the registry (the delivered `model_registry/` was not
modified), with **synthetic** feedback (`scripts/generate_synthetic_feedback.py`):

| # | Input | Outcome |
|---|---|---|
| 1 | 6 rows, all invalid | `rejected_data_requirements`; nothing trained |
| 2 | 366 rows (training variants + held-out leaks + invalid), `--validate-only` | 74 accepted (34 classes). Dropped: 2 missing, 1 empty, 1 unknown label, 1 empty after cleaning, 264 duplicates, **23 held-out collisions caught by the leakage gate** |
| 3 | same, live; real GermanFinBERT base, overrides `epochs=1`, `max_length=32` (10 min 53 s on CPU) | candidate `candidate-seed42-bdf512ef87` **rejected**: equal validation accuracy (0.49123) and macro-F1 (0.68992), but 0 % automatic decisions vs 28.5 % (under-confident after one epoch); MLflow run `024ef897f1d542a0b6f29a0ef031a50b` |
| 4 | same file again | idempotent: earlier decision returned, nothing retrained |
| 5 | rollback with no earlier approved version | clean error, exit code 1 |
| 6 | **tiny test model** (not GermanFinBERT): approved = 1 epoch, candidate = 6 epochs | candidate **promoted** (all criteria passed), previous version kept, rollback restored it; MLflow run `41d367c7926d44c39bc6153dec650a7b` |

Scenario 6 uses a deliberately weak tiny model so that the promotion path can be shown
in minutes on a CPU; its metrics are meaningless by design.
