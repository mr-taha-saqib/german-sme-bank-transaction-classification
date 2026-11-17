# Retraining report 20261116T222145Z

* Approved model at start: `tinybert-seed42-e6eee61ca5`
* Mode: live
* Decision: **promoted**
* Candidate version: `candidate-seed42-d913325f46`
* Approved model after run: `candidate-seed42-d913325f46`
* MLflow run: `41d367c7926d44c39bc6153dec650a7b`
* Training overrides: `{'epochs': 6, 'max_length': 32}`
* Test split used: **no**

## Feedback data

* Input rows: 366 | accepted: 74 | classes covered: 34
* Dropped: missing 2, empty 1, too long 0, unknown label 1, empty after cleaning 1, duplicates 264, conflicting labels 0
* Leakage gate: exact/group-key collisions with validation/test 23, near-duplicates (Jaccard >= 0.6) 0
* Requirement failures: none

## Validation comparison (frozen validation split)

| metric | approved | candidate |
|---|---|---|
| row_accuracy | 0.00877 | 0.0614 |
| row_macro_f1 | 0.00109 | 0.00723 |
| row_weighted_f1 | 0.00015 | 0.0071 |
| row_pct_auto | 0.0 | 0.0 |
| row_accuracy_auto | None | None |
| group_accuracy | 0.0625 | 0.0625 |
| group_macro_f1 | 0.00735 | 0.00735 |
| n_validation_rows | 228 | 228 |
| n_validation_groups | 16 | 16 |

## Promotion criteria

| criterion | passed | detail |
|---|---|---|
| candidate_metrics_present | True | candidate validation metrics were computed |
| row_macro_f1_improves | True | candidate 0.00723 >= approved 0.00109 + 0.0 |
| row_accuracy_not_dropped | True | candidate 0.0614 >= approved 0.00877 - 0.01 |
| automation_coverage_not_dropped | True | candidate 0.0% >= approved 0.0% - 5.0 pp |
