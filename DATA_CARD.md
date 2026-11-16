# Data card

Full detail: `DATA_PIPELINE.md`, `AUGMENTATION_DECISION.md`, `AUDIT.md`. This card
summarises what the deployed model was trained and evaluated on.

| | |
|---|---|
| Primary data | `data/synthetic_b2b_transactions_2000.xlsx`: 2,000 **synthetic**, template-generated German B2B transactions, 37 categories, 11 parent categories |
| After de-duplication | 1,968 rows (32 rows flagged as duplicates by the source system removed) |
| Distinct cleaned texts | 184, merged into 152 label-free groups (token-Jaccard ≥ 0.60) |
| Frozen split | train 1,093 rows / 103 groups · validation 228 / 16 · test 647 / 33; SHA-256 `f715f040d8a5daa6…` |
| Augmentation | corpus v2: 4,541 LLM-generated rows passing gates A (provenance), B (collision), C (near-duplicate); SHA-256 `4eb242d9f384041f…` |
| Training corpus | 1,093 real + 4,541 synthetic = 5,634 rows |
| Language | German; umlauts and ß preserved; lower-cased |
| Personal data | none known (synthetic); workbook document metadata (author, path) removed before publication |

## Preprocessing

14 deterministic transformations (`src/preprocessing.py`): dates, invoice/reference
codes, IBAN-like tokens and long digit runs removed; banking abbreviations expanded;
vendor names, legal forms and protected tokens (B2B, Q1, 24/7 ...) kept.

## Leakage protection

Groups never straddle splits; 0 shared groups, 0 shared texts and 0 same-label
near-duplicates across splits (`verify_dataset.py`, 19 checks). Augmentation rows
whose source pattern is held out, or which equal / near-duplicate a held-out
pattern, are excluded (`verify_augmentation.py`, 11 checks). The retraining workflow
applies the same gates B and C to new labelled data.

## Limitations

Synthetic, template-like text (2,000 rows → 184 distinct texts); real SME
descriptions are noisier. The test set has 33 independent groups (one group = 3.03
accuracy points). 4 categories cannot appear in test; 8 have no synthetic support.

## New data for retraining

CSV `description,categoryName` with labels from the 37-class taxonomy, confirmed by
people (e.g. manual-review outcomes). The API never stores descriptions, so this data
comes from the client's own bookkeeping. All retraining demonstrations in this
repository used **synthetic** rows (`scripts/generate_synthetic_feedback.py`).
