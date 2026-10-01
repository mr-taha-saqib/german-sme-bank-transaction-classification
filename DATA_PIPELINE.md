# Step 2: Dataset Preparation, Preprocessing, Grouping and Leakage-Safe Splitting

> Part of the final Step 2-6 pipeline. Index: [`PROJECT_INDEX.md`](PROJECT_INDEX.md).

Methodological record for the data pipeline. Every figure here is produced by
`prepare_dataset.py` and independently re-checked by `verify_dataset.py`; nothing is
transcribed by hand.

```bash
python prepare_dataset.py          # build + freeze the split
python verify_dataset.py           # prove the five checks from the frozen artefact
python tests/test_pipeline.py      # 66 unit checks on the pipeline logic
```

No model is trained, loaded or downloaded by any of the above.

---

## 1. Selected dataset

Primary dataset: `data/synthetic_b2b_transactions_2000.xlsx`. It is the only file
containing labelled transactions with the full 37-class taxonomy, the 11-class parent
taxonomy and Open-Banking metadata. Every other supplied file is LLM output derived
from it.

The four synthetic CSVs are not combined into the Step-2 dataset. They are
augmentation material for the *training* split only, and are gated on provenance in
Step 3. Mixing them in here would contaminate the evaluation splits with generated
text. `AUDIT.md` §7.1 records the per-file reasoning; in short:

| File | Used in Step 2 | Why |
|---|---|---|
| `synthetic_b2b_transactions_2000.xlsx` | **yes, primary** | only real labelled data |
| `final_synthetic_clean_7033.csv` | no (Step 3 augmentation only) | LLM-generated; has `source_text` provenance |
| `synthetic_fold1_paper_style.csv` | no | generated from a leaky fold (163 source patterns incl. held-out ones) |
| `final_new_synthetic.csv` | no | a 2 310-row subset already inside the 7 033 file |
| `synthetic_transactions_openai.csv` | no | no `source_text`, so leakage cannot be verified mechanically |

---

## 2. Dataset inspection

The workbook is a single-column sheet holding 65 semicolon-delimited fields per row;
all 2 000 rows split into exactly 65 fields, so no field contains an embedded
semicolon. Full census in `results/dataset/column_census.csv`.

| Property | Value |
|---|---|
| Rows | 2 000 |
| Columns | 65, **40 informative, 25 constant** |
| Target `categoryName` | 37 classes |
| Parent `parentCategoryName` | 11 classes |
| Class imbalance | 240 (`IT & Software`) to 9, **26.7×** |
| Distinct raw `description` | 1 444 |
| Rows whose raw description repeats | 556 |
| Descriptions mapping to >1 category | **0**. Labels are perfectly consistent |

Missing values. Only five columns have any, and none is used as a model input:
`duplicateOfTransactionId` (98.4 % missing), `duplicateGroupId` (96.8 %),
`counterpartCreditorId` (84.1 %), `counterpartMandateReference` (84.1 %),
`receiptId` (69.4 %). The three fields the pipeline actually reads (`description`, `categoryName`, `parentCategoryName`) are 100 % populated, so no
imputation or row-dropping for missingness is required.

### 2.1 Metadata that could identify related transactions

This is the part the audit flagged but did not resolve. Candidate linking identifiers:

| Identifier | Distinct | Coverage | Max categories per value | Min values per category |
|---|---|---|---|---|
| `vendorName` / `vendorId` / `counterpartIban` / `counterpartyNameRaw` | 128 | 100 % | 3 | 2 |
| `vendorGroupId` / `vendorGroupName` | 115 | 100 % | 3 | 2 |
| `duplicateGroupId` | 32 | 3.2 % | 1 | 1 |
| `fingerprint` | 1 999 | 100 % | 1 | 9 |
| `counterpartCreditorId` | 286 | 15.9 % | 1 | 35 |
| `bankAccountId` | 3 | 100 % | 37 | 1 |

Two of these turned out to matter.

---

## 3. Duplicate handling: a leak text grouping alone cannot see

`duplicateGroupId` is the source system's own duplicate detection: 64 rows flagged
`duplicate == true`, forming 32 duplicate groups. **26 of those 32 groups link rows
with *different* descriptions**, for example:

```
Software Lizenz 2025-02   /  SaaS Subscription 2025-02      same amount, same category
Werkvertrag ... Schlussrechnung  /  Projektabrechnung ERP-Migration Milestone Final
```

Text grouping would place these in different groups and could send one to train and
its twin to test. Two remedies were measured:

| Remedy | Rows | Groups | Categories with only 1 group |
|---|---|---|---|
| do nothing | 2 000 | 184 | 4 |
| **merge** the linked text groups | 2 000 | 167 | **10** |
| **remove** one row per duplicate group | **1 968** | 184 | **4** |

Merging costs six additional categories their only testable group. Removal costs 32
rows (1.6 %) and leaves group structure and category coverage untouched, while
eliminating every cross-description duplicate link. Removal was chosen. The kept
representative is selected deterministically by `id`, so the result depends on no seed
and no row order.

The single exact `fingerprint` collision is handled by the same rule (it was already
inside a flagged duplicate group, so 0 additional rows are dropped).

Result: 2 000 → 1 968 rows.

---

## 4. Why vendor identity is *not* part of the leakage unit

`vendorGroupId` looks like an attractive leakage unit, it mirrors the reference
paper's per-company split, and every category has at least 2 vendor groups (better
coverage than text, where 4 categories have only 1 group). It was tested and rejected
on evidence:

Text groups and vendors cross-cut heavily. 124 of 184 text groups span more than
one vendor, and 118 of 128 vendors span more than one text group. They do not nest, so
neither is a superset of the other.

Consequently, taking the connected components of *(text-group ↔ vendor)* (the only way to make both safe simultaneously) collapses the dataset:

| Leakage unit | Groups | Largest group | Categories with only 1 group |
|---|---|---|---|
| text only | 184 | 78 rows (4 %) | 4 |
| vendor only | 115 | 81 rows (4 %) | 0 |
| **text ∪ vendor** | **22** | **505 rows (25 %)** | **36 of 37** |

At 22 groups a leakage-safe split is impossible: 36 of 37 categories would have a
single group, so holding one out removes the category from training entirely.

A vendor-only split is equally unusable, for a different reason: because text
groups span vendors, it would place *identical cleaned text* on both sides. The exact
leak this project exists to prevent.

The decisive argument is that there is no leakage channel. The model's only input
is `description`. Vendor name, IBAN and vendor group live in separate columns that are
never fed to it. Vendor identity therefore cannot leak *through the model*, whereas
identical text demonstrably can. Constraining the split by an identifier the model
never sees would destroy the experiment to prevent a leak that cannot occur.

Decision: the leakage unit is the transaction text. Vendor metadata is documented,
retained in the frozen artefact for later analysis, and deliberately unused for
grouping.

---

## 5. Preprocessing

Fourteen numbered, deterministic transformations (`src/preprocessing.py`). The guiding
rule from the brief is *be conservative*: remove only what is provably uninformative.

| Aspect | Decision |
|---|---|
| **Normalisation** | Unicode NFKC (T0), so visually identical characters unify |
| **Case** | Lower-cased (T12), configurable. Chosen for consistency with the already-lower-cased augmentation corpus used in Step 3; the cost is German noun capitalisation |
| **Punctuation** | Replaced with a space (T7), never deleted, so `Shop-Auszahlung` becomes two tokens rather than one fused token |
| **Whitespace** | Collapsed to single spaces, trimmed (T13) |
| **Numbers, dates** | ISO (`2024-06`) and German (`01.06.2024`) forms removed (T2, T3) |
| **Numbers, references** | Invoice/reference codes removed (T5): `RE-2024-8096`, `INV12345`, `ABC-99887`. Order matters. The prefixed form is matched first, otherwise removing `2024-6278` from `RE-2024-6278` strands the orphan token `re` |
| **Numbers, other** | Standalone digit runs of length ≥ 3 removed (T6). **Short numbers are kept**: `Milestone 2` retains its index, which distinguishes real project stages |
| **Numbers, protected** | `B2B`, `B2C`, `365`, `24/7`, `Q1`–`Q4`, `4.0`, `5G`, `3D`, `O2` survive intact (T1) |
| **IBANs** | Removed (T4) before reference matching so they are not partially consumed |
| **German-specific** | Umlauts and `ß` preserved throughout; 18 unambiguous banking abbreviations expanded (T8): `mwst`/`ust` → `umsatzsteuer`, `sozialvers` → `sozialversicherung`, `instandh` → `instandhaltung`, … |
| **Legal forms** (`GmbH`, `AG`) | **Kept.** Removal is implemented (T9) but off by default, the brief asks for conservative normalisation |
| **Domain stop words** (`ref`, `beleg`, months) | **Kept.** Implemented (T10), off by default, same reason |
| **Empty results** | Replaced with the placeholder `nodescription` (T13) and then dropped. On this dataset **0 rows** clean to empty |

### Measured effect

| Metric | Before | After |
|---|---|---|
| Distinct descriptions | 1 427 | **184** (7.8× collapse) |
| Vocabulary (token types) | 1 083 | 175 |
| Mean tokens per description | 3.58 | 2.43 |
| Token types removed | — | 911, overwhelmingly dates and reference codes (`2024-06`, `2026-01`, `RE-2024-…`) |
| Rows cleaning to empty | — | 0 |

Worked examples are in `results/dataset/preprocessing_examples.csv`:

```
Domain und Server 2024-06                      ->  domain und server
Consulting Retainer 2024-06 RE-2024-8096       ->  consulting retainer
Software License Revenue 2024-06 RE-2024-9874  ->  software license revenue
Payroll Sammelüberweisung 2024-06              ->  payroll sammelüberweisung
LinkedIn Ads B2B Leadgen 2024-06               ->  linkedin ads b2b leadgen
```

The last line is the correction to the prototype, which produced
`linkedin ads b b leadgen`: a `b b` bigram then shared by three unrelated categories.

Preprocessing rules were not designed using the test set. They target date and
reference syntax visible in the schema and in the raw text, were fixed before any
split existed, and reference neither the labels nor any split membership. The rule set
is identical for every row.

---

## 6. Grouping

Two deterministic stages, both label-free (`src/grouping.py`):

1. **Canonical key**. The cleaned text is reduced to its sorted set of unique tokens,
   so `miete büro` and `büro miete` collapse together.
2. **Near-duplicate merge**. Distinct keys with token-Jaccard ≥ 0.60 are merged
   by union-find.

The target label is never consulted when forming groups. Label purity is computed
*afterwards* as a diagnostic. A unit test asserts that permuting the label column
leaves the grouping bit-identical.

### Choosing the threshold from the data

`near_duplicate_census` measured the actual similarity structure of the 184 canonical
keys before any threshold was fixed:

| Jaccard ≥ | Pairs | Same category | Cross category |
|---|---|---|---|
| 0.90 / 0.80 / 0.70 | 0 | 0 | 0 |
| 0.65 | 35 | 35 | 0 |
| **0.60** | **61** | **61** | **0** |
| 0.55 | 61 | 61 | 0 |
| 0.50 | 168 | 161 | **7** |

0.60 is the highest threshold that captures real near-duplicates while merging nothing
across a category boundary. 0.55 gives an identical grouping, so 0.60 sits on a
plateau rather than a knife edge. At 0.50 the merge starts fusing genuinely different
categories (`b2b beratung portal projekt` [Beratung] with
`b2b förderung portal projekt zuschussbescheid` [Zuschüsse]), which is why it is
rejected.

The pairs merged at 0.60 are chiefly milestone-numbered variants of one project:

```
1 automation milestone projektabrechnung  ~  3 automation milestone projektabrechnung
1 data milestone pipeline projektabrechnung  ~  data final milestone pipeline projektabrechnung
```

Splitting those across train and test would test the model on a string it has all but
memorised.

Note on the previous threshold. The earlier default of 0.80 merged nothing on this
corpus, because no pair exceeds 0.70. The cross-split audit at 0.80 was reporting
"zero violations" only because it was looking above the data's range. The audit
threshold is now tied to the merge threshold.

### Group identifiers and result

Groups are numbered by sorting the canonical keys lexicographically, so `group_id` is
stable across runs and independent of row order (asserted by test).

| Metric | Value |
|---|---|
| Canonical keys | 184 |
| Pairs merged at 0.60 | 61 |
| **Groups** | **152** |
| Group size min / mean / max | 1 / 12.95 / 78 |
| **Label-impure groups** | **0** |
| Groups per category: median | **2** |
| Categories with 1 group | 4 |
| Categories with < 3 groups | 21 of 37 |

### Limitations

1. Grouping is lexical. `gehaltslauf` and `payroll sammelüberweisung` mean the same
   thing but share no tokens, so they stay separate. Conservative for correctness (it never hides leakage), but it makes the split harder than semantic grouping would.
2. Jaccard over token sets ignores word order and token frequency.
3. Because merging is label-free it *could* fuse two categories; that would be reported
   as an impure group rather than silently prevented. On this corpus it does not occur.
4. **Union-find is transitive, so chains can merge keys that are not pairwise
   similar.** Measured: 12 of 152 groups contain more than one canonical key, and
   only one of them drifts. That group holds 10 keys / 16 rows, all
   `Projektumsätze`, where 25 of its 45 internal pairs sit below the 0.60 threshold:
   `1 automation milestone projektabrechnung` and
   `3 forecasting milestone projektabrechnung` end up together because a chain of
   pairwise-similar milestone variants links them. Every other multi-key group is
   internally consistent (all pairs ≥ 0.667).

   This is over-merging, which makes the split stricter rather than leakier, and
   all merged members share a category. It costs granularity, not safety:
   `Projektumsätze` still contributes 18 groups. Complete-linkage clustering would
   avoid the chaining at the cost of a more complex, less deterministic algorithm;
   single-linkage union-find was kept for reproducibility.
5. Group sizes are very unequal (1–78 rows), so a group-level split cannot also balance
   row counts: visible in §7.

---

## 7. Leakage-safe group-based splitting

Per category, over that category's distinct groups, shuffled with seed 42:

| Groups in category | test | validation | train |
|---|---|---|---|
| ≥ 3 | 1 | 1 | rest |
| 2 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |

Groups are the atomic unit: a group is assigned to exactly one split, never divided.

### Why train/validation/test and not a 4-way split

A separate calibration split was tested and is not affordable: only **13 of 37
categories have ≥ 4 groups**, so a 4-way split would leave 65 % of classes unrepresented
in one or more splits. The validation split therefore also serves as the held-out
calibration set in Step 3. That remains sound because it is disjoint from test, and
with `training.model_selection = "fixed_epochs"` no model-selection decision consumes
it either.

### Result

| Split | Rows | Groups | Categories | Distinct texts |
|---|---|---|---|---|
| train | 1 093 | 103 | 37 | 134 |
| validation | 228 | 16 | 16 | 16 |
| **test** | **647** | **33** | **33** | 34 |

*(test holds 34 distinct texts in 33 groups because one test group contains a merged
near-duplicate pair. The group, not the text, is the leakage unit.)*

Full per-category counts: `results/dataset/class_distribution.csv`.

### Consequences, stated rather than hidden

- **4 categories cannot appear in test**. `Darlehensaufnahmen`,
  `Gesellschaftereinlagen`, `Kapitalerhöhungen`, `Versicherungsleistungen` have a
  single group each and stay in training.
- **All 33 test categories hold exactly one group**, so their per-class recall can only
  be 0.0 or 1.0 and macro-F1 will have very high variance.
- **21 categories are absent from validation**, so calibration in Step 3 will see 16 of
  37 classes. Scalar temperature scaling (1 parameter) tolerates this; a per-class
  vector variant would not.
- **Row counts are unbalanced by construction.** `Löhne & Gehälter` has exactly two
  similar-sized groups (`gehaltslauf`, `payroll sammelüberweisung`), so it splits 78
  train / 78 test. This is the correct behaviour of a group split with unequal group
  sizes, not a bug.

---

## 8. Validation checks

`verify_dataset.py` re-derives every claim from the frozen artefact without importing
the split logic. 19 checks, 0 failures.

| # | Check | Result |
|---|---|---|
| 1 | No group occurs in more than one split | **PASS**, 0 shared `group_id` in all three pairs |
| 2 | No exact duplicate cleaned description across splits | **PASS**, 0 shared texts; every text maps to exactly one split |
| 3 | Near-duplicate leakage measured and reported | **0 same-category pairs** at Jaccard ≥ 0.60 in all three pairs (0 cross-category) |
| 4 | Class distributions reported per split | `class_distribution.csv`, 37 categories × rows and groups per split |
| 5 | Rows and groups per split reported | matches the manifest exactly for all three splits |

Plus artefact integrity: SHA-256 of `(row_uid, group_id, split)` matches the manifest,
`row_uid` is unique, and every row carries a valid split label.

---

## 9. Reproducibility and the frozen split

`random`, `numpy` and (when present) `torch`/CUDA are seeded from `split.seed`
(default 42), with `cudnn.deterministic = True`.

The split is frozen to disk rather than recomputed downstream:

```
results/dataset/
├── split_assignment.csv      canonical: row_uid, description, clean_description,
│                             group_key, group_id, categoryName, parentCategoryName, split
├── split_manifest.json       per-split counts, group→split map, sha256 checksum
├── dataset_summary.json      every figure in this document
├── column_census.csv         all 65 columns: missingness, cardinality, example
├── class_distribution.csv    rows and groups per category per split
├── near_duplicate_census.json
├── grouping_report.json
├── split_report.json
├── preprocessing_audit.json
├── preprocessing_examples.csv
├── config.json               exact configuration used
└── prepare_dataset.log
```

Current checksum: `f715f040d8a5daa6…`

Step 3 must load `split_assignment.csv` and assert this checksum, rather than
regenerating the split. That is what guarantees GermanFinBERT and FinBERT train and
evaluate on byte-identical data. Determinism is additionally covered by unit tests:
the same seed reproduces the same split, and shuffling the input rows leaves the
group→split mapping unchanged.

---

## 10. What the test set has not influenced

- **Preprocessing rules**, fixed from schema and raw-text syntax before any split
  existed; no rule references a label or a split.
- **The grouping threshold**, chosen from the similarity census over all 184 canonical
  keys, a split-independent property of the corpus. Category labels were used only to
  *verify* afterwards that no merge crossed a category boundary.
- **The split itself**, a fixed per-category rule plus one seeded shuffle. No
  performance signal exists at this stage, because no model has been trained.
- **Duplicate removal**, driven by the source system's own flags, not by labels or
  splits.

Model training, hyperparameter selection and calibration fitting are Step 3 and have
not been performed against this frozen split.
