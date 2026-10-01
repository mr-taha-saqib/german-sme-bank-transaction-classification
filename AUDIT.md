# AUDIT: German SME Transaction Classification Prototype

Audit date: 2026-08-31
Auditor scope: Phase 1 of the agreed engagement (audit only; no changes to the original notebook).
Artefacts audited:

| Artefact | Path | Size |
|---|---|---|
| Experimental notebook (144 cells) | `notebooks/finbert_transaction_classification_sanitized.ipynb` | 288 KB |
| Primary raw dataset | `data/synthetic_b2b_transactions_2000.xlsx` | 482 KB |
| LLM synthetic data (final) | `data/final_synthetic_clean_7033.csv` | 535 KB |
| LLM synthetic data (increment) | `data/final_new_synthetic.csv` | 161 KB |
| LLM synthetic data (fold-1 run) | `data/synthetic_fold1_paper_style.csv` | 538 KB |
| LLM synthetic data (first run) | `data/synthetic_transactions_openai.csv` | 22 KB |
| Prior results workbook | `reference/Ergebnisuebersicht_FinBERT.xlsx` | 10 KB |
| Reference paper | `ErstesPaper.pdf` (Aluffi et al., 2025, arXiv:2508.05425) | 1.4 MB |

The original notebook has not been modified or deleted. It is preserved verbatim in `notebooks/`.

---

## 1. Current architecture

The prototype is a single linear Jupyter notebook of 144 cells with no module structure, no functions
beyond a handful of inline definitions, and no persisted artefacts other than two CSV/XLSX exports.
All state is carried in global variables that are redefined repeatedly. Cell outputs were stripped
before handover, so no executed result in the notebook can be independently verified. The only
surviving numbers are those hard-coded by hand into cell 58 and exported to
`Ergebnisuebersicht_FinBERT.xlsx`.

The notebook contains at least six overlapping experiment generations, each redefining the same
concepts:

| Gen | Cells | What it does | Split used |
|---|---|---|---|
| V1 | 0–11 | Load XLSX, random stratified split, fine-tune `ProsusAI/finbert` | Random (row-level) |
| V2 | 12–40 | Add `clean_description`, exact-match grouping, `GroupShuffleSplit`, rule-based augmentation, retrain | Group split |
| V3 | 41–52 | Corrected focal loss, TF-IDF baseline, seeded rerun, best-checkpoint selection | Group split |
| V4 | 53–58 | First GermanFinBERT (`scherrmann/GermanFinBert_FP`) run + Excel export | Group split |
| V5 | 59–99 | Azure OpenAI (GPT) augmentation to ≥15 patterns/class, FinBERT + GermanFinBERT, temperature scaling, confidence thresholds | Group split (cell 22) |
| V6 | 100–143 | "Paper-style" run: `StratifiedKFold` on rows, inverse-frequency LLM augmentation, train on synthetic only, temperature+bias calibration, then a final "unseen-pattern" holdout | **Two different splits, one leaky** |

There is no single entry point, no seed discipline across generations (the seed is set once at
cell 51, after three models have already been trained), and no saved model checkpoints. Re-running
the notebook top-to-bottom would not reproduce the reported numbers, and several cells cannot run at
all without an interactive Azure OpenAI credential prompt (cells 63–70).

### Model loading (current)

* **FinBERT (comparison model)**. `ProsusAI/finbert`, loaded seven separate times
  (cells 5, 35, 45, 48, 52, 78, 85, 115, 138) with `num_labels=37, ignore_mismatched_sizes=True`.
  This is an English financial-sentiment model; its WordPiece vocabulary has no German subword
  coverage, so German tokens fragment heavily.
* **GermanFinBERT (primary model)**. `scherrmann/GermanFinBert_FP`, loaded at cells 53 and 88, same
  arguments. This is the correct primary choice for German text.
* Both are always trained with a weighted focal loss, AdamW, `lr=2e-5`, `batch_size=16`,
  `max_length=256`, 3 epochs: matching Table 3 of the paper.

### Calibration (current)

Two distinct implementations exist:

* **Cell 94**, scalar temperature only, fitted with `LBFGS(lr=0.01, max_iter=100)` on the
  validation logits, then applied to the test set. Functionally on the right side of the
  train/test boundary, but the same validation set was already consumed for best-epoch selection
  (cell 92), so it is reused twice.
* **Cell 122**, temperature and a per-class bias vector (`z/T + b`, 38 free parameters), fitted
  with `LBFGS(lr=0.1, max_iter=500, strong_wolfe)` on a 20 % slice of the fold-1 *training* data.
  This matches the paper's formulation (§3.1.5) more closely.

### Evaluation (current)

Metrics implemented across the notebook: accuracy, macro-F1, weighted-F1, top-2 accuracy,
per-category accuracy and mean confidence, confidence-threshold sweeps, ECE (10 equal-width bins,
cell 123), NLL, and a group-level ("one row per pattern") accuracy. The metric *code* is broadly
correct, the ECE implementation and the top-2 computation are both sound. The problem is not the
metrics; it is the data they are computed on.

---

## 2. Dataset summary

### 2.1 `synthetic_b2b_transactions_2000.xlsx`: the primary dataset

Despite the `.xlsx` extension this file is a single-column sheet holding semicolon-delimited text.
Header row and all 2 000 data rows split cleanly into exactly 65 fields (verified: field-count
distribution is `{65: 2000}`, no embedded semicolons, so the notebook's naive `str.split(";")` is
safe here).

| Property | Value |
|---|---|
| Transactions | 2 000 |
| Columns | 65 (Open-Banking-style schema) |
| Target: `categoryName` | **37 classes** |
| Parent: `parentCategoryName` | 11 classes |
| Unique **raw** `description` | 1 444 |
| Unique **cleaned** description | **170** |
| Description length | 18–63 chars, mean 34.4 |
| Description word count | 2–6 words, mean 3.6 |
| Raw descriptions mapped to >1 category | **0** (labels are perfectly consistent) |
| Date range (`transactionDate`) | 679 distinct dates |
| Vendors (`vendorName`) | 128 |

Only three columns are used by the notebook: `description`, `parentCategoryName`, `categoryName`.
The remaining 62 (including `amount`, `transactionDate`, `type`, `counterpartyNameRaw`, `vendorName`, `vendorGroupName`) are discarded. (Metadata fusion is outside the agreed
text-classification scope, but it is worth noting the data supports it.)

Class distribution is severely imbalanced (largest 240, smallest 9, ratio 26.7×):

```
240 IT & Software        76 Wartungsverträge     15 Kapitalerhöhungen
158 Löhne & Gehälter      73 Produktverkäufe      15 Tilgungen
149 Dienstleistungen      72 Energie              14 Dividenden / Ausschüttungen
121 Marketing & Vertrieb  65 Lizenzen             13 Zinsen
104 Abonnements           63 Leasingraten         13 Zuschüsse / Förderungen …
 99 Sozialabgaben         62 Miete & Nebenkosten  13 Maschinen & Anlagen
 98 Projektumsätze        56 Versicherungen       11 Fahrzeuge
 84 Handelswaren          51 Reisekosten          10 Versicherungsleistungen
 81 Beratung & Dienstl.   38 Instandhaltung       10 Boni / Prämien
                          38 Rohstoffe             9 Darlehensaufnahmen
                          35 IT / Hardware         9 Verkauf von Anlagevermögen
                          34 Rückvergütungen       9 Boni / Provisionen
                          26 Büromaterial          9 Immobilien / Umbauten
                          18 Software (CapEx)      9 Gesellschaftereinlagen
```

### 2.2 The decisive structural finding: template-generated text

The data was generated from a small set of templates with a rotating date suffix and a random
invoice reference:

```
'Software Lizenz 2024-09'                     -> 'software lizenz'
'Software Lizenz 2025-02'                     -> 'software lizenz'
'Rechnung RE-2024-6278 Beratungsleistung 2024-06' -> 'rechnung beratungsleistung'
'Rechnung RE-2024-4206 Beratungsleistung 2024-06' -> 'rechnung beratungsleistung'
```

After cleaning, 2 000 transactions collapse to 170 distinct texts. Mean group size is 11.8 rows;
the largest groups hold 79 rows each (`payroll sammelüberweisung`, `gehaltslauf`).

Unique cleaned patterns per category. This is the number that governs everything:

| Patterns/class | # classes | Classes |
|---|---|---|
| 1 | **4** | Versicherungsleistungen, Kapitalerhöhungen, Darlehensaufnahmen, Gesellschaftereinlagen |
| 2 | **17** | Versicherungen, Löhne & Gehälter, Lizenzen, Abonnements, Leasingraten, Wartungsverträge, Rohstoffe, Sozialabgaben, Zinsen, Produktverkäufe, Büromaterial, Handelswaren, Rückvergütungen, Tilgungen, Boni / Provisionen, Dividenden / Ausschüttungen, Boni / Prämien |
| 3 | 3 | Miete & Nebenkosten, Marketing & Vertrieb, Energie |
| 4–9 | 9 | IT & Software (4), Fahrzeuge (4), Verkauf von Anlagevermögen (5), Immobilien / Umbauten (5), Maschinen & Anlagen (6), Zuschüsse (7), Instandhaltung (8), IT / Hardware (8), Software (CapEx) (9) |
| 11–31 | 4 | Reisekosten (11), Dienstleistungen (12), Beratung & Dienstleistungen (13), Projektumsätze (31) |

Median = 2 patterns per class. 21 of 37 classes (57 %) have fewer than 3.

The *effective* sample size of this dataset for a leakage-safe experiment is therefore **170, not
2 000**: a 11.8× overstatement.

### 2.3 LLM synthetic datasets

| File | Rows | Classes | Unique texts | `source_text` col | Casing | Provenance |
|---|---|---|---|---|---|---|
| `synthetic_transactions_openai.csv` | 448 | 36 | 448 | ✗ | cleaned | Cell 75, first pass, target ≥15 patterns/class |
| `synthetic_fold1_paper_style.csv` | 7 078 | 37 | 7 041 (37 dups) | ✓ (163 sources) | **raw** | Cell 110, generated for the **leaky** `StratifiedKFold` fold-1 split |
| `final_new_synthetic.csv` | 2 310 | 35 | 2 310 | ✓ (110 sources) | **raw** | Cell 132, top-up increment for the final split |
| `final_synthetic_clean_7033.csv` | **7 033** | **37** | 7 025 (8 dups) | ✓ (**137 sources**) | cleaned | Cell 136, final, de-duplicated, test-filtered |

The `source_text` column is the critical asset: it records which real cleaned pattern each synthetic
example was derived from, which makes it possible to enforce leakage safety mechanically rather
than by trusting the generation procedure.

137 source patterns + 33 held-out test patterns = 170. Exactly the full pattern inventory. This
confirms `final_synthetic_clean_7033.csv` was generated *only* from the training side of the cell-126
holdout, and that the notebook's own leakage filters (cells 133–136) did their job for that
particular split.

`synthetic_fold1_paper_style.csv` draws on 163 source patterns. I.e. it was generated from a
split in which most test patterns were also visible in training. It is not reusable for a
leakage-safe experiment without filtering.

### 2.4 `Ergebnisuebersicht_FinBERT.xlsx`: prior results

Four sheets. The `Modellergebnisse` sheet is the only quantitative record that survived:

| Experiment | Val acc | Test acc | Note (translated) |
|---|---|---|---|
| FinBERT, original random split | — | **1.0000** | "Not meaningful: leakage through identical/similar patterns" |
| FinBERT V2, group split | 0.0840 | — | "Clean group split; no group overlap" |
| FinBERT V3, group split, corrected focal loss | **0.4809** | — | "Best observed run; strongly dependent on initialisation" |
| TF-IDF + Logistic Regression, group split | 0.0992 | — | "Baseline on identical group split" |
| FinBERT Final, group split, seed 42 | 0.0916 | — | "Reproducible run with fixed seed 42" |
| GermanFinBERT, group split | 0.0992 | — | "Best checkpoint epoch 2" |

The `Fehlklassifikationen` sheet is quietly the most informative artefact in the whole handover: it
lists 118 misclassifications drawn from only 5 distinct texts (`domain und server`,
`miete büro`, `strom und energie`, `online marketing kampagne`, and one more). The validation set of
131 rows contained roughly five independent items. Every reported validation accuracy in the
table above is, statistically, an average over a handful of Bernoulli trials.

---

## 3. Current methodology, step by step

### 3.1 Preprocessing (cell 13)

```python
def clean_description(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b20\d{2}[-/.]\d{1,2}\b", " ", text)   # 1. dates  2024-06
    text = re.sub(r"\b[a-z]{1,4}-?\d{3,}\b", " ", text)    # 2. refs   RE-2024-8096
    text = re.sub(r"[^a-zäöüß\s]", " ", text)              # 3. everything non-letter
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Steps 1 and 2 are well targeted and appropriate. Step 3 is too aggressive: it deletes *all*
digits and all uppercase-only distinctions, unconditionally. Observed damage:

```
'LinkedIn Ads B2B Leadgen 2024-06'        -> 'linkedin ads b b leadgen'
'Shop-Auszahlung 2024-06 B2B Produktverkauf ...' -> 'shop auszahlung b b produktverkauf'
'Projektabrechnung Onboarding Milestone 2 ...'   -> 'projektabrechnung onboarding milestone'
'Büroumbau B2B Portal'                    -> 'büroumbau b b portal'
```

`B2B` becomes the meaningless bigram `b b`; the milestone index vanishes. It also silently strips
`ß`-adjacent punctuation and any Latin-1 characters outside `äöüß` (e.g. `é`, `à` in vendor names).
There is no empty-string guard, the paper specifies a `nodescription` placeholder (§3.1.2); the
notebook has none (it happens not to bite on this dataset: 0 rows clean to empty).

The paper additionally specifies financial-abbreviation expansion (`ATM`→`cash`, `BACS`→`debit`)
and domain stop-word removal (`ref`, `ltd`, month abbreviations). Neither is implemented.

### 3.2 Grouping (cell 16)

```python
data["group_id"] = data.groupby("clean_description").ngroup()
```

Exact string match on the cleaned text. Cells 14–15 build a TF-IDF cosine-similarity matrix at
threshold 0.85 and print similar pairs, but **that similarity information is never used to form
groups**. It is diagnostic output only. So the grouping is exact-match, not semantic.

This is defensible and deterministic, and on this dataset it happens to be nearly optimal (the
templates collapse cleanly). Its limitation is that near-duplicates that survive cleaning (`miete büro` vs `miete bürofläche`, `boni prämien` vs `bonuszahlung mitarbeitende`) land in
different groups and can therefore be split across train and test.

### 3.3 Splitting: three mutually inconsistent methods

1. **Cell 4**. `train_test_split(stratify=label)` at row level. Pure random. Used for the V1
   experiment that produced 100 % test accuracy.
2. **Cells 17 / 21 / 22**. Group-aware. Cell 17 uses `GroupShuffleSplit`; cells 21 and 22 replace it
   with a hand-written per-category allocator. Cell 22 (the version that survives into all later
   cells) assigns, per category, sorted by group size ascending: smallest group → validation,
   second-smallest → test, remainder → training. Categories with 1 group go entirely to training;
   with 2 groups, one to test and one to training (no validation).
3. **Cell 100** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `cv_data` at row level, ignoring `group_id` entirely. 4. Cell 126) "final unseen-pattern holdout": per category with ≥2 patterns, one pattern is
   moved to test, the rest to train. Produces 33 test patterns and **no validation or calibration
   split at all**.

### 3.4 Augmentation

* **Cells 28–31**, rule-based: prepend/append one of `{zahlung, rechnung, buchung, abbuchung}` to
  an existing cleaned text. This creates *lexical* variants of patterns the model already sees; it
  adds no new semantic content and cannot help with unseen vendors.
* **Cells 23–27**, `seduerr/t5_base_paws_ger` paraphrasing and `google/flan-t5-small` generation.
  Explored and abandoned; outputs never enter any training set.
* **Cells 72–75 / 105–110 / 132–136**. Azure OpenAI (GPT) generation. This is the branch that
  produced the shipped CSVs and is the one aligned with the paper (§3.1.3). Prompt quality is
  reasonable: it pins the target category, lists all 37 categories to discourage cross-category
  drift, forbids verbatim copying, and demands a JSON array. `temperature=0.7` is set in cell 105 but
  dropped in cells 107/109, so the shipped data was generated at the API default.

### 3.5 Training

Consistent with paper Table 3: `lr=2e-5`, `batch=16`, `max_len=256`, `epochs=3`, weighted focal loss
(γ=2), balanced class weights, linear warmup. Later cells add gradient clipping (`max_norm=1.0`) and
best-checkpoint-by-validation-accuracy selection. Warmup is 10 % of steps in most cells but a literal
`500` in cells 36, 117 and 139. On a small training set that is more warmup steps than total steps,
which means the learning rate never reaches 2e-5 and decays from a fraction of it.

The focal loss appears in two forms:

* **Cells 9 / 81 (`FocalLoss`, `FocalLossFinal`)**. Computes `pt = torch.exp(-ce_loss)` where
  `ce_loss` is the *class-weighted* cross-entropy. Because the weight is inside the exponent,
  `pt ≠ p_t` whenever `α_t ≠ 1`, so the focal modulation `(1-pt)^γ` is wrong for every non-average
  class. This is the standard buggy implementation.
* **Cells 42 / 83 / 117 / 139 (`FocalLossV2`, `FocalLossCorrect`, `WeightedFocalLoss`)**: computes
  `p_t` from the softmax directly and multiplies by `α_t` outside. This is correct and matches
  the paper's `L = -α_t (1-p_t)^γ log(p_t)`.

The results table shows the difference is large: V2 (buggy loss) 8.4 % vs V3 (correct loss) 48.1 %.
Only the correct form should be carried forward.

---

## 4. Problems found

Ordered by severity.

### P1: Reported test accuracy of 1.0000 is a leakage artefact (critical)

Cell 4 splits 2 000 rows randomly. With 170 distinct texts and mean group size 11.8, essentially
every test text also appears in training. The model memorises a 170-entry lookup table. Cell 12's own
duplicate check confirms this. **This number must never appear in the thesis as a performance
result**; the notebook's annotation already says as much, and that judgement is correct.

### P2: The "paper-style" 5-fold cross-validation (cell 100) reintroduces the same leakage (critical)

`StratifiedKFold` at row level on `cv_data` ignores `group_id`. Cell 124's own leakage check exists
precisely because the author suspected this. Every metric derived from `fold1_val`, the entire
cells 119/123 evaluation, including the calibrated-vs-uncalibrated comparison and the ECE/NLL
figures. Is measured on a validation fold that shares texts with its training fold.

Consequence for the data: `synthetic_fold1_paper_style.csv` (7 078 rows, 163 source patterns) was
generated from that leaky fold. Because the LLM saw held-out patterns as generation seeds, this file
is contaminated with respect to any clean split and must be either filtered by `source_text` or
discarded.

### P3: No validation/calibration set survives the final split (high)

Cell 126 produces only train and test. Cells 137–140 then compensate by carving a validation set out
of the *synthetic* data (`train_test_split` on `final_synthetic`), so the best-epoch checkpoint is
selected on synthetic examples that are lexical siblings of the training examples. That validation
accuracy is not a generalisation estimate; it is close to a training metric. Meanwhile the real
calibration set used in cell 120 is drawn from `fold1_train`, i.e. it belongs to the leaky V6
split, not the final one.

### P4: Calibration set overlaps model-selection set (high)

Cell 94 fits temperature on the same validation logits already used to choose the best epoch.
Cell 122 fits `T` and 37 bias terms on ~256 rows that contain only a handful of distinct texts.
Fitting 38 free parameters on ~5–20 effectively-independent points will overfit; the resulting ECE
improvement is not a reliable estimate of deployment calibration.

### P5: Effective sample size is overstated ~12× everywhere (high)

All row-level metrics (accuracy over 590 test rows, 131 validation rows) treat 11.8 replicas of the
same string as 11.8 independent observations. The `Fehlklassifikationen` sheet makes this concrete:
118 errors, 5 distinct texts. Confidence intervals computed on row counts are far too narrow. The
notebook does contain a correct group-level evaluation (cells 98–99, 142). That is the honest view
and should be the headline, with row-level reported alongside.

### P6: Two-thirds of classes cannot support a leakage-safe test (high, structural)

With 4 classes at 1 pattern and 17 at 2 patterns, holding out one pattern per class means: 4 classes
get no test representation at all, and 17 classes are represented in test by exactly one text.
So their per-class recall can only ever be 0.000 or 1.000. Macro-F1 over 37 classes computed this way
has enormous variance. This is a property of the provided dataset, not a fixable bug. It is the
single most important limitation to state in the thesis.

### P7: Cross-generation state contamination (medium)

`label_encoder_v2` is fitted on `train_augmented` at cell 32 and then reused, unchanged, to transform
labels for `val_grouped`, `test_grouped`, `train_final`, and validation predictions in cells 39, 58,
77, 96. Later, a *different* encoder `cv_label_encoder` is fitted at cell 100 on `cv_data`. Class-index
→ name mappings are not guaranteed identical between the two, yet results from both are compared
side by side. Any per-class table mixing the two encoders is unsafe.

Similarly, `train_grouped`/`val_grouped`/`test_grouped` are overwritten three times (cells 17, 21, 22)
without re-running downstream cells, so what a cell actually operated on depends on execution order.

### P8: Preprocessing destroys informative tokens (medium)

`B2B` → `b b` (see §3.1). `b b` is now a shared token across `linkedin ads b b leadgen`,
`shop auszahlung b b produktverkauf` and `büroumbau b b portal`. Three *different* categories. The
cleaner is actively manufacturing a spurious cross-category feature.

### P9: Warmup misconfiguration (medium)

`num_warmup_steps=500` with `total_steps` sometimes < 500 (cells 36, 117, 139). The paper's "500" was
calibrated to a much larger corpus. Where total steps < warmup steps, the LR schedule is a pure ramp
that never reaches its peak. Training is effectively at a fraction of the intended 2e-5.

### P10: Buggy focal loss in the early generations (medium)

See §3.5. Cells 9 and 81 are mathematically incorrect. Any result attributed to them (V2's 8.4 %)
conflates the loss bug with the split change and cannot be interpreted.

### P11: Non-reproducibility (medium)

Seed is set once at cell 51. After V1–V4 were trained. `random_state=random.randint(0,100000)` in
cells 28–30 makes augmentation non-deterministic *even with* the seed set. `DataLoader(shuffle=True)`
has no generator. No `torch.use_deterministic_algorithms`. Outputs were stripped, so nothing can be
checked against a previous run.

### P12: Interactive secrets in code (low, but must be fixed)

Cells 63–65 and 68 call `input()` and `getpass.getpass()` for the Azure endpoint, deployment name and
API key. No key is committed (the handover was sanitised correctly, and `.env.example` is a template
only): but the pattern blocks non-interactive execution. Environment variables via `.env` are the
correct replacement.

### P13: Dead and duplicated code (low)

Roughly 40 % of cells are superseded: three paraphrase/generation experiments never used, four focal
loss classes, five near-identical training loops, `generate_paper_variants` redefined three times
(cells 105, 107, 109) with different signatures.

---

## 5. Leakage risk register

| # | Risk | Where | Status in prototype |
|---|---|---|---|
| L1 | Identical text in train and test | Cell 4 random split | **Present**, caused the 1.0000 |
| L2 | Identical text across CV folds | Cell 100 `StratifiedKFold` | **Present**, affects all V6 metrics |
| L3 | Synthetic data seeded from held-out patterns | Cell 110 → `synthetic_fold1_paper_style.csv` | **Present** in that file |
| L4 | Synthetic data seeded from held-out patterns | Cell 136 → `final_synthetic_clean_7033.csv` | **Absent**, correctly filtered (137 train sources only) |
| L5 | Synthetic text coinciding with a test pattern | Cells 133–136 | **Checked and removed** by the notebook |
| L6 | Calibration set overlapping test set | Cell 120 vs 119 | Absent for that pair, but the *split itself* (L2) is leaky |
| L7 | Calibration set overlapping model-selection set | Cell 94 | **Present** |
| L8 | Near-duplicate (not identical) texts across splits | All splits | **Unchecked**, no near-duplicate audit anywhere |
| L9 | Hyperparameters tuned on the test set | Cells 41→48 (`epochs 3→1` after inspecting results) | **Likely**. Epoch count was changed after observing performance |
| L10 | Label encoder fitted on data spanning splits | Cells 32, 100 | Benign here (labels only), but fragile |

---

## 6. Recommended corrections

| # | Correction | Addresses |
|---|---|---|
| C1 | One deterministic pipeline in `src/`, driven by a single `run_experiment.py` and one config; original notebook retained untouched as a historical record. | P7, P11, P13 |
| C2 | Rewrite cleaning: keep alphanumeric tokens, protect `B2B`-style tokens, strip only dates/reference numbers/punctuation; add German-aware abbreviation expansion and an explicit `nodescription` fallback. Log every transformation with before/after counts. | P8 |
| C3 | Keep exact-match-on-cleaned-text grouping as the primary, deterministic mechanism; add an **optional** near-duplicate merge (token-set Jaccard ≥ τ within the same label) and a mandatory **near-duplicate leakage audit** across the final splits. | L8 |
| C4 | Single group-based splitter producing **train / calibration / test**, disjoint at group level, with a hard assertion that all three group-ID sets are pairwise disjoint and that no cleaned text crosses splits. Fail loudly rather than warn. | P2, P3, L1, L2 |
| C5 | Use `final_synthetic_clean_7033.csv` as the augmentation corpus, but **re-filter it mechanically** by `source_text ∈ train_patterns` at run time. Do not rely on the file already being clean. Discard `synthetic_fold1_paper_style.csv` (or filter it identically). | L3, L4 |
| C6 | Augment **training groups only**. Never augment calibration or test. Assert zero synthetic text collides with any calibration/test pattern. | L3, L5 |
| C7 | Fit temperature scaling on a calibration split that is used for **nothing else**, not for early stopping, not for epoch selection. Prefer scalar `T` (1 parameter). Offer the paper's `T + b` vector variant as a clearly-labelled secondary result with an explicit over-fitting caveat, given ~16 effective calibration points. | P4, L7 |
| C8 | Report **both** row-level and group-level metrics side by side, and state the number of independent test patterns next to every accuracy figure. Treat group-level as the headline. | P5 |
| C9 | Use only the corrected focal loss. Set warmup as a *ratio* (10 %) with the literal-500 paper value available as a config option, and warn when warmup ≥ total steps. | P9, P10 |
| C10 | Full seed discipline: seed `random`, `numpy`, `torch`, CUDA; seeded `DataLoader` generator; deterministic augmentation sampling. Persist the exact config, split manifest and per-example predictions with every run. | P11 |
| C11 | Credentials from environment/`.env` only, never `input()`. **No new LLM generation is required**. The provided CSVs are sufficient, so the pipeline runs fully offline apart from the initial Hugging Face model download. | P12 |
| C12 | Report a deliberately-labelled **leaky random-split reference run** alongside the clean one, to quantify the leakage effect (expected ≈100 % vs the true figure). This turns the prototype's biggest mistake into the thesis's clearest evidence. | P1 |
| C13 | State the 170-pattern / median-2-per-class limitation prominently, and do not claim reproduction of the paper's 73.49 % / 90.36 % figures. Different language, different label set (37 vs 11), different data scale, synthetic rather than real bank data. | P6 |

---

## 7. Proposed final experiment design

### 7.1 Primary dataset: decision and justification

**Selected: `synthetic_b2b_transactions_2000.xlsx` as the labelled base, augmented by
`final_synthetic_clean_7033.csv`.**

Reasons:

1. It is the only file containing real (as-provided) labelled transactions with the full 37-class
   taxonomy, the 11-class parent taxonomy and Open-Banking metadata. Every other file is LLM output
   derived from it.
2. The handover README explicitly names it as the expected input workbook.
3. `final_synthetic_clean_7033.csv` is the only synthetic file that (a) covers all 37 classes,
   (b) carries `source_text` provenance, (c) is already cleaned and de-duplicated, and (d) was
   generated exclusively from the 137 training-side patterns. Verified above (137 + 33 = 170).

Rejected, with reasons:

* `synthetic_fold1_paper_style.csv`, generated from the leaky `StratifiedKFold` fold (163 source
  patterns, i.e. includes held-out ones). Usable only after `source_text` filtering; superseded by
  the 7 033 file.
* `final_new_synthetic.csv`, a 2 310-row *increment* already contained within the 7 033 file; using
  both would double-count. Covers only 35 of 37 classes.
* `synthetic_transactions_openai.csv`, 448 rows, no `source_text` provenance, so leakage safety
  cannot be verified mechanically. Excluded on that ground alone.

*Ambiguity flagged for the client:* the 7 033 file's leakage safety is guaranteed only for the
cell-126 split. The implementation will therefore re-derive the training pattern set and filter
by `source_text` at run time, so the guarantee holds for whatever split the config produces.

### 7.2 Pipeline

```
XLSX (2 000 rows, 65 cols)
  └─ parse semicolon payload, keep description / categoryName / parentCategoryName
     └─ Clean(·)      German-aware, conservative                    → clean_description
        └─ Group(·)   exact match on clean_description (+ optional  → group_id  (~170)
                      near-duplicate merge)
           └─ GroupSplit(train / calibration / test)   disjoint group IDs
              ├─ train  ── + LLM synthetic filtered by source_text ∈ train patterns
              │            └─ Fine-tune  GermanFinBert_FP   (primary)
              │            └─ Fine-tune  ProsusAI/finbert   (comparison, identical everything else)
              ├─ calib  ── fit temperature T  (NLL, LBFGS)  ← used for nothing else
              └─ test   ── final evaluation, uncalibrated vs calibrated
```

### 7.3 Split policy (per category, deterministic, seed 42)

| Patterns in class | Test | Calibration | Train |
|---|---|---|---|
| ≥ 3 | 1 | 1 | rest |
| 2 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |

Expected outcome: 33 test patterns (33 classes represented), ~16 calibration patterns,
~121 training patterns. Every count will be printed and saved, not assumed.

Hard assertions before any training starts:
`train ∩ cal = ∅`, `train ∩ test = ∅`, `cal ∩ test = ∅` on group IDs; zero exact cleaned-text overlap
across splits; near-duplicate audit (token-Jaccard ≥ 0.8 cross-split pairs) reported and required to
be zero or explicitly acknowledged; zero synthetic rows whose `source_text` lies outside train.

### 7.4 Models: held identical

Same split, same label encoder, same focal loss, same optimiser, same schedule, same seeds, same
metrics, same 0.8 threshold. The only differences are the checkpoint and its tokenizer, which is
exactly the comparison of interest:

| | Primary | Comparison |
|---|---|---|
| Checkpoint | `scherrmann/GermanFinBert_FP` | `ProsusAI/finbert` |
| Pre-training language | German (financial) | English (financial) |
| Head | fresh, 37 classes, `ignore_mismatched_sizes=True` | same |

### 7.5 Metrics

*Classification:* accuracy, macro-F1, weighted-F1, per-class P/R/F1, confusion matrix, top-2 accuracy.
Each reported twice, once row-level and once group-level (one row per independent test pattern).

*Confidence & calibration:* mean/median/std confidence, ECE (15 bins), MCE, NLL, Brier, reliability
diagram: uncalibrated vs calibrated, on the calibration split and on the test split.

*Decision workflow at τ = 0.8 (configurable):* share ≥ τ (auto-processing rate), accuracy among
≥ τ, count and share < τ (manual review), plus a full threshold sweep. Reported with the explicit
caveat that ≥ 0.8 confidence is a routing heuristic, never a correctness guarantee.

### 7.6 Reference runs (clearly labelled, not headline)

1. **Leaky random split**. Reproduces the prototype's ≈100 % to quantify P1.
2. **TF-IDF + Logistic Regression** on the identical clean split, the paper's baseline family.
3. **Majority class / random**, floor.

### 7.7 What will *not* be claimed

* No claim of reproducing the paper's 73.49 % / 90.36 % / ECE 0.0048. Different language, 37 classes
  vs 11, synthetic rather than real bank data, 170 vs thousands of independent patterns.
* No claim that confidence ≥ 0.8 implies correctness.
* No claim that the group split is "the" correct split. It is *a* defensible leakage-safe split,
  and its cost (many classes reduced to a single test item) will be stated.

### 7.8 Expected honest outcome

Given 121 training patterns spread over 37 classes and a test set of 33 single-pattern items, test
accuracy well below the paper's 73 % is the realistic expectation, and a wide spread across seeds is
likely. **The deliverable is a methodologically sound, reproducible measurement: not a good
number.** If the LLM augmentation lifts unseen-pattern accuracy materially above the ~9 % the
prototype reached, that is the finding; if it does not, that is equally a finding, and the audit
above explains why the dataset makes it hard.

---

## 7b. Post-implementation addendum: which audit predictions held

> Scope note. This addendum was written after the *first* implementation of the
> pipeline. The numbers in the table below therefore come from that earlier pipeline
> (`token_set` grouping, label-aware near-duplicate merge at Jaccard 0.80, no
> duplicate removal, 184 groups, single seed): not from the final Step 2-6
> methodology. They are retained unchanged because the point of this section is
> whether the audit's *predictions* held, and they did.
>
> The final methodology and results are in [`DATA_PIPELINE.md`](DATA_PIPELINE.md),
> [`FINAL_REPORT.md`](FINAL_REPORT.md) and
> [`results/final_comparison.md`](results/final_comparison.md). Under the final
> pipeline the corresponding figures are GermanFinBERT 0.6485 ± 0.0271 and
> FinBERT 0.6121 ± 0.0254 group-level accuracy over five seeds, on 152 groups
> rather than 184. The audit's qualitative findings are unaffected by that change.

Sections 1-7 above are the original Phase-1 audit and have not been rewritten.

| Audit finding | Prediction | Measured outcome | Verdict |
|---|---|---|---|
| **P1** random split leaks | ≈100 % is memorisation | TF-IDF + LogReg on a row-level random split: **accuracy 1.0000**, with **97.25 %** of test rows carrying a cleaned text also present in training | **confirmed** |
| **P5** effective sample size overstated ~12× | row metrics too optimistic | 2 000 rows → **184** independent patterns (10.9×); test = 647 rows but only **33** patterns | **confirmed** |
| **P6** most classes cannot support a fair test | per-class recall degenerates | Every per-class recall in the test report is exactly **0.0000 or 1.0000**, all 33 test classes hold a single repeated pattern; 4 classes cannot appear at all | **confirmed, starkly** |
| **P4** calibration set too small for 38 parameters | vector scaling will overfit | Vector scaling drove calibration NLL from 1.1841 to **0.0000** on 16 independent patterns. Scalar `T` behaved sensibly (0.8246) | **confirmed** |
| **P8** cleaner destroys `B2B` | manufactured cross-category feature | Prototype: `LinkedIn Ads B2B Leadgen` → `linkedin ads b b leadgen`. Corrected cleaner preserves `b2b`; group count rises 170 → **184** because the spurious merges disappear | **confirmed** |
| **P3** no usable validation/calibration split survives | must be rebuilt | New split yields train 135 / calibration 16 / test 33 groups, all disjoint, verified | **addressed** |
| **L4** the 7 033-row corpus is clean for *its* split | reusable but split-dependent | 137 source patterns + 33 held-out = 170 exactly. Under a *different* split, 2 344 of 7 033 rows are correctly discarded by the provenance gate | **confirmed** |
| **§7.8** expected honest outcome | "test accuracy well below the paper's 73 %" | *(first implementation)* GermanFinBERT **0.6061**, FinBERT **0.5758**: far above the prototype's ~9 %, below the paper's 73 % | **confirmed** |

Two things the audit did not anticipate, both discovered during execution:

1. **The dominant error mode is direction-of-money confusion.** Twelve of GermanFinBERT's
   thirteen wrong test patterns cross a *parent* category boundary, and the confusions
   are near-synonym pairs where the discriminator is whether money flows in or out:
   `Versicherungen` ↔ `Versicherungsleistungen`, `Tilgungen` ↔ `Darlehensaufnahmen`,
   `Dividenden / Ausschüttungen` ↔ `Gesellschaftereinlagen`, and
   `software license revenue` classified as the *expense* class `IT & Software`. The sign
   of the `amount` column would separate 8 of the 13 observed errors outright; the
   other five are same-direction confusions it would not fix. `amount` is available in
   the source workbook but outside the agreed text-only scope. This is the single most
   actionable item for future work.
2. **The two models differ far more in confidence and ranking than in accuracy.** Across
   seeds 42/43/44 the top-1 gap is 0.5758 vs 0.5657, one point, with the sign flipping
   on seed 43, i.e. not a real difference on 33 test patterns. What *is* stable on every
   seed is top-2 accuracy (0.7374 vs 0.5960) and the auto-processing rate at the 0.8
   threshold (51–76 % vs exactly 0 %, because FinBERT's calibrated confidence never
   reaches 0.80). The accuracy-only comparison in the prototype's results workbook could
   not have revealed either. And a single-seed accuracy comparison would have reported
   a model difference that does not replicate.

---

## 8. Environment verified during audit

| Component | Value |
|---|---|
| OS | Windows 11 Pro 26200 |
| Python | 3.13.2 |
| GPU | NVIDIA RTX 3060, 12 GB VRAM, driver 610.88 (CUDA 13.3) |
| Torch | 2.13.0+cu126 (installed into project `.venv`) |
| Base env before audit | bare, no pandas / torch / transformers present |

12 GB VRAM is comfortable for BERT-base at `batch_size=16, max_len=256`; gradient accumulation will
still be exposed as a config option per the brief.
