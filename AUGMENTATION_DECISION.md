# Augmentation Decision: Verification, Correction and Final Corpus

**Status: RESOLVED. Option A applied. Corpus `v2` is the final training corpus for the
model comparison.**

| | |
|---|---|
| Verification | `python verify_augmentation.py` → **11 checks passed, 0 findings** |
| Build corpus | `python prepare_augmentation.py --version v2` |
| Final corpus | `results/dataset/augmentation_corpus_v2.csv`, sha256 `4eb242d9f384041f…` |
| Rows removed by the correction | **9 of 4 550 (0.20 %)** |
| Frozen split | unchanged, sha256 `f715f040d8a5daa6…` |
| GermanFinBERT non-augmented result | unchanged (group accuracy 0.3939) |
| Original `data/final_synthetic_clean_7033.csv` | never modified |

---

## 1. Decision

Option A was applied: a third gate rejects any synthetic row reaching the project's own
grouping threshold (token-Jaccard ≥ 0.60) against any validation or test pattern.
The threshold is read from `GroupingConfig.near_duplicate_threshold` (the same value Step 2 uses to merge real patterns into groups), so the grouping definition and the
augmentation gate cannot drift apart. **No new knob was introduced and the grouping
threshold was not changed.**

Two corpus versions are now built and frozen, so the correction is auditable rather
than merely asserted:

| Version | Gates | Rows | Max similarity to a held-out pattern |
|---|---|---|---|
| `v1` | A + B | 4 550 | **0.6667**, violates the rule |
| **`v2`** | **A + B + C** | **4 541** | **0.5000**, compliant |

`v2` is the corpus used from here on.

---

## 2. The three gates

| Gate | Rejects | Dropped |
|---|---|---|
| **A, provenance** | rows whose `source_text` resolves to any non-training group | 2 483 |
| **B, collision** | rows whose text equals, or shares a canonical group key with, a held-out pattern | 0 |
| **C, near-duplicate** *(new)* | rows reaching Jaccard ≥ 0.60 against any held-out pattern | **9** |
| within-corpus duplicates | identical (text, label) pairs | 0 |
| | **7 033 → 4 541** | **2 492 removed** |

Reconciliation is exact and verified: 7 033 − 4 541 = 2 492 = 2 483 + 0 + 9 + 0.

### The 9 rows gate C removed

All same-category, all Jaccard 0.667, all traceable to two legitimate *training*
source patterns:

| Synthetic training text | Rows | Generated from (training) | Collided with held-out |
|---|---|---|---|
| `maintenance support vertrag` | 2 | `wartungsvertrag service level` | `maintenance support` (test group 134) |
| `contract maintenance support` | 1 | `wartungsvertrag service level` | `maintenance support` |
| `fee maintenance support` | 1 | `wartungsvertrag service level` | `maintenance support` |
| `maintenance service support` | 1 | `wartungsvertrag service level` | `maintenance support` |
| `maintenance sla support` | 1 | `wartungsvertrag service level` | `maintenance support` |
| `abbuchung office supplies` | 2 | `büromaterial bestellung` | `office supplies` (test group 138) |
| `kauf office supplies` | 1 | `büromaterial bestellung` | `office supplies` |

The causal path was always innocent. The generation prompt carries one training
`source_text` and its category, nothing else, so no test text ever reached the LLM.
The problem was the *effect*: by the project's own grouping rule those training rows
and those test items were the same leakage unit.

---

## 3. Dataset counts after correction

| Quantity | v1 | **v2 (final)** |
|---|---|---|
| Raw corpus rows | 7 033 | 7 033 |
| Kept synthetic rows | 4 550 | **4 541** |
| Distinct source patterns used | 103 | 103 |
| Training groups drawn on | 88 of 103 | 88 of 103 |
| Categories with synthetic support | 29 of 37 | **29 of 37** |
| Real training rows | 1 093 | 1 093 |
| **Combined training corpus** | 5 643 | **5 634** |
| Class imbalance (combined) | 89.7× | 89.7× |
| Max similarity to a held-out pattern | 0.6667 | **0.5000** |

Only two classes lost rows, and neither lost coverage:

| Category | v1 | v2 | Δ |
|---|---|---|---|
| `Wartungsverträge` | 200 | 194 | −6 |
| `Büromaterial` | 190 | 187 | −3 |

Categories still without synthetic support (unchanged by the correction):
`Boni / Prämien`, `Dividenden / Ausschüttungen`, `Energie`, `Leasingraten`,
`Löhne & Gehälter`, `Rückvergütungen`, `Sozialabgaben`, `Versicherungen`.

Combined training distribution ranges from 269 (`Projektumsätze`) down to 3
(`Boni / Prämien`).

---

## 4. Leakage verification after correction

`verify_augmentation.py`. 11 passed, 0 findings.

| # | Check | Result |
|---|---|---|
| 1 | Every raw row carries a `source_text` | PASS |
| 2 | Rejection counts reconcile (now including gate C) | PASS |
| 3 | Every rejected row has an identified reason | PASS (0 unattributed) |
| 4 | Every kept row's source resolves only to training groups | PASS (0 violations) |
| 5 | No kept row's source touches validation/test | PASS (0 violations) |
| 6 | No synthetic text equals a held-out pattern | PASS (0) |
| 7 | No synthetic text shares a group key with held-out | PASS (0) |
| 8 | **No synthetic text is a near-duplicate (J ≥ 0.60) of a held-out pattern** | **PASS (0)** |
| 9 | Gating is deterministic | PASS |
| 10 | Corpus is model-agnostic | PASS |
| 11 | Corpus is plain text + label, consumable identically by both models | PASS |

Similarity distribution against held-out patterns, before and after:

| Max similarity ≥ | v1 keys | **v2 keys** |
|---|---|---|
| 0.9 / 0.8 / 0.7 | 0 | 0 |
| **0.6** | **7** | **0** |
| 0.5 | 34 | 27 |

The 27 remaining keys at ≥ 0.5 sit below the project's leakage threshold and are left
in place deliberately: 0.5 is where real patterns of *different* categories start
colliding (`DATA_PIPELINE.md` §6), so gating there would remove legitimate variety.

### Model-agnostic guarantee, now auditable

`prepare_augmentation.py` freezes the corpus with a SHA-256 over `(clean_description,
categoryName)`. `train_model.py` loads that file and verifies the checksum before
training, exactly as it does for the split. Previously the corpus was rebuilt per run,
deterministic, but only as trustworthy as the gating code. Both models will now
provably consume the same 4 541 rows.

---

## 5. Retrained result: and an honest finding

GermanFinBERT re-trained on `v2`, identical frozen split, identical configuration,
seed 42.

| Metric | Group level (n=33) | Row level (n=647) |
|---|---|---|
| Accuracy | **0.6667** | 0.4946 |
| Macro F1 | 0.5638 | 0.5375 |
| Weighted F1 | 0.5980 | 0.4377 |
| Top-2 accuracy | 0.7576 | 0.6291 |
| Mean confidence | 0.7796 | 0.6844 |
| Predictions ≥ 0.8 | **63.6 %** | 42.8 % |
| Accuracy among ≥ 0.8 | **0.8095** | 0.7437 |

The correction changed no prediction. Both previously-compromised test groups were
still classified correctly, with marginally *higher* confidence, and **0 of 33 test
groups flipped**:

| Test group | v1 (leaky) | v2 (corrected) |
|---|---|---|
| 134 `maintenance support` | Wartungsverträge, conf 0.897: correct | Wartungsverträge, conf 0.900: correct |
| 138 `office supplies` | Büromaterial, conf 0.884: correct | Büromaterial, conf 0.894: correct |

So the earlier 0.6667 was not inflated by the leakage. That is a genuine result and
it is worth stating plainly rather than quietly dropping: the leakage was real by the
project's own definition and had to be removed, but its measurable effect on the
headline number was zero. The reason is that both affected categories were already
supported by ~190 other synthetic rows, so the nine near-copies carried no information
the model did not already have.

The correction is still the right call. The guarantee now holds, and a future split or
corpus might not be so forgiving. But nobody should claim it "fixed" an inflated score.

---

## 6. Comparison: A / B / C

Group level, n = 33 independent test groups, uncalibrated:

| | **A** no augmentation | **B** previous augmented (v1, leaky) | **C** corrected augmented (v2) |
|---|---|---|---|
| Training rows | 1 093 | 5 643 | **5 634** |
| Accuracy | 0.3939 | 0.6667 | **0.6667** |
| Macro F1 | 0.2924 | 0.5638 | **0.5638** |
| Weighted F1 | 0.3101 | 0.5980 | **0.5980** |
| Top-2 accuracy | 0.4242 | 0.7273 | **0.7576** |
| Mean confidence | 0.1721 | 0.7472 | **0.7796** |
| Max confidence | 0.4210 | 0.9242 | **0.9313** |
| Predictions ≥ 0.8 | **0.0 %** | 57.6 % | **63.6 %** |
| Accuracy among ≥ 0.8 | – | 0.7895 | **0.8095** |

Row level, n = 647:

| | A | B | C |
|---|---|---|---|
| Accuracy | 0.2535 | 0.4946 | **0.4946** |
| Macro F1 | 0.2623 | 0.5357 | **0.5375** |
| Weighted F1 | 0.2156 | 0.4389 | **0.4377** |
| Top-2 accuracy | 0.2906 | 0.5920 | **0.6291** |
| Predictions ≥ 0.8 | 0.0 % | 33.7 % | **42.8 %** |
| Accuracy among ≥ 0.8 | – | 0.6743 | **0.7437** |

Reading of this:

* **A → C is the real effect.** Augmentation is worth +27.3 accuracy points at group
  level and is what makes the 0.8 workflow function at all: without it the model's
  single most confident prediction scores 0.42 and nothing is ever automated.
* **B → C is not a meaningful difference.** Accuracy, macro-F1 and weighted-F1 are
  identical to five decimals. Top-2 (+3.0 pts) and the confidence figures differ because
  removing 9 of 5 643 training rows perturbs the optimisation trajectory, ordinary
  run-to-run variation, not a systematic gain. It should not be reported as an
  improvement.

---

## 7. Limitations

1. **Lexical only.** Gate C mirrors the grouping rule, so it inherits its blind spot: a
   synthetic row semantically identical to a held-out pattern but lexically disjoint
   (`gehaltslauf` vs `payroll sammelüberweisung`) is caught by neither.
2. **Threshold-dependent.** At 0.5 there are still 27 near-neighbours. 0.6 is used
   because it is the project's grouping threshold; a stricter project would remove more.
3. **Eight categories have no synthetic support**, and the corpus *increases* class
   imbalance from 67.5× to 89.7×: the opposite of the paper's inverse-frequency intent.
   This is a consequence of reusing a corpus generated against the prototype's split, and
   is unaffected by the correction.
4. **Label correctness is unverified.** Nothing here checks that a synthetic row's
   category is semantically right; the notebook relied on prompt constraints, and no
   manual validation was performed.
5. **Single seed.** All figures are seed 42. At n = 33 one test group is worth 3.03
   accuracy points, so B-vs-C differences of this size are inside noise.
6. **Deviation from the paper**, unchanged by the correction: the paper fine-tunes on the
   balanced synthetic set alone (§3.1.4); this project trains on real + synthetic
   combined, which is more conservative and matches the client's notebook.

---

## 8. Status and what comes next

Done in Step 3A:

- gate C added to `src/splitting.py`, driven by the Step-2 grouping threshold;
- `prepare_augmentation.py` builds and freezes versioned, checksummed corpora;
- `train_model.py` loads and verifies the corpus instead of rebuilding it;
- `verify_augmentation.py` extended, 11 checks, 0 findings;
- GermanFinBERT re-trained on `v2` → `results/step3a_germanfinbert_augmented_v2/`.

Unchanged, as required: the frozen split, the GermanFinBERT non-augmented results, the
grouping threshold, the original corpus CSV.

FinBERT has not been trained. Step 4 should use `--augment --augmentation-version v2`
so both models consume the identical checksum-verified corpus, and should run several
seeds before drawing any conclusion about which model is better.
