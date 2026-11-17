# External validation data assessment (Project 1)

Assessment date: 2026-11-17. Goal: find a genuinely real, anonymised, appropriately licensed
**German SME bank-transaction** dataset with category labels, to validate the delivered model
outside its synthetic training data. This is an engineering summary of published terms, not legal
advice.

## Result

**No suitable dataset could be verified.** No external validation was run, and no external
data was added to the project. The frozen research split, corpus and published results are
unchanged. **Authorised, labelled real transaction data from the client is required** (see
"What is needed" below).

## Candidates checked

| Dataset | Real data? | Language / domain | Labels | Licence / access | Why it does not fit |
|---|---|---|---|---|---|
| **MoneyData** (Firat et al., EuroVis 2023; Mendeley Data DOI 10.17632/dnxtg6n4rv.1) | yes, anonymised | UK retail bank (English context), apparently **one personal customer**, ~6,500 transactions 2015–2022 | 19 categories assigned manually by the authors (described by them as arbitrary), plus keyword-based and bank type codes | CC BY 4.0 (commercial use allowed with attribution) | not German, not SME, one consumer's account, taxonomy unrelated to the 37 German SKR-style categories. Results would say nothing about German SME bookkeeping |
| Berka / PKDD'99 Czech financial dataset | yes, anonymised | Czech retail bank, 1993–1998 | transaction type codes, no free-text descriptions | research challenge data | no German descriptions, no bookkeeping categories |
| UCI / Statlog German Credit (and its Kaggle/HF copies) | yes | German credit applicants | good / bad credit risk | UCI citation policy | credit-scoring attributes, no transactions |
| Multimodal Banking Dataset (MBD) | anonymised real | corporate clients, transaction sequences | product-purchase targets | public research release | no raw textual descriptions |
| AML time-series (Kaggle, GPL-3) | anonymised real (differential privacy + PCA) | corporate accounts | fraud flag | GPL-3 | no descriptions; fraud task |
| anhaltai/german-multifin (HF, CC BY 4.0) | real | German financial news **headlines** | topic classes | CC BY 4.0 | not transactions |
| amp Data Intelligence "2K bank account transactions, 18–35 year olds in Germany" (datarade.ai) | stated as GDPR-compliant | German **consumer** accounts, with purpose text | not stated | **commercial product**, price on request | consumer not SME, labels unknown, would need a purchase and contract review; not obtained |
| finAPI, Tink and similar categorisation APIs | real (customer data) | German accounts | provider taxonomies | commercial service contracts | not datasets; customer data cannot be redistributed |

Synthetic sets on Kaggle and Hugging Face (for example generic "bank transaction" or fraud sets)
were excluded: they would not answer the question, and labelling them as real is not allowed.

## Why the requirement is hard to meet publicly

Transaction descriptions contain counterparties, names, IBANs, invoice and customer numbers.
Under the GDPR they are personal data of the account holder and of third parties, and banks are
also bound by banking secrecy. Public releases therefore either drop the text (Berka, MBD, AML)
or come from a single consenting individual (MoneyData). A labelled German SME corpus would have
to come from the client or from a data partner under contract.

## What is needed from the client

1. **Data:** a representative export of real SME transactions (target: at least 2,000
   transactions from several businesses and banks, covering all 37 categories, ideally ≥ 30 per
   category) with description, amount, date and the **category a bookkeeper actually booked**.
2. **Authorisation:** written confirmation of the legal basis for using the data for model
   validation (and, if wanted, retraining), a data-processing agreement where a processor is
   involved, and retention/deletion rules.
3. **Pseudonymisation before hand-over:** replace names, IBANs, customer/invoice numbers with
   placeholders consistently (the model's preprocessing already strips long digit runs, see
   `strip_long_digit_runs` in `src/config.py`).
4. **Label mapping:** if the client's chart of accounts differs from the 37 categories in
   `reference/`, a mapping table, with unmapped accounts listed explicitly.

## How the evaluation will be run once data is available

* Keep the client data **separate** from the frozen research split (`results/dataset/`); never
  mix it into training, validation or test of the published experiment.
* Evaluate the delivered model as is (no tuning on the evaluation set) with `src/inference.py`
  on the external CSV: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix,
  automatic-routing coverage at the 0.80 threshold, accuracy of automatic decisions, and
  confidence distribution; list the worst failure cases.
* Report external results in their own file next to, never instead of, the research results.
* Propose retraining only if there are enough labelled rows per category; use the leakage-gated
  retraining workflow (`RETRAINING.md`) with a held-out portion of the client data for validation.
