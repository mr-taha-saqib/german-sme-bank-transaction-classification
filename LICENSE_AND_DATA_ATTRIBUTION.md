# Licence and data attribution

Release `v1.1.0-mlops`. Facts verified on 2026-11-17; this is not legal advice.

## 1. Project source code

**No open-source licence file is included in this repository** (there is no `LICENSE`). Usage
and ownership rights in the delivered code and documentation are therefore governed by the
agreement between the client and the developer (the Fiverr order terms), not by an open-source
licence. If the client wants the code under a specific licence (for example proprietary terms or
an open-source licence for publication), the copyright holder should add the corresponding
`LICENSE` file. No licence has been assumed here.

## 2. Model checkpoints

| Model | Role | Licence (from Hugging Face model metadata) | Commercial use |
|---|---|---|---|
| `scherrmann/GermanFinBert_FP`, revision `0b3622dd589d2fc2f518b4be12a9dc306b903df2` | base of the **delivered** model | **Apache-2.0** | permitted under Apache-2.0 (keep the licence and notices; changes must be stated) |
| `ProsusAI/finbert` | research **comparison** model only (`STEP4_FINBERT.md`) | **CC BY-NC-SA 4.0** | **not permitted**; do not deploy FinBERT-based models commercially |

The fine-tuned weights are not distributed in the repository or the ZIP; they are reproduced
from the pinned GermanFinBERT checkpoint (INSTALLATION_AND_SETUP.md §5). If you redistribute
fine-tuned weights, include the Apache-2.0 licence and attribute the base model:
*GermanFinBERT, scherrmann/GermanFinBert_FP, Hugging Face* (see the model card for the
associated publication).

## 3. Data

| Data | Origin | Status |
|---|---|---|
| `data/synthetic_b2b_transactions_2000.xlsx` | supplied with the research project; **template-generated synthetic** transactions; document metadata (author, path) removed | no real banking data, no personal data known |
| `data/final_synthetic_clean_7033.csv`, `final_new_synthetic.csv`, `synthetic_fold1_paper_style.csv`, `synthetic_transactions_openai.csv` | supplied with the research project; **LLM-generated** from the primary workbook | synthetic; the generating service's terms are not documented in the repository, so confirm them if the data is redistributed |
| `reference/` (taxonomy workbook, example transactions) | supplied with the research project | synthetic examples |
| `examples/demo/synthetic_batch.json` | written for this delivery | invented examples |

No third-party dataset is bundled. External real-world validation data was searched for and
none suitable was found (docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md). Any client data used
later remains the client's and must not be added to the repository.

## 4. Third-party software (main dependencies)

Licences as declared in the installed package metadata (pinned versions):

| Package | Version | Licence |
|---|---|---|
| PyTorch (`torch`) | 2.13.0 | BSD-3-Clause AND BSD-2-Clause AND Apache-2.0 (incl. LLVM exception) AND BSL-1.0 AND MIT (declared `License-Expression`) |
| transformers, tokenizers, safetensors, huggingface_hub | 5.16.1, 0.23.1, 0.8.0, 1.33.0 | Apache-2.0 |
| FastAPI, Pydantic | 0.163.1, 2.14.3 | MIT |
| Uvicorn, Starlette, httpx | 0.54.0, 1.10.0, 0.28.1 | BSD-3-Clause |
| MLflow | 3.18.0 | Apache-2.0 |
| prometheus-client | 0.26.0 | Apache-2.0 and BSD-2-Clause |
| numpy, pandas, scipy, scikit-learn | 2.5.x, 3.0.x, 1.18.1, 1.9.x | BSD |
| matplotlib | 3.11.x | matplotlib licence (PSF-based) |
| openpyxl, PyYAML | 3.1.5, 6.0.3 | MIT |
| pytest, ruff (development) | 9.2.0, 0.17.5 | MIT |

Docker images additionally contain Debian packages from `python:3.12-slim` (various free
licences) and, for the optional MLflow service, the official `ghcr.io/mlflow/mlflow` image
(Apache-2.0). All listed licences permit commercial use subject to their notice requirements.

## 5. Approvals still required from the client

* Agreement on the licence/ownership terms of the delivered code (section 1).
* Legal basis and approvals for any real transaction data used for validation or retraining.
* Confirmation that the supplied synthetic data may be redistributed, if the client plans to
  publish it.
