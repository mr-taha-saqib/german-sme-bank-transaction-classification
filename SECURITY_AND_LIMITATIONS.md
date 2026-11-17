# Security and limitations

Release `v1.1.0-mlops`. Summary for decision makers; the detailed review with evidence is
[SECURITY.md](SECURITY.md). **Status: delivery-ready reference implementation, not production-ready.**

## 1. Current security measures (implemented and tested)

| Measure | Detail |
|---|---|
| Input validation | strict schemas (`extra="forbid"`), description 1–512 characters, batch ≤ 256, per-item batch errors, HTTP 422 with field-level messages |
| No input echo | error bodies and logs never contain the submitted description |
| Privacy-preserving monitoring | only numeric buckets (length, token count, unseen-token share, digit share) and the prediction are stored; verified 0 descriptions in the event log |
| Model integrity | weights SHA-256 verified before serving; safetensors only (no pickle checkpoints) |
| Admin endpoint | `/model/reload` disabled unless `TX_ADMIN_TOKEN` is set; constant-time token comparison |
| Container | non-root uid 10001, code not writable at runtime, registry mounted read-only, `cap_drop: ALL`, `no-new-privileges`, health check, memory limit |
| Network exposure | Compose publishes the API and MLflow on **127.0.0.1 only** (`API_BIND` to change) |
| Dependencies | pinned versions; pip-audit clean for the API image and the project environment at release |
| Secrets | none required; `.env` is git-ignored; `.env.example` contains placeholders only; build-time proxy CA via BuildKit secret, never in a layer |

## 2. Status of controls you must provide

| Control | Status | Required action |
|---|---|---|
| Authentication / authorisation | **not implemented** for `/predict`, `/predict/batch`, `/feedback`, `/metrics`, `/monitoring/summary`, `/model/info` | API gateway or reverse proxy with OIDC/OAuth2 or API keys; restrict `/feedback` to trusted reviewers (feedback influences retraining) and `/metrics` to monitoring |
| TLS | **not implemented** (plain HTTP) | terminate TLS at the proxy |
| Rate limiting | **not implemented** | per-client limits at the proxy (BERT inference is CPU-heavy) |
| Request-size limit | schema limits only; the full body is read before validation | body-size limit at the proxy |
| Secret management | environment variables / `.env` | a secret manager for `TX_ADMIN_TOKEN` |
| Registry access control | file permissions only; approval and rollback are CLI operations | restrict write access to the release operator |
| MLflow | no authentication (localhost only) | authenticated tracking server if shared |
| Log retention | `monitoring/events.jsonl` grows without bound | rotation and retention policy |
| Backups, scaling, high availability | not provided (single host, per-process monitoring state) | your platform |

## 3. Data privacy

* No real banking or personal data is contained in the repository or the package; all data is
  synthetic.
* The running API processes descriptions in memory only. Your reverse proxy, gateway or client
  logs may still record request bodies; configure them accordingly.
* Real client data used for validation or retraining must have a documented legal basis
  (GDPR, banking secrecy), pseudonymisation and retention rules
  ([docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md](docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md)).

## 4. Dataset and model licensing

* Training data: synthetic data supplied with the research repository; no third-party dataset.
* Base model: `scherrmann/GermanFinBert_FP` is licensed **Apache-2.0** (verified on Hugging Face).
* The research **comparison** model `ProsusAI/finbert` is licensed **CC BY-NC-SA 4.0
  (non-commercial)**; it is not the delivered model and must not be used commercially.
* Details: [LICENSE_AND_DATA_ATTRIBUTION.md](LICENSE_AND_DATA_ATTRIBUTION.md).

## 5. Known risks and model limitations

* **No validation on real transactions.** All accuracy figures come from synthetic data and are
  not evidence of production accuracy. Authorised, labelled client data is required.
* Small test set (33 independent groups; one group = 3 accuracy points).
* About one in five automatic decisions was wrong on the research test set; confidences are
  uncalibrated. Keep a human review step and spot-check automatic bookings.
* Drift alerts indicate changed inputs, not measured accuracy loss.
* The default promotion gate accepts a candidate that ties the current model
  (`min_improvement` 0.0) and promotes automatically; configure a margin and human approval.
* GPU training not re-verified in the delivery environment; CPU training takes ≈ 2 h.
* The research scripts (`final_evaluation.py`, `calibrate_models.py`) read the saved research
  outputs `results/**/*.npz` with `allow_pickle=True`, because each file stores the 37 class names
  as an object array next to the numeric logits. Only run them on the files shipped in this
  package or produced by your own runs; never on `.npz` files from untrusted sources. The MLOps
  layer (packaging, API) reads numeric arrays only (`allow_pickle=False`).

## 6. Production deployment requirements

Before production use: client data validation (and likely retraining) with sign-off; gateway with
TLS, authentication, authorisation and rate limiting; secret management; registry access control;
log retention; monitoring/alerting integration (Prometheus scrape); backup of `model_registry/`;
a documented promotion policy with human approval; a data-protection assessment.

## 7. Outside the delivered scope

Cloud deployment, ERP/accounting-system integration, user management, a review-queue UI,
calibrated probabilities, horizontal scaling, real-data collection and labelling, legal review.
