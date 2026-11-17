# Security and production-readiness review

Review date: 2026-11-17 (session 3). Scope: the FastAPI inference service, Docker images and
Compose file, model registry, monitoring, retraining tooling and dependencies. **Result: the
service is a reference implementation, not production-ready.** The items marked
*prerequisite* must be in place before the API is exposed beyond a trusted host.

## Summary

| # | Area | Finding | Status |
|---|---|---|---|
| 1 | Authentication / authorisation | `/predict`, `/predict/batch`, `/feedback`, `/metrics`, `/monitoring/summary`, `/model/info` are unauthenticated. Only `/model/reload` is protected (header `X-Admin-Token`, disabled unless `TX_ADMIN_TOKEN` is set) | **Prerequisite**: authenticate every client at a gateway / reverse proxy (OIDC or API keys); restrict `/feedback` to trusted labelling users (unauthenticated feedback could poison retraining); keep `/metrics` internal |
| 2 | Admin token comparison | was a plain `!=` string comparison (timing side channel) | **Fixed**: `hmac.compare_digest`; tests for missing and near-miss tokens |
| 3 | Input validation | Pydantic: description 1–2,000 chars, `transaction_id` ≤ 128, batch ≤ 256 items (`TX_MAX_BATCH_SIZE`), per-item errors in batches; verified 422 for empty, 2,001-char and 257-item requests (`docs/evidence/12_session3_docker_hardened_compose.log`) | OK. **Prerequisite**: request-body size limit at the proxy (FastAPI reads the whole body before validating) |
| 4 | Rate limiting | none in the application | **Prerequisite**: per-client rate limits at the proxy; inference is CPU-heavy (BERT), so this is also a cost/DoS control |
| 5 | TLS | Uvicorn serves plain HTTP | **Prerequisite**: terminate TLS at the proxy. Compose now publishes the API on **127.0.0.1 only** by default (`API_BIND`); MLflow likewise |
| 6 | Secrets | no secrets in the repository or images (scans in `FINAL_E2E_ACCEPTANCE.md`); `.env` is git-ignored; the optional build CA is a BuildKit secret, never a layer; `TX_ADMIN_TOKEN` comes from the environment | OK. Recommended: a secret manager instead of `.env` files |
| 7 | Container | non-root user uid 10001; code not writable by that user; registry mounted read-only; Compose now adds `cap_drop: ALL` and `no-new-privileges`; health check | OK (verified). Recommended: pin the base image by digest, rebuild regularly |
| 8 | Dependencies | pip-audit (2026-11-17): API image had pip 25.0.1 and setuptools 78.1.0 from the base image with published advisories; research `requirements.txt` pinned pillow 12.3.0 with advisories. torch 2.13.0 (checked as the PyPI version): none | **Fixed**: image upgrades pip 26.3 / setuptools 84.2.1, rebuilt image audits clean; pillow → 13.0.0 (plotting only; research checks re-run, results unchanged). Recommended: run pip-audit in CI |
| 9 | Privacy / data retention | descriptions are never logged or stored: the monitor keeps only numeric buckets (verified: 0 occurrences of a submitted description in the event log). Feedback stores prediction id + confirmed category | OK. **Prerequisite**: retention and rotation for `monitoring/events.jsonl` (it grows without bound) and a retention policy for feedback |
| 10 | Logging | access log has request id, method, endpoint, status, latency; bodies are never logged | OK |
| 11 | Model / registry access | file-based registry with SHA-256 checksums verified at load; approval and rollback are CLI operations without user authentication; the API mounts the registry read-only | **Prerequisite**: restrict write access to the registry directory to the release operator; keep `registry.json` history as the audit trail |
| 12 | MLflow | no authentication | Compose binds it to 127.0.0.1. **Prerequisite** for shared use: an authenticated tracking server |
| 13 | Model risk | about one in five automatic (≥ 0.80) decisions was wrong on the research test set; confidences are uncalibrated; trained on synthetic data only | **Prerequisite**: validation on authorised real data (`docs/EXTERNAL_VALIDATION_DATA_ASSESSMENT.md`) and human review of the manual-review queue |

## Reference deployment shape

```
client ──HTTPS──> reverse proxy / API gateway  (TLS, authN, rate limit, body-size limit, access log)
                     └──HTTP (private network)──> txclf API container (uid 10001, read-only registry)
operators ──> registry directory (write access restricted) ──> /model/reload with X-Admin-Token
```

## Reporting a vulnerability

Report privately to the repository owner (GitHub: mr-taha-saqib). Do not open a public issue
with exploit details.
