# Delivery summary: German SME transaction classification

**Status: delivered on branch `mlops-production-delivery`; all 25 mandated end-to-end
steps PASS (CPU; no GPU available for the acceptance run). Research results unchanged.**

| Area | Delivered | Verified |
|---|---|---|
| Research | pipeline, data, frozen split, corpus v2, results untouched | split + corpus regenerate bit-identically; 19/11/66/12/120 checks pass |
| Model | GermanFinBERT seed 42, research configuration, trained on CPU, registered with checksums | test group accuracy 0.697 (published GPU run 0.667; five-seed mean 0.6485) |
| API | FastAPI: predict, batch (per-item errors), feedback, metrics, monitoring; 0.80 routing | 21 API tests + live curl transcript |
| Privacy | descriptions never logged or stored | allow-list check on 600 events: 0 violations |
| Docker | API image 1.77 GB (CPU PyTorch) + tools image; registry mounted read-only | container healthy, real inference |
| MLflow | local SQLite; 10 research runs imported; model `champion` alias | run `6e5f29aa…` |
| Monitoring | PSI input/prediction drift, delayed-label accuracy | quiet on held-out traffic, alert on drift traffic |
| Retraining | leakage-gated, validation-only promotion gate, dry-run, rollback | real candidate rejected; promotion + rollback shown with tiny model |
| CI | research integrity, lint + tests, Docker smoke test | green on GitHub Actions |
| Tests | 59 pytest (incl. the fixed research checks) | 78 % coverage |

**Before any business use:** validate on real (non-synthetic) transactions; about one
in five automatic decisions was wrong on the research test set; a GPU is recommended
for training (CPU ≈ 2 h per run).

Start: `README.md` (section 13) → `DEPLOYMENT.md` → `API_DOCUMENTATION.md`.
