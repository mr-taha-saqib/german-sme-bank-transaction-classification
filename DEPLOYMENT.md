# Deployment

Local / single-host container deployment. **No public cloud deployment has been
performed.** Evidence of the executed Docker run: `docs/evidence/`.

## Requirements

| | API (inference) | Training / retraining |
|---|---|---|
| CPU | 2 cores recommended | 2+ cores |
| RAM | ≈ 1 GB in use; 3 GB limit in compose | ≈ 5 GB on CPU (GermanFinBERT, batch 16, max_length 128) |
| GPU | not needed | optional; RTX 3060 12 GB: ≈ 2 min per run (research). CPU (2 cores): ≈ 2 h per run (measured). The delivered model was trained on CPU; GPU training was not re-run during delivery (no NVIDIA GPU available) |
| Disk | image 1.77 GB (CPU PyTorch) + model 420 MB | + 420 MB Hugging Face cache per base checkpoint |
| Network | none at runtime (`HF_HUB_OFFLINE=1`) | Hugging Face download on first training run |

## Model weights

Weights are not in git (≈ 420 MB). Produce them reproducibly, then register:

```bash
# 1. base checkpoint (pinned revision 0b3622dd589d2fc2f518b4be12a9dc306b903df2)
python -c "from huggingface_hub import snapshot_download as s; print(s('scherrmann/GermanFinBert_FP', revision='0b3622dd589d2fc2f518b4be12a9dc306b903df2'))"
# 2. fine-tune exactly as in the research (seed 42, corpus v2); GPU if available
python train_model.py --model germanfinbert --augment --augmentation-version v2 --seed 42 \
    --output-dir artifacts/runs --run-name germanfinbert_seed42
# 3. package into the registry and approve explicitly
python -m src.model_artifact --run artifacts/runs/germanfinbert_seed42 --approve \
    --reason "reviewed validation/test metrics against the published seed-42 run"
```

The Hugging Face checkpoint is cached in `~/.cache/huggingface` (override with
`HF_HOME`). No token is needed; `HF_TOKEN` only raises rate limits.

## Docker

```bash
docker compose build api                          # image german-sme-txclf-api:1.0.0 (target 'base')
docker compose up -d api                          # mounts ./model_registry read-only, publishes 127.0.0.1:8000
docker compose ps                                 # (healthy) once the model is loaded (~10-20 s)
docker compose logs -f api                        # JSON logs, no request bodies
docker compose restart api
docker compose down                               # keeps the 'monitoring' volume
docker compose build api && docker compose up -d api   # rebuild after code changes
```

Retraining and research tools in a container (CPU):

```bash
docker compose --profile tools build tools
docker compose --profile tools run --rm tools python verify_dataset.py
docker compose --profile tools run --rm -v "$PWD/new.csv:/data/new.csv:ro" tools \
    python -m src.retrain --feedback /data/new.csv --dry-run
```

Optional MLflow UI: `docker compose --profile mlflow up -d mlflow` → http://localhost:5000
(bound to 127.0.0.1; MLflow has no authentication).

Compose hardening (default): the API port is published on **127.0.0.1** only (`API_BIND`,
`API_PORT`); the container runs as uid 10001, drops all Linux capabilities, sets
`no-new-privileges`, cannot write its own code and mounts the registry read-only. The image
upgrades the base image's pip/setuptools (pip-audit clean on 2026-11-17). Expose the API to
other hosts only behind a TLS reverse proxy with authentication and rate limiting (SECURITY.md).

Behind a TLS-inspecting proxy, pass its CA as a build secret (never stored in a layer):
`docker build --secret id=pip_ca,src=/path/to/ca.pem --target base -t german-sme-txclf-api:1.0.0 .`

## Production prerequisites not included

TLS termination, API authentication and authorisation, rate limiting and request-size limits
at a gateway (SECURITY.md), log rotation and retention for `monitoring/`, horizontal
scaling (monitoring state is per process), external metrics storage, GPU serving, and
validation on authorised real transaction data.
This is still not a production banking system: no ERP integration, receipt
processing or audit-grade data retention.
