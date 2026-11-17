# Installation and setup

Release `v1.1.0-mlops`. Each block is marked:

* **[Tested]**: executed in the final delivery QA (results in
  [FINAL_DELIVERY_QA_REPORT.md](FINAL_DELIVERY_QA_REPORT.md)).
* **[Executed earlier]**: executed during the delivery (evidence in `docs/evidence/`), not
  repeated in the final QA because of its run time.
* **[Not independently verified]**: documented for convenience; not run by us.

## 1. Supported environments

| Item | Status |
|---|---|
| Linux x86-64 (Ubuntu 24.04), Python 3.12.3 | **verified** (local runs and containers) |
| Docker 24+ / Compose v2, image base Debian 13 (`python:3.12-slim`) | **verified** |
| GitHub Actions `ubuntu-latest`, Python 3.12 | **verified** (CI) |
| macOS, Windows (native or WSL 2) | not verified; expected to work with Python 3.12 or Docker |
| Python 3.11 or older | **not supported** (pinned numpy needs 3.12) |
| NVIDIA GPU / CUDA training | not re-verified in the delivery environment |

Hardware: API ≈ 1 GB RAM, 2 CPU cores recommended; training ≈ 5 GB RAM on CPU (≈ 2 h per run).

## 2. Get the code

From the release ZIP: unzip it and `cd` into the extracted folder. From GitHub **[Tested]**:

```bash
git clone https://github.com/mr-taha-saqib/german-sme-bank-transaction-classification.git
cd german-sme-bank-transaction-classification
git checkout v1.1.0-mlops          # the released version (or stay on main)
```

## 3. Virtual environment and dependencies **[Tested]**

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1  [Not independently verified]
python -m pip install --upgrade pip   # the pip bundled with some Python builds (e.g. Ubuntu's 24.0) has published advisories
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
pip install -r requirements-dev.txt -r requirements.txt
pip check
```

* `requirements.txt`: research pipeline (pinned). `requirements-mlops.txt`: API + MLOps layer
  (included by `requirements-dev.txt`). `requirements-api.txt`: what the API image installs.
* For a CUDA build of PyTorch use the matching index URL from pytorch.org instead of `/whl/cpu`
  **[Not independently verified]**.

## 4. Configuration

```bash
cp .env.example .env               # optional; every value has a safe default
```

Settings live in `configs/mlops.yaml`; environment variables override them (full list in
[TECHNICAL_DOCUMENTATION.md §11](TECHNICAL_DOCUMENTATION.md#11-configuration-and-environment-variables)).
`TX_ADMIN_TOKEN` is the only secret: leave it empty unless you need `POST /model/reload`, and
never commit `.env`.

## 5. Model weights

The weights (≈ 420 MB) are not in the repository or the ZIP. Choose one:

### 5a. Tiny demo model (minutes, offline) **[Tested]**

```bash
python scripts/build_demo_model.py --registry model_registry
```

A 1-layer BERT trained briefly on the frozen train split. **Its predictions are meaningless**;
use it only to try the API, monitoring and Docker.

### 5b. Real GermanFinBERT model **[Executed earlier]** (`docs/evidence/02_model_training_and_registration.log`)

```bash
# base checkpoint, pinned revision (≈ 440 MB download, cached in ~/.cache/huggingface)
python -c "from huggingface_hub import snapshot_download as s; print(s('scherrmann/GermanFinBert_FP', revision='0b3622dd589d2fc2f518b4be12a9dc306b903df2'))"
# fine-tune exactly as in the research (seed 42, corpus v2); ≈ 2 h on 2 CPU cores
python train_model.py --model germanfinbert --augment --augmentation-version v2 --seed 42 \
    --output-dir artifacts/runs --run-name germanfinbert_seed42
# package into the registry and approve explicitly
python -m src.model_artifact --run artifacts/runs/germanfinbert_seed42 --approve \
    --reason "reviewed validation/test metrics"
```

If a registry from 5a exists, the newly approved real model replaces it as the approved version
(both stay in the registry). Expected test group accuracy on CPU: 0.6970 (see MODEL_CARD.md).
Bit-identical weights are not guaranteed across hardware (CPU fp32 vs GPU mixed precision).

## 6. Start the API locally **[Tested]**

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health          # {"status":"ok","model_loaded":true,...}
./examples/demo/run_demo.sh                # synthetic walk-through of every endpoint
```

Swagger UI: http://127.0.0.1:8000/docs. Stop with Ctrl+C.

## 7. Docker

Build and run **[Tested]**:

```bash
docker compose config -q                   # validate the Compose file
docker compose build api                   # behind a TLS-inspecting proxy see below
docker compose up -d api                   # needs an approved model in ./model_registry (section 5)
docker compose ps                          # STATUS shows (healthy) after ≈ 45 s
curl http://127.0.0.1:8000/health
docker compose logs -f api                 # JSON logs (Ctrl+C to leave)
```

The registry is bind-mounted **read-only**; the container runs as uid 10001 and must be able to
read `model_registry/` (default file permissions are fine). The port is published on
127.0.0.1 only; set `API_BIND=0.0.0.0` in `.env` only behind a TLS reverse proxy with
authentication.

Behind a TLS-inspecting corporate proxy **[Tested]** (in the delivery environment):

```bash
DOCKER_BUILDKIT=1 docker build --secret id=pip_ca,src=/path/to/proxy-ca.pem --target base \
    -t german-sme-txclf-api:1.0.0 .
```

Research and retraining tools in a container **[Tested]** (build + `verify_dataset.py`; the image is ≈ 2.7 GB):

```bash
docker compose --profile tools build tools
docker compose --profile tools run --rm tools python verify_dataset.py
```

## 8. MLflow

Local UI against the project database **[Tested]**:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts
# open http://127.0.0.1:5000   (no authentication: keep it on localhost)
```

Containerised server (`ghcr.io/mlflow/mlflow:v3.18.0`): `docker compose --profile mlflow up -d mlflow`
(status in the QA report; it needs access to ghcr.io).

A fresh checkout has no `mlflow.db`; it is created by the first training, packaging or
retraining run with tracking enabled.

## 9. Health checks

| Check | Expected |
|---|---|
| `curl http://127.0.0.1:8000/health` | HTTP 200, `"model_loaded": true` |
| no approved model, or weights checksum mismatch | HTTP 503, `"status": "unavailable"`, `"model_loaded": false`, `detail` gives the reason |
| `docker compose ps` | `(healthy)`; the image's health check calls `/health` every 20 s (60 s start period) |

## 10. Stopping services and cleaning up **[Tested]**

```bash
docker compose down                        # stop containers, keep the monitoring volume
docker compose down -v                     # also delete the monitoring volume
deactivate                                 # leave the virtual environment
rm -rf .venv monitoring reports artifacts mlartifacts mlflow.db   # local state (keeps model_registry)
```

Delete `model_registry/` only if you want to remove all model versions.
