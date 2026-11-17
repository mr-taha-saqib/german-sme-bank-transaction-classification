# syntax=docker/dockerfile:1
# German SME transaction classification API (CPU inference).
# Model weights are NOT baked in: mount the registry at /app/model_registry.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_FORMAT=json \
    TX_REGISTRY_DIR=/app/model_registry \
    TX_MONITORING_DIR=/app/monitoring \
    TX_MLFLOW_ENABLED=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

WORKDIR /app

# Optional build secret for networks with a TLS-inspecting proxy:
#   docker build --secret id=pip_ca,src=/path/to/ca.pem .
# Without it these are plain pip installs; the CA never enters an image layer.
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
COPY requirements-api.txt ./
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements-api.txt

RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app
COPY configs/ ./configs/
COPY src/ ./src/
RUN mkdir -p /app/model_registry /app/monitoring && chown -R app:app /app/monitoring

USER app
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ── Tools image: research pipeline + retraining + MLflow (CPU) ─────────────────
FROM base AS tools
USER root
COPY requirements.txt requirements-mlops.txt ./
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements-mlops.txt
COPY *.py ./
COPY data/ ./data/
COPY results/dataset/ ./results/dataset/
COPY scripts/ ./scripts/
ENV HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TX_MLFLOW_ENABLED=true \
    MLFLOW_TRACKING_URI=sqlite:////mlflow-data/mlflow.db TX_MLFLOW_ARTIFACTS=/mlflow-data/mlartifacts \
    MLFLOW_DISABLE_AGENT_HINT=1 HF_HOME=/app/.cache/huggingface
RUN mkdir -p /mlflow-data /app/reports /app/artifacts /app/.cache && chown -R app:app /app /mlflow-data
USER app
CMD ["python", "-m", "src.retrain", "--help"]
