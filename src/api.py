"""FastAPI inference service for German SME transaction classification.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
    Swagger UI: http://localhost:8000/docs

Serves the APPROVED registry version. Predictions carry a confidence and a routing
decision: ``automatic`` when confidence >= 0.80 (the research operating point),
otherwise ``manual_review``. Confidence is the model's uncalibrated softmax
probability; it is NOT a guarantee of correctness (on the research test set about
one in five automatic decisions was wrong).

Privacy: descriptions are processed in memory only. They are not logged, not
written to the monitoring log and not stored for retraining.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from .mlops_settings import get, load_mlops_config, resolve_path
from .model_artifact import ArtifactError, verify_version
from .monitoring import TransactionMonitor, text_features
from .registry import ModelRegistry, RegistryError

logger = logging.getLogger("api")
API_VERSION = "1.0.0"
CONFIDENCE_NOTE = ("confidence is the model's uncalibrated probability for the predicted category; "
                   "it is a routing signal, not a guarantee of correctness")


# ── schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"description": "Gehaltslauf 2026-01", "transaction_id": "tx-0001"}})
    description: str = Field(..., min_length=1, max_length=2000,
                             description="German bank-transaction description (Verwendungszweck)")
    transaction_id: str | None = Field(None, max_length=128, description="optional client reference, echoed back")


class Alternative(BaseModel):
    category: str
    probability: float


class PredictionResult(BaseModel):
    prediction_id: str
    transaction_id: str | None = None
    category: str
    parent_category: str | None = None
    confidence: float = Field(..., description=CONFIDENCE_NOTE)
    decision: str = Field(..., description="'automatic' (confidence >= threshold) or 'manual_review'")
    threshold: float
    alternatives: list[Alternative]
    calibrated: bool
    model_version: str


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: Any = Field(..., description="validated per item; invalid items are reported, not fatal")
    transaction_id: str | None = Field(None, max_length=128)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"items": [
        {"description": "SaaS Subscription 2025-04", "transaction_id": "a"},
        {"description": "Miete Büro Juni", "transaction_id": "b"}]}})
    items: list[BatchItem] = Field(..., min_length=1)


class BatchItemResult(BaseModel):
    index: int
    status: str = Field(..., description="'ok' or 'error'")
    transaction_id: str | None = None
    result: PredictionResult | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    model_version: str
    n_items: int
    n_ok: int
    n_errors: int
    n_automatic: int
    n_manual_review: int
    results: list[BatchItemResult]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"prediction_id": "3f6c...", "true_category": "Gehälter"}})
    prediction_id: str = Field(..., min_length=1, max_length=64)
    true_category: str = Field(..., min_length=1, max_length=200)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    api_version: str = API_VERSION
    detail: str | None = None


# ── service state ─────────────────────────────────────────────────────────────
class ServiceState:
    def __init__(self, cfg: dict[str, Any], registry_dir: Path, monitoring_dir: Path | None):
        self.cfg = cfg
        self.registry = ModelRegistry(registry_dir)
        self.monitoring_dir = monitoring_dir
        self.classifier = None
        self.metadata: dict[str, Any] = {}
        self.parents: dict[str, str] = {}
        self.vocab: set[str] = set()
        self.version: str | None = None
        self.load_error: str | None = None
        self.monitor = TransactionMonitor(None, cfg.get("monitoring", {}), "none", [], 0.8)

    def load(self) -> None:
        try:
            vdir = self.registry.approved_dir()
            meta = verify_version(vdir)
            from .inference import TransactionClassifier  # imports torch/transformers lazily
            clf = TransactionClassifier(vdir / "model", threshold=float(meta.get("confidence_threshold", 0.8)))
        except (RegistryError, ArtifactError, OSError, ValueError, KeyError, SystemExit) as exc:
            self.classifier, self.version = None, None
            self.load_error = str(exc)
            logger.error("Model not loaded: %s", exc)
            return
        self.classifier, self.metadata, self.version = clf, meta, meta["version_id"]
        self.parents = json.loads((vdir / "category_parents.json").read_text(encoding="utf-8"))
        self.vocab = set(json.loads((vdir / "train_vocabulary.json").read_text(encoding="utf-8")))
        reference = json.loads((vdir / "reference_profile.json").read_text(encoding="utf-8"))
        self.monitor = TransactionMonitor(reference, self.cfg.get("monitoring", {}), self.version,
                                          clf.class_names, clf.threshold, self.monitoring_dir)
        self.load_error = None
        logger.info("Serving model version %s (threshold %.2f)", self.version, clf.threshold)

    def require(self):
        if self.classifier is None:
            raise HTTPException(status_code=503, detail={"error": "model_unavailable",
                                                         "detail": self.load_error or "no model loaded"})
        return self.classifier

    def classify(self, texts: list[str], tx_ids: list[str | None]) -> list[PredictionResult]:
        clf = self.require()
        cleaned, probs_u, probs = clf.predict_proba(texts)
        top_k = int(get(self.cfg, "model.top_k", 3))
        out = []
        for raw, cln, p, tid in zip(texts, cleaned, probs, tx_ids):
            order = np.argsort(-p)
            best = int(order[0])
            conf = float(p[best])
            decision = "automatic" if conf >= clf.threshold else "manual_review"
            pid = uuid.uuid4().hex
            cat = clf.class_names[best]
            self.monitor.record_prediction(pid, text_features(raw, cln, self.vocab), best, conf, decision)
            out.append(PredictionResult(
                prediction_id=pid, transaction_id=tid, category=cat, parent_category=self.parents.get(cat),
                confidence=round(conf, 5), decision=decision, threshold=clf.threshold,
                alternatives=[Alternative(category=clf.class_names[int(i)], probability=round(float(p[int(i)]), 5))
                              for i in order[1:top_k]],
                calibrated=clf.calibrator is not None, model_version=self.version))
        return out


def _validate_description(value: Any, max_len: int) -> str:
    if not isinstance(value, str):
        raise ValueError("description must be a string")
    if not value.strip():
        raise ValueError("description is empty or whitespace")
    if len(value) > max_len:
        raise ValueError(f"description longer than {max_len} characters")
    return value


def create_app(cfg: dict[str, Any] | None = None, registry_dir: str | Path | None = None,
               monitoring_dir: str | Path | None = "default") -> FastAPI:
    cfg = cfg or load_mlops_config()
    reg_dir = Path(registry_dir) if registry_dir else resolve_path(get(cfg, "registry.dir", "model_registry"))
    if monitoring_dir == "default":
        monitoring_dir = resolve_path(get(cfg, "monitoring.log_dir", "monitoring"))
    state = ServiceState(cfg, reg_dir, Path(monitoring_dir) if monitoring_dir else None)
    max_len = int(get(cfg, "api.max_text_length", 512))
    max_batch = int(get(cfg, "api.max_batch_size", 256))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.load()
        yield

    app = FastAPI(title="German SME Transaction Classification API", version=API_VERSION, lifespan=lifespan,
                  description=("Classifies German bank-transaction descriptions into 37 categories with "
                               "GermanFinBERT and routes each prediction to automatic processing "
                               "(confidence >= 0.80) or manual review. " + CONFIDENCE_NOTE + "."))
    app.state.service = state

    @app.middleware("http")
    async def timing(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"request_id": rid})
            state.monitor.record_error("internal")
            response = JSONResponse(status_code=500, content={"error": "internal_error",
                                                              "detail": "unexpected server error", "request_id": rid})
        elapsed = time.perf_counter() - t0
        endpoint = getattr(request.scope.get("route"), "path", "unmatched")
        if endpoint != "/metrics":
            state.monitor.record_request(endpoint, response.status_code, elapsed)
        response.headers["x-request-id"] = rid
        # Never log bodies: they contain financial descriptions.
        logger.info("request", extra={"request_id": rid, "method": request.method, "endpoint": endpoint,
                                      "status": response.status_code, "latency_ms": round(elapsed * 1000, 2)})
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        state.monitor.record_error("validation")
        # Do not echo the offending input back: it may be a financial description.
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": errors,
                                                      "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(HTTPException)
    async def on_http(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "detail": exc.detail}
        detail.setdefault("request_id", getattr(request.state, "request_id", None))
        return JSONResponse(status_code=exc.status_code, content=detail)

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}, tags=["operations"])
    def health():
        if state.classifier is None:
            return JSONResponse(status_code=503, content=HealthResponse(
                status="unavailable", model_loaded=False, detail=state.load_error).model_dump())
        return HealthResponse(status="ok", model_loaded=True, model_version=state.version)

    @app.get("/model/info", tags=["operations"])
    def model_info() -> dict[str, Any]:
        clf = state.require()
        m = state.metadata
        return {"model_version": state.version, "model_name": m.get("model_name"),
                "base_checkpoint": m.get("base_checkpoint"), "base_revision": m.get("base_revision"),
                "n_classes": len(clf.class_names), "categories": clf.class_names,
                "confidence_threshold": clf.threshold, "calibrated": clf.calibrator is not None,
                "max_length": clf.max_length, "seed": m.get("seed"), "created_at": m.get("created_at"),
                "status": m.get("status"), "data": m.get("data"), "metrics": m.get("metrics"),
                "training_device": m.get("training", {}).get("device"), "git_commit": m.get("git_commit"),
                "confidence_note": CONFIDENCE_NOTE,
                "limits": {"max_text_length": max_len, "max_batch_size": max_batch}}

    @app.post("/predict", response_model=PredictionResult, tags=["classification"])
    def predict(body: PredictRequest):
        try:
            text = _validate_description(body.description, max_len)
        except ValueError as exc:
            state.monitor.record_error("validation")
            raise HTTPException(status_code=422, detail={"error": "validation_error", "detail": str(exc)}) from None
        return state.classify([text], [body.transaction_id])[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["classification"])
    def predict_batch(body: BatchRequest):
        state.require()
        if len(body.items) > max_batch:
            raise HTTPException(status_code=422, detail={"error": "validation_error",
                                                         "detail": f"batch larger than {max_batch} items"})
        results: list[BatchItemResult | None] = [None] * len(body.items)
        valid_idx, texts, ids = [], [], []
        for i, item in enumerate(body.items):
            try:
                texts.append(_validate_description(item.description, max_len))
                valid_idx.append(i)
                ids.append(item.transaction_id)
            except ValueError as exc:
                state.monitor.record_error("batch_item")
                results[i] = BatchItemResult(index=i, status="error", transaction_id=item.transaction_id, error=str(exc))
        if texts:
            for i, res in zip(valid_idx, state.classify(texts, ids)):
                results[i] = BatchItemResult(index=i, status="ok", transaction_id=res.transaction_id, result=res)
        ok = [r for r in results if r and r.status == "ok"]
        n_auto = sum(1 for r in ok if r.result.decision == "automatic")
        return BatchResponse(model_version=state.version, n_items=len(results), n_ok=len(ok),
                             n_errors=len(results) - len(ok), n_automatic=n_auto,
                             n_manual_review=len(ok) - n_auto, results=results)

    @app.post("/feedback", tags=["monitoring"])
    def feedback(body: FeedbackRequest) -> dict[str, Any]:
        """Attach the confirmed category to an earlier prediction (delayed ground truth)."""
        clf = state.require()
        if body.true_category not in clf.class_names:
            raise HTTPException(status_code=422, detail={"error": "validation_error",
                                                         "detail": "true_category is not in the label taxonomy"})
        outcome = state.monitor.record_label(body.prediction_id, body.true_category)
        if outcome != "matched":
            raise HTTPException(status_code=404, detail={"error": "unknown_prediction",
                                                         "detail": "prediction_id not found in the monitoring window"})
        return {"accepted": True, "model_version": state.version}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    def metrics():
        state.monitor.summary()
        return PlainTextResponse(generate_latest(state.monitor.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/monitoring/summary", tags=["monitoring"])
    def monitoring_summary() -> dict[str, Any]:
        return state.monitor.summary()

    @app.post("/model/reload", tags=["operations"])
    def reload(x_admin_token: str | None = Header(default=None)):
        """Reload the approved model after a promotion. Disabled unless TX_ADMIN_TOKEN is set."""
        expected = os.environ.get("TX_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(status_code=404, detail={"error": "not_found", "detail": "reload endpoint disabled"})
        if not hmac.compare_digest((x_admin_token or "").encode(), expected.encode()):
            raise HTTPException(status_code=403, detail={"error": "forbidden", "detail": "invalid admin token"})
        state.load()
        if state.classifier is None:
            raise HTTPException(status_code=503, detail={"error": "model_unavailable", "detail": state.load_error})
        return {"reloaded": True, "model_version": state.version,
                "time": datetime.now(UTC).isoformat(timespec="seconds")}

    return app


def _default_app() -> FastAPI:
    from .logging_utils import setup_logging
    setup_logging(os.environ.get("TX_LOG_LEVEL", "INFO"), fmt=os.environ.get("LOG_FORMAT", "json"))
    return create_app()


app = _default_app() if os.environ.get("TX_API_NO_AUTOCREATE") != "1" else None
