"""API tests (real inference through a tiny test model)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api import create_app


@pytest.fixture
def client(approved_registry, mlops_cfg, tmp_path):
    with TestClient(create_app(mlops_cfg, monitoring_dir=tmp_path / "mon")) as c:
        yield c


@pytest.fixture
def no_model_client(mlops_cfg, tmp_path):
    with TestClient(create_app(mlops_cfg, registry_dir=tmp_path / "empty", monitoring_dir=None)) as c:
        yield c


def test_health_and_model_info(client, approved_registry):
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["model_version"] == approved_registry[1]
    info = client.get("/model/info").json()
    assert info["n_classes"] == 37 and len(info["categories"]) == 37
    assert info["confidence_threshold"] == 0.8 and info["calibrated"] is False
    assert "not a guarantee" in info["confidence_note"]


def test_predict_schema_and_routing(client):
    r = client.post("/predict", json={"description": "Gehaltslauf 2026-01", "transaction_id": "t-1"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction_id", "transaction_id", "category", "parent_category", "confidence", "decision",
                         "threshold", "alternatives", "calibrated", "model_version"}
    assert body["transaction_id"] == "t-1"
    assert body["decision"] == ("automatic" if body["confidence"] >= 0.8 else "manual_review")
    assert 0.0 <= body["confidence"] <= 1.0 and len(body["alternatives"]) == 2
    assert all(a["probability"] <= body["confidence"] for a in body["alternatives"])


def test_api_matches_direct_classifier(client, approved_registry):
    from src.inference import TransactionClassifier
    reg, _ = approved_registry
    clf = TransactionClassifier(reg.approved_dir() / "model")
    direct = clf.predict(["SaaS Subscription 2025-04"]).iloc[0]
    api = client.post("/predict", json={"description": "SaaS Subscription 2025-04"}).json()
    assert api["category"] == direct["predicted_category"]
    assert api["confidence"] == pytest.approx(direct["confidence"], abs=1e-5)


@pytest.mark.parametrize("payload", [
    {}, {"description": ""}, {"description": "   "}, {"description": 123}, {"description": None},
    {"description": "x" * 600}, {"description": "ok", "unexpected": 1},
])
def test_invalid_predict_requests(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_validation_errors_do_not_echo_input(client):
    secret = "IBAN DE89370400440532013000 Gehalt Max Mustermann" + "x" * 600
    r = client.post("/predict", json={"description": secret})
    assert r.status_code == 422 and "Mustermann" not in r.text


def test_malformed_json(client):
    r = client.post("/predict", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch_reports_item_errors_without_corrupting_batch(client):
    items = [{"description": "Miete Büro Juni", "transaction_id": "a"}, {"description": "  ", "transaction_id": "b"},
             {"description": 7, "transaction_id": "c"}, {"description": "Software Lizenz 2025-02", "transaction_id": "d"}]
    body = client.post("/predict/batch", json={"items": items}).json()
    assert body["n_items"] == 4 and body["n_ok"] == 2 and body["n_errors"] == 2
    assert [r["index"] for r in body["results"]] == [0, 1, 2, 3]
    assert [r["status"] for r in body["results"]] == ["ok", "error", "error", "ok"]
    assert body["results"][3]["result"]["transaction_id"] == "d"
    single = client.post("/predict", json={"description": "Software Lizenz 2025-02"}).json()
    assert body["results"][3]["result"]["category"] == single["category"]
    assert body["n_automatic"] + body["n_manual_review"] == 2


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422
    big = {"items": [{"description": "Miete"}] * 257}
    assert client.post("/predict/batch", json=big).status_code == 422


def test_feedback_flow(client):
    pred = client.post("/predict", json={"description": "Miete Büro"}).json()
    assert client.post("/feedback", json={"prediction_id": pred["prediction_id"],
                                          "true_category": "Not-A-Category"}).status_code == 422
    ok = client.post("/feedback", json={"prediction_id": pred["prediction_id"], "true_category": pred["category"]})
    assert ok.status_code == 200
    again = client.post("/feedback", json={"prediction_id": pred["prediction_id"], "true_category": pred["category"]})
    assert again.status_code == 404


def test_metrics_and_monitoring(client):
    for t in ("Miete Büro", "Gehaltslauf", "Strom Abschlag"):
        client.post("/predict", json={"description": t})
    m = client.get("/metrics").text
    assert "txclf_predictions_total" in m and "txclf_confidence_bucket" in m
    s = client.get("/monitoring/summary").json()
    assert s["predictions_in_window"] == 3
    assert s["decisions"]["automatic"] + s["decisions"]["manual_review"] == 3
    assert s["observed_performance"]["status"] == "insufficient_labels"


def test_monitoring_log_never_contains_descriptions(client, tmp_path):
    text = "Zahlung an Firma Geheimname GmbH 2026-01"
    client.post("/predict", json={"description": text})
    log = (tmp_path / "mon" / "events.jsonl").read_text(encoding="utf-8")
    assert "Geheimname" not in log and "Zahlung" not in log
    ev = json.loads(log.splitlines()[-1])
    assert {"char_bucket", "token_bucket", "unseen_bucket", "category", "confidence"} <= set(ev)


def test_model_unavailable(no_model_client):
    assert no_model_client.get("/health").status_code == 503
    for r in (no_model_client.post("/predict", json={"description": "Miete"}),
              no_model_client.get("/model/info"),
              no_model_client.post("/predict/batch", json={"items": [{"description": "Miete"}]})):
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"


def test_corrupted_weights_are_not_served(approved_registry, mlops_cfg):
    reg, _ = approved_registry
    w = next((reg.approved_dir() / "model").glob("*.safetensors"))
    data = bytearray(w.read_bytes())
    data[-5] ^= 0xFF
    w.write_bytes(bytes(data))
    with TestClient(create_app(mlops_cfg, monitoring_dir=None)) as c:
        r = c.get("/health")
        assert r.status_code == 503 and "Checksum" in r.json()["detail"]


def test_reload_protection(client, monkeypatch):
    assert client.post("/model/reload").status_code == 404
    monkeypatch.setenv("TX_ADMIN_TOKEN", "t0k")
    assert client.post("/model/reload", headers={"x-admin-token": "bad"}).status_code == 403
    assert client.post("/model/reload").status_code == 403                      # header missing
    assert client.post("/model/reload", headers={"x-admin-token": "t0k-"}).status_code == 403
    assert client.post("/model/reload", headers={"x-admin-token": "t0k"}).json()["reloaded"]


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/model/info", "/predict", "/predict/batch", "/metrics", "/monitoring/summary", "/feedback"):
        assert p in paths
