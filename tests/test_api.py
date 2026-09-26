"""API integration tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.db import init_db
from api.main import app

# Ensure tables exist before any request runs — normally created by
# FastAPI's startup event, but that only fires when TestClient is used as
# a context manager, so we call it explicitly here for a plain client.
init_db()
client = TestClient(app)

SAMPLE_CUSTOMER = {
    "customer_id": 12345,
    "recency_days": 15.0,
    "frequency": 8.0,
    "monetary_total": 450.0,
    "monetary_avg": 56.25,
    "historical_clv": 450.0,
    "avg_basket_size": 2.5,
    "purchase_trend_slope": 0.1,
    "purchase_interval_variance": 20.0,
    "seasonal_concentration": 0.3,
    "return_rate": 0.05,
    "discount_sensitivity": 0.2,
    "engagement_score": 65.0,
}


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()


def test_health_endpoint_reports_status():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "database_reachable" in body


def test_predict_requires_valid_schema():
    """Malformed input (negative recency) must return a clear 422, not a 500."""
    bad_payload = {**SAMPLE_CUSTOMER, "recency_days": -5.0}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_missing_field_returns_422():
    incomplete = {"customer_id": 1}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_returns_prediction_when_model_available():
    """Only meaningful once a model artifact exists (run_pipeline.py has
    been run); otherwise the API correctly reports 503."""
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["customer_id"] == SAMPLE_CUSTOMER["customer_id"]


def test_model_metadata_endpoint():
    resp = client.get("/model/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_name" in body
    assert "feature_columns" in body


def test_batch_predict_rejects_non_csv():
    resp = client.post(
        "/predict/batch",
        files={"file": ("data.txt", b"not,a,csv", "text/plain")},
    )
    assert resp.status_code == 400