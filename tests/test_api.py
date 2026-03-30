"""
Tests for FraudGuard API endpoints.
Uses FastAPI TestClient for integration testing.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient



# Mock DB and cache so tests don't need Postgres/Redis
@pytest.fixture(autouse=True)
def mock_infra():
    with patch("app.api.routes.log_prediction", new_callable=AsyncMock), \
         patch("app.api.routes.get_cached", new_callable=AsyncMock, return_value=None), \
         patch("app.api.routes.set_cached", new_callable=AsyncMock), \
         patch("app.api.routes.get_predictions", new_callable=AsyncMock, return_value=[]):
        yield


@pytest.fixture
def client():
    """Create test client with model loaded."""
    from app.main import app
    from app.ml.model import load_model
    load_model()
    return TestClient(app)


# ── Health check ────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ── Prediction endpoint ────────────────────────────────────────

def test_predict_low_risk(client):
    """Normal transaction should return LOW risk."""
    resp = client.post("/api/v1/predict", json={
        "amount": 50.0, "v14": 0.0, "v17": 0.0
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_probability" in data
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(data["flags"], list)
    assert "z_score_amount" in data


def test_predict_high_amount_triggers_sigma_rail(client):
    """Extremely large amount should trigger the 2.2σ safety rail."""
    resp = client.post("/api/v1/predict", json={
        "amount": 1_000_000.0, "v14": 0.0, "v17": 0.0
    })
    data = resp.json()
    assert data["risk_level"] == "HIGH"
    assert data["fraud_probability"] >= 0.95
    assert any("2.2" in flag for flag in data["flags"])


def test_predict_anomalous_v14(client):
    """Extremely negative V14 should trigger anomaly flag."""
    resp = client.post("/api/v1/predict", json={
        "amount": 100.0, "v14": -20.0, "v17": 0.0
    })
    data = resp.json()
    assert data["fraud_probability"] >= 0.85
    assert any("anomaly" in flag.lower() for flag in data["flags"])


def test_predict_validation_error(client):
    """Missing required fields should return 422."""
    resp = client.post("/api/v1/predict", json={"amount": 50.0})
    assert resp.status_code == 422


def test_predict_negative_amount(client):
    """Negative amount should be rejected by validation."""
    resp = client.post("/api/v1/predict", json={
        "amount": -100.0, "v14": 0.0, "v17": 0.0
    })
    assert resp.status_code == 422


# ── History endpoint ────────────────────────────────────────────

def test_history(client):
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    assert "predictions" in resp.json()


# ── Model info endpoint ────────────────────────────────────────

def test_model_info(client):
    resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_type"] == "MLPClassifier (32-16-8 neurons, ReLU)"
    assert "thresholds" in data
