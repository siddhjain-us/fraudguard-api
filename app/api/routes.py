"""
API routes for the FraudGuard prediction service.
"""

import hashlib
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.model import predict, get_feature_stats
from app.db.database import log_prediction, get_predictions
from app.db.cache import get_cached, set_cached

router = APIRouter()


# ── Request / Response schemas ──────────────────────────────────

class TransactionInput(BaseModel):
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    v14: float = Field(..., description="PCA feature V14")
    v17: float = Field(..., description="PCA feature V17")

    model_config = {"json_schema_extra": {
        "examples": [{"amount": 150.00, "v14": -1.5, "v17": -0.8}]
    }}


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str
    flags: list[str]
    z_score_amount: float
    cached: bool = False
    timestamp: str


class HistoryResponse(BaseModel):
    predictions: list[dict]
    total: int


# ── Endpoints ───────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(txn: TransactionInput):
    """
    Predict fraud probability for a transaction.
    Uses Redis cache for repeated inputs, logs result to PostgreSQL.
    """
    # Check Redis cache first
    cache_key = _make_cache_key(txn)
    cached = await get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    # Run prediction through the MLP + 2.2σ rail
    try:
        result = predict(txn.amount, txn.v14, txn.v17)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = PredictionResponse(
        fraud_probability=result.fraud_probability,
        risk_level=result.risk_level,
        flags=result.flags,
        z_score_amount=result.z_score_amount,
        cached=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Cache in Redis (TTL 5 min) + log to PostgreSQL
    await set_cached(cache_key, response.model_dump(), ttl=300)
    await log_prediction(txn.model_dump(), response.model_dump())

    return response


@router.get("/history", response_model=HistoryResponse)
async def prediction_history(limit: int = 20, offset: int = 0):
    """Retrieve recent prediction history from PostgreSQL."""
    rows = await get_predictions(limit=limit, offset=offset)
    return HistoryResponse(predictions=rows, total=len(rows))


@router.get("/model/info")
async def model_info():
    """Return model metadata and statistical thresholds."""
    return {
        "model_type": "MLPClassifier (32-16-8 neurons, ReLU)",
        "training_method": "SMOTE-balanced dataset (455K transactions)",
        "safety_rail": "2.2σ statistical override",
        "recall": 0.94,
        "thresholds": get_feature_stats(),
    }


def _make_cache_key(txn: TransactionInput) -> str:
    """Deterministic cache key from input features."""
    raw = f"{txn.amount:.6f}:{txn.v14:.6f}:{txn.v17:.6f}"
    return f"pred:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
