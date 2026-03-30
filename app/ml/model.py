"""
Model loading and prediction logic.
Reuses your existing fraud_model.pkl, scaler.pkl, and feature_stats.pkl
with the 2.2σ statistical safety rail from the original FraudGuard project.
"""

import os
import numpy as np
import joblib
from dataclasses import dataclass


@dataclass
class PredictionResult:
    fraud_probability: float
    risk_level: str
    flags: list[str]
    z_score_amount: float


# Global model state (loaded once at startup)
_model = None
_scaler = None
_stats = None
_amt_threshold = None
_v14_danger = None
_v17_danger = None


MODEL_DIR = os.getenv("MODEL_DIR", "models")


def load_model():
    """Load model artifacts at startup. Called by FastAPI lifespan."""
    global _model, _scaler, _stats, _amt_threshold, _v14_danger, _v17_danger

    _model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _stats = joblib.load(os.path.join(MODEL_DIR, "feature_stats.pkl"))

    # 2.2 Sigma thresholds (from your original app.py)
    _amt_threshold = _stats["Amount"]["mean"] + (2.2 * _stats["Amount"]["std"])
    _v14_danger = _stats["V14"]["mean"] - (2.2 * _stats["V14"]["std"])
    _v17_danger = _stats["V17"]["mean"] - (2.2 * _stats["V17"]["std"])


def predict(amount: float, v14: float, v17: float) -> PredictionResult:
    """
    Run fraud prediction using the MLP model + 2.2σ safety rail.
    Mirrors the logic from your original Streamlit app.py.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Build the 30-feature vector (same as your app.py)
    vec = np.zeros((1, 30))
    vec[0, 14] = v14   # V14 feature
    vec[0, 17] = v17   # V17 feature
    vec[0, 29] = amount # Amount feature

    # Neural network prediction
    prob = float(_model.predict_proba(_scaler.transform(vec))[0][1])

    # 2.2σ Statistical Safety Rail overrides
    flags = []
    if amount > _amt_threshold:
        prob = max(prob, 0.95)
        flags.append(f"Amount exceeds 2.2σ threshold (>${_amt_threshold:,.2f})")
    if v14 < _v14_danger or v17 < _v17_danger:
        prob = max(prob, 0.85)
        flags.append("Statistical pattern anomaly in V14/V17")

    # Z-score for audit trail
    z_score = (amount - _stats["Amount"]["mean"]) / _stats["Amount"]["std"]

    # Risk level classification
    if prob < 0.15:
        risk_level = "LOW"
    elif prob < 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return PredictionResult(
        fraud_probability=round(prob, 6),
        risk_level=risk_level,
        flags=flags,
        z_score_amount=round(z_score, 4),
    )


def get_feature_stats() -> dict:
    """Return feature statistics for the API info endpoint."""
    if _stats is None:
        return {}
    return {
        "amount_threshold_2_2sigma": round(_amt_threshold, 2),
        "v14_danger_threshold": round(_v14_danger, 4),
        "v17_danger_threshold": round(_v17_danger, 4),
        "amount_mean": round(_stats["Amount"]["mean"], 2),
        "amount_std": round(_stats["Amount"]["std"], 2),
    }
