"""
FraudGuard API - Production ML Prediction Service
Wraps the MLP neural network fraud detection model in a FastAPI service
with PostgreSQL logging, Redis caching, and the 2.2σ statistical safety rail.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.db.database import init_db, close_db
from app.ml.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load ML model + init DB. Shutdown: close connections."""
    load_model()
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="FraudGuard API",
    description=(
        "Production ML service for credit card fraud detection. "
        "Uses an MLP neural network with a 2.2σ statistical safety rail."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Kubernetes liveness/readiness probe endpoint."""
    return {"status": "healthy", "model_loaded": True}
