# FraudGuard API

A production-grade ML prediction service for credit card fraud detection. Built with FastAPI, Docker, CI/CD (GitHub Actions), and Kubernetes — wrapping an MLP neural network with a statistical safety rail.

> **Origin:** This project extends [FraudGuard](https://github.com/siddhjain-us/creditCardFraudDetection), transforming a Streamlit ML prototype into a deployable, containerized microservice with production infrastructure.

---

## How it works

A transaction comes in through the REST API. The service runs it through a trained MLP neural network (32→16→8 neurons, ReLU activation) that was trained on 455,000 SMOTE-balanced transactions. Before returning the result, a **2.2σ statistical safety rail** checks whether the transaction amount falls in the extreme top 2% of the historical distribution — if it does, the system overrides the neural network and flags it as high risk regardless.

Predictions are cached in Redis (5-minute TTL) for repeated inputs and logged to PostgreSQL for audit history.

### API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Predict fraud probability for a transaction |
| `GET` | `/api/v1/history` | Retrieve logged prediction history |
| `GET` | `/api/v1/model/info` | Model metadata and statistical thresholds |
| `GET` | `/health` | Kubernetes liveness/readiness probe |

### Sample request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.00, "v14": -1.5, "v17": -0.8}'
```

```json
{
  "fraud_probability": 0.0342,
  "risk_level": "LOW",
  "flags": [],
  "z_score_amount": 0.1523,
  "cached": false,
  "timestamp": "2026-03-30T22:15:00Z"
}
```

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              GitHub Actions CI/CD             │
                    │   lint (ruff) → test (pytest) → build/push   │
                    └──────────────────┬───────────────────────────┘
                                       │ Docker image
                                       ▼
┌─────────┐     ┌──────────────────────────────────────┐     ┌───────────┐
│  Client  │────▶│         FastAPI Application          │────▶│   Redis   │
│  (curl/  │     │                                      │     │  (cache)  │
│  browser)│◀────│  MLP Neural Net + 2.2σ Safety Rail   │     └───────────┘
└─────────┘     └──────────────┬───────────────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ PostgreSQL  │
                        │  (logging)  │
                        └─────────────┘
```

All three services (API, PostgreSQL, Redis) are containerized and orchestrated with Docker Compose locally or Kubernetes in production.

---

## Tech stack

**Application:** Python, FastAPI, scikit-learn (MLP), NumPy, Pydantic

**Infrastructure:** Docker (multi-stage build), Docker Compose, Kubernetes (Deployment, Service, ConfigMap, Secrets)

**CI/CD:** GitHub Actions (lint → test → build → push to Docker Hub)

**Data layer:** PostgreSQL (asyncpg), Redis (async caching)

**Testing:** pytest, ruff, mypy

---

## Run locally

### Option 1: Docker Compose (recommended)

```bash
git clone https://github.com/siddhjain-us/fraudguard-api.git
cd fraudguard-api
docker-compose up --build
```

Open http://localhost:8000/docs for the interactive API.

### Option 2: Python directly

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --reload
```

Note: PostgreSQL and Redis features are disabled without Docker. Predictions still work.

### Run tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest
```

---

## Deploy to Kubernetes

```bash
# Start a local cluster
minikube start

# Deploy
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Access the service
minikube service fraudguard-api --url
```

The deployment runs 2 replicas with rolling updates (zero-downtime), liveness/readiness probes, and resource limits.

---

## CI/CD Pipeline

Every push to `main` triggers:

1. **Lint** — ruff checks code quality
2. **Test** — pytest runs 8 integration tests with coverage
3. **Build** — multi-stage Docker image built and pushed to Docker Hub
4. **Deploy** — (when configured) rolling update to Kubernetes cluster

---

## Model details

| Property | Value |
|----------|-------|
| Algorithm | MLP Neural Network (scikit-learn) |
| Architecture | 3 layers: 32 → 16 → 8 neurons |
| Activation | ReLU + Sigmoid output |
| Training data | 455,000 transactions (SMOTE-balanced) |
| Recall | 0.94 |
| Safety rail | 2.2σ statistical override for extreme amounts |

The 2.2σ rule catches transactions in the top ~2% of the historical distribution, protecting against cases where the neural network hasn't seen sufficiently large amounts during training.

---

## Project structure

```
fraudguard-api/
├── .github/workflows/
│   └── ci-cd.yml            # GitHub Actions pipeline
├── app/
│   ├── api/routes.py         # API endpoints
│   ├── db/database.py        # PostgreSQL connection & logging
│   ├── db/cache.py           # Redis caching layer
│   ├── ml/model.py           # Model loading & prediction logic
│   └── main.py               # FastAPI application entry point
├── k8s/
│   ├── deployment.yaml       # Kubernetes Deployment (2 replicas)
│   ├── service.yaml          # Kubernetes Service (LoadBalancer)
│   ├── configmap.yaml        # Environment configuration
│   └── secrets.yaml          # Database credentials
├── models/                   # Trained model artifacts (.pkl)
├── tests/test_api.py         # Integration tests
├── Dockerfile                # Multi-stage production build
├── docker-compose.yml        # Local dev stack (API + PG + Redis)
└── requirements.txt
```
