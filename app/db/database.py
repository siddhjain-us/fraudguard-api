"""
PostgreSQL connection and prediction logging.
Uses asyncpg for async database operations.
"""

import os
import json
from datetime import datetime, timezone

# Optional: gracefully degrade if asyncpg not available (for testing)
try:
    import asyncpg
except ImportError:
    asyncpg = None

_pool = None

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraudguard:fraudguard@localhost:5432/fraudguard"
)


async def init_db():
    """Initialize connection pool and create tables."""
    global _pool
    if asyncpg is None:
        print("WARNING: asyncpg not installed. DB logging disabled.")
        return

    try:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    input_data JSONB NOT NULL,
                    result JSONB NOT NULL,
                    risk_level VARCHAR(10) NOT NULL,
                    fraud_probability FLOAT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_created
                    ON predictions (created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_risk
                    ON predictions (risk_level);
            """)
    except Exception as e:
        print(f"WARNING: Could not connect to PostgreSQL: {e}")
        _pool = None


async def close_db():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def log_prediction(input_data: dict, result: dict):
    """Log a prediction to PostgreSQL."""
    if _pool is None:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions (input_data, result, risk_level, fraud_probability)
                VALUES ($1, $2, $3, $4)
                """,
                json.dumps(input_data),
                json.dumps(result),
                result.get("risk_level", "UNKNOWN"),
                result.get("fraud_probability", 0.0),
            )
    except Exception as e:
        print(f"WARNING: Failed to log prediction: {e}")


async def get_predictions(limit: int = 20, offset: int = 0) -> list[dict]:
    """Retrieve recent predictions."""
    if _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, input_data, result, risk_level,
                       fraud_probability, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset,
            )
            return [
                {
                    "id": r["id"],
                    "input": json.loads(r["input_data"]),
                    "result": json.loads(r["result"]),
                    "risk_level": r["risk_level"],
                    "fraud_probability": r["fraud_probability"],
                    "created_at": r["created_at"].isoformat(),
                }
                for r in rows
            ]
    except Exception as e:
        print(f"WARNING: Failed to fetch predictions: {e}")
        return []
