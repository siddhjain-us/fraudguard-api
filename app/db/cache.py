"""
Redis caching layer for prediction results.
Caches identical transactions to avoid redundant model inference.
"""

import os
import json

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

_redis = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


async def get_redis():
    """Lazy-init Redis connection."""
    global _redis
    if aioredis is None:
        return None
    if _redis is None:
        try:
            _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await _redis.ping()
        except Exception as e:
            print(f"WARNING: Redis unavailable: {e}")
            _redis = None
    return _redis


async def get_cached(key: str) -> dict | None:
    """Get a cached prediction result."""
    r = await get_redis()
    if r is None:
        return None
    try:
        data = await r.get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


async def set_cached(key: str, value: dict, ttl: int = 300):
    """Cache a prediction result with TTL (default 5 minutes)."""
    r = await get_redis()
    if r is None:
        return
    try:
        await r.set(key, json.dumps(value), ex=ttl)
    except Exception:
        pass
