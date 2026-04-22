"""Health check endpoints for liveness and readiness probes."""
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db

router = APIRouter()


@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe — app is running."""
    return {"status": "ok", "version": settings.APP_VERSION}


@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe — dependencies are reachable.
    Checks PostgreSQL connectivity.
    """
    checks = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
