from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.auth import router as auth_router
from app.api.matches import router as matches_router
from app.api.analytics import router as analytics_router
from app.core.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield


app = FastAPI(
    title="ShuttleVision API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(matches_router, prefix="/matches", tags=["matches"])
app.include_router(analytics_router, prefix="/matches", tags=["analytics"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "shuttlevision-api"}


@app.get("/ready")
async def ready():
    """Readiness checks real dependencies; liveness stays lightweight."""
    import asyncio
    import shutil
    from fastapi.responses import JSONResponse
    from sqlalchemy import text
    from redis.asyncio import Redis
    from app.core.database import engine
    from app.core.config import settings
    from app.services.storage.local_storage import media_root

    async def dependencies():
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        async with Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2) as redis:
            await redis.ping()
        if settings.use_local_storage and shutil.disk_usage(media_root()).free < 1024**3:
            raise RuntimeError("storage reserve")

    try:
        await asyncio.wait_for(dependencies(), timeout=5)
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ready"}
