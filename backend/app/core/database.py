from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def create_tables():
    from sqlalchemy import text
    from app.models import analysis_job, match, media_deletion, user  # noqa: F401 — register metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all never alters existing tables; add v2 columns idempotently.
        for ddl in (
            "ALTER TABLE matches ADD COLUMN IF NOT EXISTS fps DOUBLE PRECISION DEFAULT 60.0",
            "ALTER TABLE matches ADD COLUMN IF NOT EXISTS analysis JSON",
        ):
            await conn.execute(text(ddl))


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
