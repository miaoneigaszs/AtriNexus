from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.platform_core.database_config import build_async_database_url


ASYNC_DATABASE_URL = build_async_database_url()
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)
