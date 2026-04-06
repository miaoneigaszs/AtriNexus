from __future__ import annotations

import os


def _require_database_url() -> str:
    raw_url = (os.getenv("ATRINEXUS_DATABASE_URL") or "").strip()
    if not raw_url:
        raise RuntimeError("ATRINEXUS_DATABASE_URL is required. SQLite fallback has been removed.")
    return raw_url


def build_sync_database_url() -> str:
    """构建 PostgreSQL 同步 ORM URL。"""
    raw_url = _require_database_url()

    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql+psycopg://"):
        return raw_url
    raise RuntimeError(f"Unsupported database URL scheme for sync engine: {raw_url}")


def build_async_database_url() -> str:
    """构建 PostgreSQL 异步 ORM URL。"""
    raw_url = _require_database_url()

    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if raw_url.startswith("postgresql+psycopg://"):
        return raw_url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    raise RuntimeError(f"Unsupported database URL scheme for async engine: {raw_url}")


def build_sync_engine_kwargs() -> dict:
    return {
        "future": True,
        "pool_pre_ping": True,
    }
