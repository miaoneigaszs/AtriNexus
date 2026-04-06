from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_sqlite_path() -> Path:
    return get_project_root() / "data" / "database" / "chat_history.db"


def build_sync_database_url() -> str:
    """
    统一构建当前同步 ORM 使用的数据库 URL。

    当前线上仍默认 SQLite，但这里已经为 PostgreSQL 迁移留出入口。
    """
    raw_url = (os.getenv("ATRINEXUS_DATABASE_URL") or "").strip()
    if not raw_url:
        sqlite_path = get_default_sqlite_path()
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{sqlite_path}"

    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw_url


def build_async_database_url() -> str | None:
    """
    构建未来 AsyncSession 使用的数据库 URL。

    这里明确只为 PostgreSQL 异步链路准备，不引入 aiosqlite 作为长期方案。
    """
    raw_url = (os.getenv("ATRINEXUS_DATABASE_URL") or "").strip()
    if not raw_url:
        return None

    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url
    if raw_url.startswith("postgresql+psycopg://"):
        return raw_url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return None


def is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite:")


def build_sync_engine_kwargs(database_url: str) -> dict:
    kwargs = {"future": True}
    if not is_sqlite_url(database_url):
        kwargs["pool_pre_ping"] = True
    return kwargs
