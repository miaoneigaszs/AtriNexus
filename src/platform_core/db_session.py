from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.platform_core.database import Session
from src.platform_core.database_async import AsyncSessionLocal


def new_session(**kwargs):
    """统一的同步会话入口，便于后续替换底层 session factory。"""
    return Session(**kwargs)


@contextmanager
def session_scope(**kwargs) -> Iterator:
    """提供统一的同步会话上下文。"""
    session = Session(**kwargs)
    try:
        yield session
    finally:
        session.close()


def get_async_session_factory():
    """返回 PostgreSQL 异步 session factory。"""
    return AsyncSessionLocal
