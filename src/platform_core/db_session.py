from __future__ import annotations

from src.platform_core.database import Session


def new_session(**kwargs):
    """创建同步 SQLAlchemy 会话。"""
    return Session(**kwargs)
