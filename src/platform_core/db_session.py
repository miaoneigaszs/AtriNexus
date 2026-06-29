from __future__ import annotations

from src.platform_core.database import Session


def new_session(**kwargs):
    """Create a synchronous SQLAlchemy session."""
    return Session(**kwargs)
