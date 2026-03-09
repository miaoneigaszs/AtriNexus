"""
中间件模块
"""

from src.wecom.middleware.dedup_middleware import DedupMiddleware

__all__ = [
    'DedupMiddleware',
]
