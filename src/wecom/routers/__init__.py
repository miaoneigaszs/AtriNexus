"""
路由模块
按功能域拆分的 API 路由
"""

from src.wecom.routers.callback import router as callback_router
from src.wecom.routers.knowledge import router as knowledge_router
from src.wecom.routers.memory import router as memory_router
from src.wecom.routers.config import router as config_router
from src.wecom.routers.tasks import router as tasks_router
from src.wecom.routers.diary import router as diary_router
from src.wecom.routers.token import router as token_router
from src.wecom.routers.agent import router as agent_router

__all__ = [
    'callback_router',
    'knowledge_router',
    'memory_router',
    'config_router',
    'tasks_router',
    'diary_router',
    'token_router',
    'agent_router',
]
