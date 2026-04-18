from .callback import router as callback_router
from .config import router as config_router
from .diary import router as diary_router
from .knowledge import router as knowledge_router
from .memory import router as memory_router
from .tasks import router as tasks_router
from .token import router as token_router

__all__ = [
    "callback_router",
    "config_router",
    "diary_router",
    "knowledge_router",
    "memory_router",
    "tasks_router",
    "token_router",
]
