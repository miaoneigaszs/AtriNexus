"""
异步工具模块
提供线程池执行器，将同步 I/O 操作异步化
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Callable

from data.config import config

logger = logging.getLogger('wecom')

T = TypeVar('T')

# 全局线程池执行器（用于 ChromaDB 等同步库的异步包装）
_executor = ThreadPoolExecutor(
    max_workers=config.system_performance.thread_pool.max_workers,
    thread_name_prefix="async_io_"
)


async def run_sync(func: Callable[..., T], *args, **kwargs) -> T:
    """
    在线程池中运行同步函数，不阻塞事件循环

    Args:
        func: 同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Example:
        result = await run_sync(collection.query, query_texts=[query])
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: func(*args, **kwargs)
    )
