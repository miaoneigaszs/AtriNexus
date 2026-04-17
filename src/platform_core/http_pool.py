"""
HTTP 连接池管理模块
提供共享的异步 HTTP 客户端，复用连接，提升性能
"""

import logging
from typing import Optional, Dict

import httpx

from data.config import config

logger = logging.getLogger('wecom')

# 全局异步 HTTP 客户端（懒加载）
_async_client: Optional[httpx.AsyncClient] = None
_sync_client: Optional[httpx.Client] = None

# 检测是否支持 HTTP/2
try:
    import h2
    HTTP2_SUPPORTED = True
except ImportError:
    HTTP2_SUPPORTED = False
    logger.debug("h2 package not installed, HTTP/2 support disabled")


def get_async_client() -> httpx.AsyncClient:
    """
    获取全局异步 HTTP 客户端（单例）
    
    Returns:
        httpx.AsyncClient: 异步 HTTP 客户端
    """
    global _async_client
    if _async_client is None or _async_client.is_closed:
        http_config = config.system_performance.http_client
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(http_config.timeout, connect=http_config.connect_timeout),
            limits=httpx.Limits(
                max_connections=http_config.max_connections,
                max_keepalive_connections=http_config.max_keepalive_connections,
                keepalive_expiry=http_config.keepalive_expiry
            ),
            http2=HTTP2_SUPPORTED  # 根据环境自动启用/禁用 HTTP/2
        )
        logger.debug(f"HTTP 异步连接池已创建 (HTTP/2: {HTTP2_SUPPORTED})")
    return _async_client


def get_sync_client() -> httpx.Client:
    """
    获取全局同步 HTTP 客户端（单例）
    
    Returns:
        httpx.Client: 同步 HTTP 客户端
    """
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        http_config = config.system_performance.http_client
        _sync_client = httpx.Client(
            timeout=httpx.Timeout(http_config.timeout, connect=http_config.connect_timeout),
            limits=httpx.Limits(
                max_connections=http_config.max_connections,
                max_keepalive_connections=http_config.max_keepalive_connections,
                keepalive_expiry=http_config.keepalive_expiry
            )
        )
        logger.debug("HTTP 同步连接池已创建")
    return _sync_client


async def close_async_client():
    """关闭异步客户端（应用退出时调用）"""
    global _async_client
    if _async_client and not _async_client.is_closed:
        await _async_client.aclose()
        _async_client = None
        logger.debug("HTTP 异步连接池已关闭")


def close_sync_client():
    """关闭同步客户端（应用退出时调用）"""
    global _sync_client
    if _sync_client and not _sync_client.is_closed:
        _sync_client.close()
        _sync_client = None
        logger.debug("HTTP 同步连接池已关闭")


def build_headers(api_key: str, include_version: bool = True) -> Dict[str, str]:
    """
    构建标准 API 请求头
    
    Args:
        api_key: API 密钥
        include_version: 是否包含版本信息（默认 True）
    
    Returns:
        Dict[str, str]: 标准请求头字典
    
    Example:
        >>> headers = build_headers("sk-xxx")
        >>> # {'Authorization': 'Bearer sk-xxx', 'Content-Type': 'application/json', ...}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if include_version:
        from src.platform_core.version import get_current_version, get_version_identifier
        headers["User-Agent"] = get_version_identifier()
        headers["X-AtriNexus-Version"] = get_current_version()
    
    return headers
