"""
健康检查模块
提供系统各组件的健康状态检查
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

import httpx

from src.utils.http_pool import get_async_client
from data.config import config

logger = logging.getLogger('wecom')


class HealthStatus:
    """健康状态常量"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResult:
    """健康检查结果"""
    
    def __init__(
        self, 
        name: str, 
        status: str, 
        latency_ms: float = 0,
        message: str = "",
        details: Dict[str, Any] = None
    ):
        self.name = name
        self.status = status
        self.latency_ms = latency_ms
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            **self.details
        }


class HealthChecker:
    """
    健康检查器
    检查系统各组件的运行状态
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    def register(self, name: str, check_func: Callable):
        """注册健康检查函数"""
        self._checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """运行单个健康检查"""
        check_func = self._checks.get(name)
        if not check_func:
            return HealthCheckResult(name, HealthStatus.UNHEALTHY, message="Check not found")
        
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            if isinstance(result, HealthCheckResult):
                result.latency_ms = (time.time() - start) * 1000
                self._last_results[name] = result
                return result
            elif isinstance(result, dict):
                latency = (time.time() - start) * 1000
                health_result = HealthCheckResult(
                    name=name,
                    status=result.get("status", HealthStatus.HEALTHY),
                    latency_ms=latency,
                    message=result.get("message", ""),
                    details=result.get("details")
                )
                self._last_results[name] = health_result
                return health_result
            else:
                return HealthCheckResult(
                    name, HealthStatus.HEALTHY, 
                    latency_ms=(time.time() - start) * 1000
                )
                
        except Exception as e:
            return HealthCheckResult(
                name, HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e)
            )
    
    async def run_all(self) -> Dict[str, HealthCheckResult]:
        """运行所有健康检查"""
        results = {}
        for name in self._checks:
            results[name] = await self.run_check(name)
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """获取整体健康状态"""
        if not results:
            return HealthStatus.UNHEALTHY
        
        statuses = [r.status for r in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


# 全局健康检查器
health_checker = HealthChecker()


# ========== 内置健康检查函数 ==========

async def check_database() -> HealthCheckResult:
    """检查数据库连接"""
    start = time.time()
    try:
        from src.services.database import Session
        from sqlalchemy import text
        
        # 尝试执行简单查询
        session = Session()
        try:
            session.execute(text("SELECT 1"))
            session.commit()
        finally:
            session.close()
        
        latency = (time.time() - start) * 1000
        return HealthCheckResult(
            "database",
            HealthStatus.HEALTHY,
            latency_ms=latency,
            details={"type": "SQLite"}
        )
    except Exception as e:
        return HealthCheckResult(
            "database",
            HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def check_qdrant() -> HealthCheckResult:
    """检查 Qdrant 连接"""
    start = time.time()
    try:
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("ATRINEXUS_QDRANT_URL", "").strip() or None
        qdrant_api_key = os.getenv("ATRINEXUS_QDRANT_API_KEY", "").strip() or None
        qdrant_path = os.getenv("ATRINEXUS_QDRANT_PATH", "data/vectordb_qdrant")

        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(path=qdrant_path)

        collections = client.get_collections().collections
        
        latency = (time.time() - start) * 1000
        return HealthCheckResult(
            "qdrant",
            HealthStatus.HEALTHY,
            latency_ms=latency,
            details={"collections": len(collections)}
        )
    except Exception as e:
        return HealthCheckResult(
            "qdrant",
            HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def check_llm_api() -> HealthCheckResult:
    """检查 LLM API 连接"""
    start = time.time()
    
    api_key = config.llm.api_key
    base_url = config.llm.base_url
    
    if not api_key:
        return HealthCheckResult(
            "llm_api",
            HealthStatus.DEGRADED,
            message="API key not configured"
        )
    
    try:
        client = get_async_client()
        
        # 发送最小化请求测试连通性
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = await client.get(url, headers=headers, timeout=5.0)
        
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            return HealthCheckResult(
                "llm_api",
                HealthStatus.HEALTHY,
                latency_ms=latency,
                details={"base_url": base_url}
            )
        else:
            return HealthCheckResult(
                "llm_api",
                HealthStatus.DEGRADED,
                latency_ms=latency,
                message=f"API returned status {response.status_code}"
            )
    except Exception as e:
        return HealthCheckResult(
            "llm_api",
            HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def check_embedding_api() -> HealthCheckResult:
    """检查 Embedding API 连接"""
    start = time.time()
    
    api_key = config.llm.api_key
    base_url = getattr(config.llm, 'base_url', 'https://api.siliconflow.cn/v1')
    
    if not api_key:
        return HealthCheckResult(
            "embedding_api",
            HealthStatus.DEGRADED,
            message="API key not configured"
        )
    
    try:
        client = get_async_client()
        
        # 发送最小化 embedding 请求
        url = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "BAAI/bge-large-zh-v1.5",
            "input": ["test"]
        }
        
        response = await client.post(url, headers=headers, json=payload, timeout=10.0)
        
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            return HealthCheckResult(
                "embedding_api",
                HealthStatus.HEALTHY,
                latency_ms=latency
            )
        else:
            return HealthCheckResult(
                "embedding_api",
                HealthStatus.DEGRADED,
                latency_ms=latency,
                message=f"API returned status {response.status_code}"
            )
    except Exception as e:
        return HealthCheckResult(
            "embedding_api",
            HealthStatus.DEGRADED,
            latency_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def check_memory() -> HealthCheckResult:
    """检查内存使用情况"""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        details = {
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
        
        # 判断状态
        if memory.percent > 90 or disk.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = "Resource usage critical"
        elif memory.percent > 80 or disk.percent > 80:
            status = HealthStatus.DEGRADED
            message = "Resource usage high"
        else:
            status = HealthStatus.HEALTHY
            message = "Resources OK"
        
        return HealthCheckResult("memory", status, message=message, details=details)
    except ImportError:
        return HealthCheckResult(
            "memory",
            HealthStatus.HEALTHY,
            message="psutil not installed, skipping"
        )
    except Exception as e:
        return HealthCheckResult(
            "memory",
            HealthStatus.DEGRADED,
            message=str(e)
        )


# 注册默认健康检查
def register_default_checks():
    """注册默认的健康检查"""
    health_checker.register("database", check_database)
    health_checker.register("qdrant", check_qdrant)
    health_checker.register("llm_api", check_llm_api)
    health_checker.register("embedding_api", check_embedding_api)
    health_checker.register("memory", check_memory)


async def get_health_report() -> Dict[str, Any]:
    """
    获取完整的健康报告
    
    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": "2026-02-23T...",
            "checks": {
                "database": {"status": "healthy", ...},
                "qdrant": {"status": "healthy", ...},
                ...
            }
        }
    """
    results = await health_checker.run_all()
    overall = health_checker.get_overall_status(results)
    
    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "checks": {name: result.to_dict() for name, result in results.items()}
    }


# 初始化时注册默认检查
register_default_checks()
