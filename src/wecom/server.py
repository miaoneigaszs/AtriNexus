"""
WeCom FastAPI 服务入口
职责:
- 创建 FastAPI 应用
- 配置中间件
- 注册路由
- 启动服务
"""

import logging
import uvicorn
import os
import time
import sys
import io
from logging.handlers import RotatingFileHandler
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# ========== 日志配置 ==========
# 强制 stdout 使用 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保日志目录存在
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# 获取日志文件路径（按日期）
log_file = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log")

# 配置根日志器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger('wecom')

# ========== 创建 FastAPI 应用 ==========
app = FastAPI(title="AtriNexus-WeCom", version="1.0.0")


# ========== 中间件配置 ==========

# API 鉴权配置
_API_KEY = os.getenv("API_KEY")
_SKIP_AUTH_PATHS = {
    "/api/wechat/callback",
    "/health", "/health/simple",
    "/metrics", "/metrics/debug",
    "/kb-upload", "/memory", "/setting",
}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """请求指标收集中间件"""
    from src.utils.metrics import record_http_request
    
    start_time = time.time()
    response = await call_next(request)
    
    latency = time.time() - start_time
    record_http_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=latency
    )
    
    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    API Key 鉴权中间件（可选）
    
    启用方式：设置环境变量 API_KEY
    - Header: X-API-Key
    - Query: ?api_key=xxx
    """
    if not _API_KEY:
        return await call_next(request)
    
    if request.url.path in _SKIP_AUTH_PATHS:
        return await call_next(request)
    
    if request.url.path.startswith("/static/"):
        return await call_next(request)
    
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key != _API_KEY:
        logger.warning(f"[Auth] 鉴权失败: path={request.url.path}")
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    
    return await call_next(request)


# ========== 健康检查端点 ==========

@app.get("/health")
async def health_check():
    """增强版健康检查"""
    from src.utils.health_check import get_health_report
    return await get_health_report()


@app.get("/health/simple")
def health_check_simple():
    """简单健康检查"""
    return {"status": "ok", "service": "AtriNexus-WeCom"}


# ========== Prometheus 指标端点 ==========

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    from src.utils.metrics import get_metrics_output, get_metrics_content_type
    return Response(
        content=get_metrics_output(),
        media_type=get_metrics_content_type()
    )


@app.get("/metrics/debug")
async def metrics_debug():
    """指标调试端点（生产环境禁用）"""
    from src.utils.metrics import Metrics, PROMETHEUS_AVAILABLE
    
    if os.getenv("ENV", "development") == "production":
        return JSONResponse(status_code=403, content={"error": "Debug endpoint disabled in production"})
    
    debug_info = {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_initialized": Metrics._initialized,
        "metrics": {}
    }
    
    if PROMETHEUS_AVAILABLE:
        metrics_list = [
            "llm_requests_total", "llm_request_duration", "llm_tokens_total",
            "llm_errors_total", "rag_retrieval_duration", "rag_retrieval_results",
            "rag_document_count", "memory_operation_duration", "memory_vector_count",
            "http_requests_total", "http_request_duration", "active_users",
            "active_sessions", "system_info", "token_by_type_total",
            "token_by_user_total", "token_cost_estimate_usd", "active_llm_requests"
        ]
        
        for metric_name in metrics_list:
            metric = getattr(Metrics, metric_name, None)
            debug_info["metrics"][metric_name] = {
                "registered": metric is not None,
                "type": type(metric).__name__ if metric else None
            }
    
    return JSONResponse(content=debug_info)


# ========== 注册路由 ==========
from src.wecom.routers import (
    callback_router,
    knowledge_router,
    memory_router,
    config_router,
    tasks_router,
    diary_router,
    token_router,
)

app.include_router(callback_router)
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(config_router)
app.include_router(tasks_router)
app.include_router(diary_router)
app.include_router(token_router)


# ========== 启动函数 ==========

def start_server(host: str = "0.0.0.0", port: int = 8080):
    """启动服务"""
    logger.info(f"AtriNexus-WeCom 服务启动中... http://{host}:{port}")
    
    # 启动定时任务调度器
    from src.wecom.scheduler import init_scheduler
    init_scheduler()
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
