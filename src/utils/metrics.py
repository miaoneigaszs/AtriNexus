"""
Prometheus 指标收集模块
提供业务指标和系统指标的收集能力
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional, Any

logger = logging.getLogger('wecom')

# 检查 prometheus_client 是否可用
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Info = None
    CollectorRegistry = None


# ========== 指标定义 ==========

class Metrics:
    """
    指标集合类
    统一管理所有 Prometheus 指标
    """
    
    _initialized = False
    
    # LLM 相关指标
    llm_requests_total: Optional[Counter] = None
    llm_request_duration: Optional[Histogram] = None
    llm_tokens_total: Optional[Counter] = None
    llm_errors_total: Optional[Counter] = None
    
    # RAG 相关指标
    rag_retrieval_duration: Optional[Histogram] = None
    rag_retrieval_results: Optional[Histogram] = None
    rag_document_count: Optional[Gauge] = None
    
    # 记忆相关指标
    memory_operation_duration: Optional[Histogram] = None
    memory_vector_count: Optional[Gauge] = None
    
    # HTTP 相关指标
    http_requests_total: Optional[Counter] = None
    http_request_duration: Optional[Histogram] = None
    
    # 系统指标
    active_users: Optional[Gauge] = None
    active_sessions: Optional[Gauge] = None
    system_info: Optional[Info] = None
    
    # Token 监测指标（从 token_monitor 迁移过来）
    token_by_type_total: Optional[Counter] = None
    token_by_user_total: Optional[Counter] = None
    token_cost_estimate_usd: Optional[Counter] = None
    active_llm_requests: Optional[Gauge] = None
    
    @classmethod
    def initialize(cls, registry: CollectorRegistry = None):
        """初始化所有指标"""
        if cls._initialized:
            return
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client 未安装，指标收集不可用")
            return
        
        if registry is None:
            from prometheus_client import REGISTRY
            registry = REGISTRY
        
        # LLM 指标
        cls.llm_requests_total = Counter(
            'atrinexus_llm_requests_total',
            'Total LLM API requests',
            ['model', 'status'],
            registry=registry
        )
        
        cls.llm_request_duration = Histogram(
            'atrinexus_llm_request_duration_seconds',
            'LLM API request latency',
            ['model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=registry
        )
        
        cls.llm_tokens_total = Counter(
            'atrinexus_llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'type'],  # type: prompt/completion
            registry=registry
        )
        
        cls.llm_errors_total = Counter(
            'atrinexus_llm_errors_total',
            'Total LLM errors',
            ['model', 'error_type'],
            registry=registry
        )
        
        # RAG 指标
        cls.rag_retrieval_duration = Histogram(
            'atrinexus_rag_retrieval_duration_seconds',
            'RAG retrieval latency',
            ['user_id'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=registry
        )
        
        cls.rag_retrieval_results = Histogram(
            'atrinexus_rag_retrieval_results',
            'Number of RAG retrieval results',
            ['user_id'],
            buckets=[1, 3, 5, 10, 20, 50],
            registry=registry
        )
        
        cls.rag_document_count = Gauge(
            'atrinexus_rag_document_count',
            'Number of documents in RAG knowledge base',
            ['user_id', 'category'],
            registry=registry
        )
        
        # 记忆指标
        cls.memory_operation_duration = Histogram(
            'atrinexus_memory_operation_duration_seconds',
            'Memory operation latency',
            ['operation', 'memory_type'],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0],
            registry=registry
        )
        
        cls.memory_vector_count = Gauge(
            'atrinexus_memory_vector_count',
            'Number of vectors in memory storage',
            ['user_id', 'avatar'],
            registry=registry
        )
        
        # HTTP 指标
        cls.http_requests_total = Counter(
            'atrinexus_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=registry
        )
        
        cls.http_request_duration = Histogram(
            'atrinexus_http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=registry
        )
        
        # 系统指标
        cls.active_users = Gauge(
            'atrinexus_active_users',
            'Number of active users',
            registry=registry
        )
        
        cls.active_sessions = Gauge(
            'atrinexus_active_sessions',
            'Number of active sessions',
            registry=registry
        )
        
        cls.system_info = Info(
            'atrinexus_system',
            'System information',
            registry=registry
        )
        
        # 记忆相关指标
        cls.memory_short_entries = Gauge(
            'atrinexus_memory_short_entries',
            'Number of short-term memory entries by user',
            ['user_id', 'avatar_name'],
            registry=registry
        )
        
        cls.memory_vector_entries = Gauge(
            'atrinexus_memory_vector_entries',
            'Number of vector memory entries by user',
            ['user_id', 'avatar_name'],
            registry=registry
        )
        
        cls.memory_operations_total = Counter(
            'atrinexus_memory_operations_total',
            'Total memory operations',
            ['operation', 'memory_type'],
            registry=registry
        )
        
        # Token 监测指标
        cls.token_by_type_total = Counter(
            'atrinexus_token_by_type_total',
            'Total tokens by request type (chat, intent, rag, etc.)',
            ['request_type', 'token_type'],
            registry=registry
        )
        
        cls.token_by_user_total = Counter(
            'atrinexus_token_by_user_total',
            'Total tokens by user',
            ['user_id', 'token_type'],
            registry=registry
        )
        
        cls.token_cost_estimate_usd = Counter(
            'atrinexus_token_cost_estimate_usd',
            'Estimated token cost in USD',
            ['model'],
            registry=registry
        )
        
        cls.active_llm_requests = Gauge(
            'atrinexus_active_llm_requests',
            'Number of active LLM requests',
            ['model', 'request_type'],
            registry=registry
        )
        
        cls._initialized = True
        logger.info("Prometheus 指标初始化完成")


# ========== 便捷函数 ==========

def metrics_timer(metric: Histogram, **labels):
    """
    装饰器：自动记录函数执行时间
    
    Example:
        @metrics_timer(Metrics.rag_retrieval_duration, user_id='default')
        def retrieve_knowledge(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE or metric is None:
                return func(*args, **kwargs)
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                metric.labels(**labels).observe(time.time() - start)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE or metric is None:
                return await func(*args, **kwargs)
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                metric.labels(**labels).observe(time.time() - start)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def record_llm_request(model: str, status: str, duration: float, 
                       prompt_tokens: int = 0, completion_tokens: int = 0):
    """记录 LLM 请求指标"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    if Metrics.llm_requests_total:
        Metrics.llm_requests_total.labels(model=model, status=status).inc()
    
    if Metrics.llm_request_duration and status == "success":
        Metrics.llm_request_duration.labels(model=model).observe(duration)
    
    if Metrics.llm_tokens_total:
        if prompt_tokens > 0:
            Metrics.llm_tokens_total.labels(model=model, type='prompt').inc(prompt_tokens)
        if completion_tokens > 0:
            Metrics.llm_tokens_total.labels(model=model, type='completion').inc(completion_tokens)


def record_rag_retrieval(user_id: str, duration: float, result_count: int):
    """记录 RAG 检索指标"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    if Metrics.rag_retrieval_duration:
        Metrics.rag_retrieval_duration.labels(user_id=user_id).observe(duration)
    
    if Metrics.rag_retrieval_results:
        Metrics.rag_retrieval_results.labels(user_id=user_id).observe(result_count)


def record_memory_operation(operation: str, memory_type: str, duration: float):
    """记录记忆操作指标"""
    if not PROMETHEUS_AVAILABLE or not Metrics.memory_operation_duration:
        return
    
    Metrics.memory_operation_duration.labels(
        operation=operation, 
        memory_type=memory_type
    ).observe(duration)


def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """记录 HTTP 请求指标"""
    if not PROMETHEUS_AVAILABLE:
        return
    
    if Metrics.http_requests_total:
        Metrics.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status)
        ).inc()
    
    if Metrics.http_request_duration:
        Metrics.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)


def get_metrics_output() -> bytes:
    """获取 Prometheus 指标输出（用于 /metrics 端点）"""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n"
    
    return generate_latest()


def get_metrics_content_type() -> str:
    """获取指标内容类型"""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain"


# 初始化指标
Metrics.initialize()

# 输出初始化状态
if PROMETHEUS_AVAILABLE:
    logger.info(f"Prometheus 指标已初始化: PROMETHEUS_AVAILABLE={PROMETHEUS_AVAILABLE}, Metrics._initialized={Metrics._initialized}")
else:
    logger.warning("prometheus_client 未安装，自定义指标不可用")
