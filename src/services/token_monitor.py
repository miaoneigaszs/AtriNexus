"""
Token 使用监测服务
跟踪和统计 LLM API 的 token 使用情况
支持 Prometheus 指标导出
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading
import time

logger = logging.getLogger('token_monitor')

# Prometheus 指标集成
try:
    from src.utils.metrics import Metrics, PROMETHEUS_AVAILABLE, record_llm_request
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Prometheus 指标导入失败: {e}")
    PROMETHEUS_AVAILABLE = False
    PROMETHEUS_CLIENT_AVAILABLE = False
    record_llm_request = None
    Counter = Gauge = Histogram = None


@dataclass
class TokenUsageRecord:
    """单次 token 使用记录"""
    timestamp: str
    user_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_type: str = "chat"  # chat, intent, rag, etc.
    # 详细 token 分解（可选）
    system_prompt_tokens: int = 0      # 人设+规则
    core_memory_tokens: int = 0        # 核心记忆
    kb_context_tokens: int = 0         # 知识库上下文
    chat_history_tokens: int = 0       # 历史对话
    user_message_tokens: int = 0       # 用户消息
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TokenMonitor:
    """Token 使用监测器（线程安全）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self._records: List[TokenUsageRecord] = []
        self._records_lock = threading.Lock()
        self._max_records = 10000  # 最多保存的记录数
        
        # 实时统计
        self._stats = defaultdict(lambda: {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0
        })
        self._stats_lock = threading.Lock()
        
        # 数据持久化路径
        self._data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'token_stats'
        )
        os.makedirs(self._data_dir, exist_ok=True)
        
        # 加载历史数据
        self._load_stats()
    
    def record(
        self,
        user_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_type: str = "chat",
        # 详细 token 分解（可选）
        system_prompt_tokens: int = 0,
        core_memory_tokens: int = 0,
        kb_context_tokens: int = 0,
        chat_history_tokens: int = 0,
        user_message_tokens: int = 0,
    ) -> TokenUsageRecord:
        """记录一次 token 使用"""
        total_tokens = prompt_tokens + completion_tokens
        timestamp = datetime.now().isoformat()
        
        record = TokenUsageRecord(
            timestamp=timestamp,
            user_id=user_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_type=request_type,
            system_prompt_tokens=system_prompt_tokens,
            core_memory_tokens=core_memory_tokens,
            kb_context_tokens=kb_context_tokens,
            chat_history_tokens=chat_history_tokens,
            user_message_tokens=user_message_tokens,
        )
        
        # 添加到记录列表
        with self._records_lock:
            self._records.append(record)
            # 超过最大记录数时，删除旧记录
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
        
        # 更新统计
        with self._stats_lock:
            # 按模型统计
            self._stats[f"model:{model}"]["total_prompt_tokens"] += prompt_tokens
            self._stats[f"model:{model}"]["total_completion_tokens"] += completion_tokens
            self._stats[f"model:{model}"]["total_tokens"] += total_tokens
            self._stats[f"model:{model}"]["request_count"] += 1
            
            # 按用户统计
            self._stats[f"user:{user_id}"]["total_prompt_tokens"] += prompt_tokens
            self._stats[f"user:{user_id}"]["total_completion_tokens"] += completion_tokens
            self._stats[f"user:{user_id}"]["total_tokens"] += total_tokens
            self._stats[f"user:{user_id}"]["request_count"] += 1
            
            # 按请求类型统计
            self._stats[f"type:{request_type}"]["total_prompt_tokens"] += prompt_tokens
            self._stats[f"type:{request_type}"]["total_completion_tokens"] += completion_tokens
            self._stats[f"type:{request_type}"]["total_tokens"] += total_tokens
            self._stats[f"type:{request_type}"]["request_count"] += 1
            
            # 总计
            self._stats["total"]["total_prompt_tokens"] += prompt_tokens
            self._stats["total"]["total_completion_tokens"] += completion_tokens
            self._stats["total"]["total_tokens"] += total_tokens
            self._stats["total"]["request_count"] += 1
        
        # 记录日志（简洁版）
        cost = self.estimate_cost(prompt_tokens, completion_tokens, model)
        detail_parts = []
        if system_prompt_tokens > 0:
            detail_parts.append(f"人设={system_prompt_tokens}")
        if core_memory_tokens > 0:
            detail_parts.append(f"记忆={core_memory_tokens}")
        if kb_context_tokens > 0:
            detail_parts.append(f"知识库={kb_context_tokens}")
        if chat_history_tokens > 0:
            detail_parts.append(f"历史={chat_history_tokens}")
        if user_message_tokens > 0:
            detail_parts.append(f"用户消息={user_message_tokens}")
        
        detail_str = " | ".join(detail_parts) if detail_parts else ""
        
        logger.info(
            f"[Token] {request_type} | {model} | "
            f"输入={prompt_tokens}, 输出={completion_tokens}, 合计={total_tokens} | "
            f"费用≈${cost:.6f}"
        )
        if detail_str:
            logger.info(f"[Token详情] {detail_str}")
        
        # 同步更新 Prometheus 指标
        if PROMETHEUS_AVAILABLE:
            try:
                # 使用现有的 record_llm_request 函数
                if record_llm_request:
                    record_llm_request(
                        model=model,
                        status="success",
                        duration=0,  # duration 不在这里记录
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )
                
                # 更新按请求类型的指标
                if Metrics.token_by_type_total:
                    Metrics.token_by_type_total.labels(request_type=request_type, token_type='prompt').inc(prompt_tokens)
                    Metrics.token_by_type_total.labels(request_type=request_type, token_type='completion').inc(completion_tokens)
                
                # 更新按用户的指标
                if Metrics.token_by_user_total:
                    Metrics.token_by_user_total.labels(user_id=user_id, token_type='prompt').inc(prompt_tokens)
                    Metrics.token_by_user_total.labels(user_id=user_id, token_type='completion').inc(completion_tokens)
                
                # 更新费用估算
                if Metrics.token_cost_estimate_usd:
                    cost = self.estimate_cost(prompt_tokens, completion_tokens, model)
                    Metrics.token_cost_estimate_usd.labels(model=model).inc(cost)
                    
            except Exception as e:
                logger.warning(f"更新 Prometheus 指标失败: {e}")
        
        return record
    
    def get_stats(self, key: str = "total") -> Dict[str, Any]:
        """获取统计数据"""
        with self._stats_lock:
            return dict(self._stats.get(key, {}))
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有统计数据"""
        with self._stats_lock:
            return {k: dict(v) for k, v in self._stats.items()}
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取按模型统计的数据"""
        with self._stats_lock:
            return {
                k.replace("model:", ""): dict(v)
                for k, v in self._stats.items()
                if k.startswith("model:")
            }
    
    def get_user_stats(self, user_id: str = None) -> Dict[str, Dict[str, Any]]:
        """获取按用户统计的数据"""
        with self._stats_lock:
            if user_id:
                return dict(self._stats.get(f"user:{user_id}", {}))
            return {
                k.replace("user:", ""): dict(v)
                for k, v in self._stats.items()
                if k.startswith("user:")
            }
    
    def get_recent_records(self, limit: int = 100) -> List[Dict]:
        """获取最近的记录"""
        with self._records_lock:
            return [r.to_dict() for r in self._records[-limit:]]
    
    def get_usage_summary(self, period_hours: int = 24) -> Dict[str, Any]:
        """获取指定时间段的使用摘要"""
        cutoff = datetime.now() - timedelta(hours=period_hours)
        
        summary = {
            "period_hours": period_hours,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "request_count": 0,
            "by_model": defaultdict(lambda: {"tokens": 0, "requests": 0}),
            "by_user": defaultdict(lambda: {"tokens": 0, "requests": 0}),
        }
        
        with self._records_lock:
            for record in self._records:
                try:
                    record_time = datetime.fromisoformat(record.timestamp)
                    if record_time >= cutoff:
                        summary["total_tokens"] += record.total_tokens
                        summary["total_prompt_tokens"] += record.prompt_tokens
                        summary["total_completion_tokens"] += record.completion_tokens
                        summary["request_count"] += 1
                        
                        summary["by_model"][record.model]["tokens"] += record.total_tokens
                        summary["by_model"][record.model]["requests"] += 1
                        
                        summary["by_user"][record.user_id]["tokens"] += record.total_tokens
                        summary["by_user"][record.user_id]["requests"] += 1
                except Exception as e:
                    logger.warning(f"解析时间失败: {e}")
        
        # 转换 defaultdict 为普通 dict
        summary["by_model"] = dict(summary["by_model"])
        summary["by_user"] = dict(summary["by_user"])
        
        return summary
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """估算费用（基于常见模型的价格）"""
        # 常见模型价格（每 1M tokens，美元）
        pricing = {
            # DeepSeek
            "deepseek-ai/DeepSeek-V3": {"prompt": 0.27, "completion": 1.10},
            "deepseek-ai/DeepSeek-V3.2": {"prompt": 0.27, "completion": 1.10},
            "Pro/deepseek-ai/DeepSeek-V3.2": {"prompt": 0.27, "completion": 1.10},
            "deepseek-ai/DeepSeek-R1": {"prompt": 0.55, "completion": 2.19},
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"prompt": 0.14, "completion": 0.28},
            # OpenAI
            "gpt-4o": {"prompt": 2.50, "completion": 10.00},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
            "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
            "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
            # Claude
            "claude-3-opus": {"prompt": 15.00, "completion": 75.00},
            "claude-3-sonnet": {"prompt": 3.00, "completion": 15.00},
            "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
        }
        
        # 查找匹配的价格
        model_pricing = None
        for model_name, price in pricing.items():
            if model_name.lower() in model.lower():
                model_pricing = price
                break
        
        # 默认价格（使用 DeepSeek V3 作为默认）
        if model_pricing is None:
            model_pricing = {"prompt": 0.27, "completion": 1.10}
        
        prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def get_cost_summary(self, period_hours: int = 24) -> Dict[str, Any]:
        """获取费用摘要"""
        summary = self.get_usage_summary(period_hours)
        
        total_cost = 0
        model_costs = {}
        
        with self._records_lock:
            cutoff = datetime.now() - timedelta(hours=period_hours)
            for record in self._records:
                try:
                    record_time = datetime.fromisoformat(record.timestamp)
                    if record_time >= cutoff:
                        cost = self.estimate_cost(
                            record.prompt_tokens,
                            record.completion_tokens,
                            record.model
                        )
                        total_cost += cost
                        
                        if record.model not in model_costs:
                            model_costs[record.model] = 0
                        model_costs[record.model] += cost
                except Exception:
                    pass
        
        summary["estimated_cost_usd"] = round(total_cost, 4)
        summary["cost_by_model_usd"] = {
            k: round(v, 4) for k, v in model_costs.items()
        }
        
        return summary
    
    def _load_stats(self):
        """从文件加载统计数据"""
        stats_file = os.path.join(self._data_dir, 'stats.json')
        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._stats[k] = v
                logger.info(f"已加载 token 统计数据")
        except Exception as e:
            logger.warning(f"加载 token 统计数据失败: {e}")
    
    def save_stats(self):
        """保存统计数据到文件"""
        stats_file = os.path.join(self._data_dir, 'stats.json')
        try:
            with self._stats_lock:
                data = {k: dict(v) for k, v in self._stats.items()}
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("已保存 token 统计数据")
        except Exception as e:
            logger.error(f"保存 token 统计数据失败: {e}")
    
    def reset_stats(self):
        """重置统计数据"""
        with self._stats_lock:
            self._stats.clear()
            self._stats["total"] = {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0
            }
        with self._records_lock:
            self._records.clear()
        logger.info("已重置 token 统计数据")


# 全局单例
token_monitor = TokenMonitor()
