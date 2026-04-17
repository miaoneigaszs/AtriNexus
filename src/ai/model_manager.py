"""
LLM 模型管理模块
提供模型发现、切换功能

重构说明：
- 取消硬编码的模型过滤和排序逻辑
- 全部模型直接列出，由用户在设置页面挑选
- 备用模型列表来自配置文件（fallback_models）
"""

import logging
import requests
from typing import List, Dict, Optional
from openai import OpenAI

logger = logging.getLogger('main')


class ModelManager:
    """
    模型管理器
    负责：
    - 获取 API 提供商支持的全部模型列表（不做过滤）
    - 自动模型切换（候选来自用户配置的 fallback_models）
    - 动态刷新模型列表
    """
    
    def __init__(self, client: OpenAI, original_model: str, fallback_models: List[str] = None):
        """
        初始化模型管理器
        
        Args:
            client: OpenAI 客户端实例
            original_model: 配置的原始模型名称
            fallback_models: 用户配置的备用模型列表（auto_model_switch 时使用）
        """
        self.client = client
        self.original_model = original_model
        self.fallback_models = fallback_models or []
        self.ollama_models = self._get_ollama_models() if self._is_ollama() else []
        self.available_models = self._fetch_all_models()
        
        # 记录初始化结果
        logger.info(f"[ModelManager] 初始化完成 | 主模型: {original_model} | "
                     f"备用模型: {self.fallback_models} | "
                     f"API 可用模型数: {len(self.available_models)}")
        if self.available_models:
            logger.info(f"[ModelManager] 完整模型列表: {self.available_models}")
    
    def _is_ollama(self) -> bool:
        """检查是否为 Ollama 后端"""
        return 'localhost:11434' in str(self.client.base_url)
    
    def _get_ollama_models(self) -> List[Dict]:
        """获取本地 Ollama 可用的模型列表"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"[ModelManager] Ollama 模型数: {len(models)}")
                return [
                    {
                        "id": model['name'],
                        "name": model['name'],
                        "status": "active",
                        "type": "chat",
                        "context_length": 16000
                    }
                    for model in models
                ]
            return []
        except Exception as e:
            logger.error(f"[ModelManager] 获取 Ollama 模型列表失败: {str(e)}")
            return []
    
    def _fetch_all_models(self) -> List[str]:
        """
        通过 API 获取当前提供商支持的全部模型列表，不做任何过滤。
        如果 API 调用失败，返回主模型 + 配置的备用模型。
        """
        try:
            if self._is_ollama():
                models = [model['id'] for model in self.ollama_models]
                logger.info(f"[ModelManager] Ollama 模式，模型列表: {models}")
                return models
            
            logger.info(f"[ModelManager] 正在从 {self.client.base_url} 获取全部模型列表...")
            
            try:
                models_response = self.client.models.list()
                all_models = [model.id for model in models_response.data]
                
                if all_models:
                    logger.info(f"[ModelManager] API 返回 {len(all_models)} 个模型（无过滤）: {all_models}")
                    return all_models
                else:
                    logger.warning("[ModelManager] API 返回空模型列表，使用主模型 + 备用模型")
                    return self._get_configured_models()
                    
            except Exception as api_error:
                logger.warning(f"[ModelManager] API 查询模型列表失败: {str(api_error)}，使用主模型 + 备用模型")
                return self._get_configured_models()
                
        except Exception as e:
            logger.error(f"[ModelManager] 获取模型列表异常: {str(e)}")
            return [self.original_model]
    
    def _get_configured_models(self) -> List[str]:
        """
        当 API 不可用时，返回配置中的模型列表。
        包含主模型 + 用户配置的备用模型，去重并保持顺序。
        """
        models = [self.original_model]
        for m in self.fallback_models:
            if m not in models:
                models.append(m)
        logger.info(f"[ModelManager] 使用配置模型列表（API 不可用）: {models}")
        return models
    
    def refresh_models(self) -> List[str]:
        """
        重新查询 API 获取最新模型列表。
        供设置页面的 "刷新模型列表" 按钮调用。
        
        Returns:
            最新的全部模型 ID 列表
        """
        logger.info("[ModelManager] 手动触发模型列表刷新...")
        
        # 如果是 Ollama，也刷新 Ollama 模型
        if self._is_ollama():
            self.ollama_models = self._get_ollama_models()
        
        self.available_models = self._fetch_all_models()
        logger.info(f"[ModelManager] 刷新完成，共 {len(self.available_models)} 个模型")
        return self.available_models
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self.available_models
    
    def get_next_model(self, current_model: str) -> Optional[str]:
        """
        获取下一个可用的备用模型。
        优先从用户配置的 fallback_models 中选择，
        如果 fallback_models 为空则从 available_models 中轮换。
        
        Args:
            current_model: 当前正在使用的模型名称
            
        Returns:
            下一个可用的模型名称，如果没有可用的则返回 None
        """
        # 优先使用用户配置的备用模型列表
        candidates = self.fallback_models if self.fallback_models else self.available_models
        
        if not candidates:
            logger.warning("[ModelManager] 无可用候选模型")
            return None
        
        # 过滤掉当前模型
        remaining = [m for m in candidates if m != current_model]
        
        if not remaining:
            logger.warning(f"[ModelManager] 除当前模型 '{current_model}' 外无其他候选")
            return None
        
        next_model = remaining[0]
        logger.info(f"[ModelManager] 模型切换: {current_model} → {next_model} "
                     f"(候选列表: {remaining})")
        return next_model
    
    def get_ollama_models(self) -> List[Dict]:
        """获取 Ollama 模型列表（公共接口）"""
        return self.ollama_models
