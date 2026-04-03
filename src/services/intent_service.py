"""
意图识别服务
智能判断用户消息是否需要查询知识库
使用更强的推理模型进行精准意图路由
"""

import json
import logging
from typing import Dict, Any, List, Optional

import httpx

from src.utils.http_pool import get_sync_client, get_async_client
from src.services.token_monitor import token_monitor
from data.config import config

logger = logging.getLogger('wecom')


class IntentService:
    """意图识别服务 - 智能语义路由"""

    DEFAULT_INTENT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def __init__(self, rag_engine=None):
        self.rag = rag_engine
        self._init_config()

    def _init_config(self):
        """初始化配置"""
        self.api_key = config.intent_recognition.api_key or config.llm.api_key
        self.base_url = config.intent_recognition.base_url or config.llm.base_url
        self.model = config.intent_recognition.model or self.DEFAULT_INTENT_MODEL
        self.temperature = getattr(config.intent_recognition, 'temperature', 0.1)
    
    # ========== 公共方法 ==========

    def recognize_intent(
        self, user_id: str, content: str,
        previous_context: List[Dict] = None, categories: List[str] = None
    ) -> Dict[str, Any]:
        """同步意图识别"""
        params = self._prepare_params(user_id, content, previous_context, categories)
        if not params:
            return self._default_result(content)
        return self._send_request_sync(params, content, user_id)

    async def recognize_intent_async(
        self, user_id: str, content: str,
        previous_context: List[Dict] = None, categories: List[str] = None
    ) -> Dict[str, Any]:
        """异步意图识别"""
        params = self._prepare_params(user_id, content, previous_context, categories)
        if not params:
            return self._default_result(content)
        return await self._send_request_async(params, content, user_id)
    
    # ========== 内部方法 ==========

    def _prepare_params(
        self, user_id: str, content: str,
        previous_context: List[Dict], categories: List[str]
    ) -> Optional[Dict]:
        """准备意图识别参数，返回 None 表示跳过识别"""
        # 获取知识库分类
        kb_list = self.rag.list_documents(user_id) if self.rag else {}
        if categories is None:
            categories = list(kb_list.keys()) if kb_list else []

        if not self.api_key or not categories:
            reason = "无API密钥" if not self.api_key else "知识库为空"
            logger.info(f"[意图识别] 跳过识别({reason}) -> 普通对话")
            return None

        category_descriptions = self._build_category_descriptions(categories, kb_list)
        context_str = self._format_context(previous_context)
        system_prompt = self._build_system_prompt(category_descriptions, context_str)

        return {
            "system_prompt": system_prompt
        }
    
    def _build_request(self, system_prompt: str, content: str):
        """构建请求参数"""
        return {
            "url": f"{self.base_url.rstrip('/')}/chat/completions",
            "headers": {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "json": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 500,
                "temperature": self.temperature
            }
        }

    def _send_request_sync(self, params: Dict, content: str, user_id: str) -> Dict[str, Any]:
        """同步发送意图识别请求"""
        request = self._build_request(params["system_prompt"], content)
        try:
            response = get_sync_client().post(**request)
            response.raise_for_status()
            return self._process_response(response.json(), content)
        except Exception as e:
            return self._handle_request_error(e, content)

    async def _send_request_async(self, params: Dict, content: str, user_id: str) -> Dict[str, Any]:
        """异步发送意图识别请求"""
        request = self._build_request(params["system_prompt"], content)
        try:
            response = await get_async_client().post(**request)
            response.raise_for_status()
            return self._process_response(response.json(), content)
        except Exception as e:
            return self._handle_request_error(e, content)

    def _handle_request_error(self, error: Exception, content: str) -> Dict[str, Any]:
        """统一处理请求错误"""
        error_messages = {
            json.JSONDecodeError: f"JSON解析失败: {error}",
            httpx.TimeoutException: "API超时",
            httpx.HTTPStatusError: f"HTTP错误: {error.response.status_code}" if hasattr(error, 'response') else f"HTTP错误: {error}",
            httpx.RequestError: f"请求失败: {error}"
        }
        
        error_type = type(error).__name__
        message = error_messages.get(type(error), f"异常: {error_type}: {error}")
        logger.warning(f"[意图识别] {message}")
        
        logger.info("[意图识别] 降级为普通对话")
        return self._default_result(content)
    
    def _process_response(
        self, result: Dict, content: str
    ) -> Dict[str, Any]:
        """处理 API 响应"""
        reply = result['choices'][0]['message']['content'].strip()

        # 记录 token 使用
        usage = result.get('usage', {})
        if usage:
            token_monitor.record(
                user_id="", model=self.model,  # user_id 在意图识别时不重要
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0),
                request_type="intent"
            )

        # 提取 JSON 内容
        json_str = self._extract_json(reply)
        parsed = json.loads(json_str)

        intent = parsed.get("intent", "TYPE_CHITCHAT")
        query = parsed.get("query", content)

        intent_result = {
            "intent": intent,
            "query": query
        }

        # 日志输出
        log_msg = f"[意图识别] 知识库查询" if intent == "TYPE_KNOWLEDGE_BASE" else f"[意图识别] 普通对话"
        logger.info(log_msg)

        return intent_result

    def _extract_json(self, text: str) -> str:
        """从响应文本中提取 JSON"""
        # 处理各种格式的 JSON
        if 'habiliment' in text:
            if '```' in text:
                text = text.split('```')[-1].strip()
            elif 'final_response' in text:
                parts = text.split('final_response')
                if len(parts) > 1:
                    text = parts[-1].strip()
        
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        
        start, end = text.find('{'), text.rfind('}') + 1
        return text[start:end] if start != -1 and end > start else text
    
    def _default_result(self, content: str) -> Dict[str, Any]:
        """返回默认的意图识别结果"""
        return {
            "intent": "TYPE_CHITCHAT",
            "query": content
        }
    
    def _build_category_descriptions(self, categories: List[str], kb_list: Dict) -> str:
        """构建分类描述文本"""
        return "\n".join([f"- {cat}: 包含 {len(kb_list.get(cat, []))} 个文档" for cat in categories])
    
    def _format_context(self, previous_context: List[Dict]) -> str:
        """格式化历史上下文"""
        if previous_context:
            try:
                return json.dumps(previous_context[-10:], ensure_ascii=False)
            except:
                pass
        return "无"
    
    def _build_system_prompt(self, category_descriptions: str, context_str: str) -> str:
        """构建系统提示词"""
        return f"""你是一个专业的意图识别引擎，负责分析用户消息并判断消息类型。

## 核心判断原则

**TYPE_KNOWLEDGE_BASE（需要查询知识库）**：
- 用户在寻求具体的、客观的信息、公司规章、技术资料等。
- 问题涉及需要查阅文档才能准确回答的内容。
- 明确的疑问句，期望得到事实性答案。

**TYPE_CHITCHAT（普通对话）**：
- 用户在表达个人情感、分享日常经历
- 用户在发号施令或发出控制指令
- 日常问候、闲聊、情绪倾诉或寻求建议/安慰
- 用户询问实时新闻、天气等需要网络搜索的内容

## 可用知识库分类
{category_descriptions}

## 最近聊天上下文
{context_str}

## 输出格式（严格JSON）
{{
  "intent": "TYPE_KNOWLEDGE_BASE 或 TYPE_CHITCHAT",
  "query": "如果需要查知识库，重写为消除代词的独立查询句；否则保持原样"
}}

## 判断要点
1. 公司文档、规章制度、技术资料 → TYPE_KNOWLEDGE_BASE
2. 日常遭遇、购物消费、就医经历、实时新闻、天气等 → TYPE_CHITCHAT"""
