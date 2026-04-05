"""
LLM 文本生成服务模块。

说明：
- 当前默认聊天主路径已经切到 LangChainAgentService
- 本文件现在只承担后台文本生成能力，例如记忆摘要、日记和定时任务
- 不再承担主聊天入口、主工具调用或 agent 编排职责
"""


import logging
import re # 正则表达式处理
import json
import requests
from typing import Dict, List, Optional, Any
from openai import OpenAI

import emoji
from src.services.ai.model_manager import ModelManager
from src.services.token_monitor import token_monitor

logger = logging.getLogger('main')
USER_VISIBLE_LLM_ERROR = "抱歉，我暂时无法处理你的消息，请稍后再试。"


class LLMService:
    """
        后台文本生成服务，负责：
            - 自动模型切换
            - 响应安全处理
            - DeepSeek / Ollama 兼容
            - Token 使用监测
    """
    
    def __init__(self, api_key: str, base_url: str, model: str,
                 max_token: int, temperature: float, max_groups: int,
                 auto_model_switch: bool = False, fallback_models: list = None):
        """
        初始化后台文本生成服务

        :param api_key: API认证密钥
        :param base_url: API基础URL
        :param model: 使用的模型名称
        :param max_token: 最大token限制
        :param temperature: 创造性参数(0~2)
        :param max_groups: 最大对话轮次记忆
        :param auto_model_switch: 是否启用自动模型切换
        :param fallback_models: 用户配置的备用模型列表
        """
        from src.utils.version import get_current_version, get_version_identifier
        version = get_current_version()
        version_identifier = get_version_identifier()

        # 配置超时：连接5秒，读取30秒，总超时50秒
        from httpx import Timeout # httpx的Timeout类支持更细粒度的超时控制，分别设置连接、读取、写入和连接池的超时时间
        timeout = Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0) # 连接超时5秒，读取超时30秒，写入超时10秒，连接池超时5秒, 共50s
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "Content-Type": "application/json",
                "User-Agent": version_identifier,
                "X-AtriNexus-Version": version
            }, # 添加自定义版本头，方便后端识别请求来源和版本
            timeout=timeout,
            max_retries=2  # 减少重试次数，快速失败
        )
        self.config = {
            "model": model,
            "max_token": max_token,
            "temperature": temperature,
            "max_groups": max_groups,
            "auto_model_switch": auto_model_switch
        }
        self.original_model = model
        self.safe_pattern = re.compile(r'[\x00-\x1F\u202E\u200B]')
        
        # 使用 ModelManager 管理模型（传入用户配置的备用模型列表）
        self.model_manager = ModelManager(self.client, model, fallback_models=fallback_models or [])

    def _sanitize_response(self, raw_text: str) -> str:
        """
        响应安全处理器
        1. 移除控制字符
        2. 标准化换行符
        3. 处理emoji表情符号
        """
        try:
            # 移除控制字符
            cleaned = re.sub(self.safe_pattern, '', raw_text)

            # 标准化换行符
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')

            # 处理emoji表情符号
            cleaned = self._process_emojis(cleaned)

            return cleaned
        except Exception as e:
            logger.error(f"Response sanitization failed: {str(e)}")
            return "响应处理异常，请重新尝试"

    def _process_emojis(self, text: str) -> str:
        """处理文本中的emoji表情符号，确保跨平台兼容性"""
        try:
            # 先将Unicode表情符号转换为别名再转回，确保标准化
            return emoji.emojize(emoji.demojize(text))
        except Exception:
            return text  # 如果处理失败，返回原始文本

    def _filter_thinking_content(self, content: str) -> str:
        """
        过滤思考内容，支持不同模型的返回格式
        1. R1格式: 思考过程...\n\n\n最终回复
        2. Gemini格式: <think>思考过程</think>\n\n最终回复
        3. DeepSeek推理格式: （思考过程...）最终回复
        """
        try:
            # 使用分割替代正则表达式处理 Gemini 格式
            if '<think>' in content and '</think>' in content:
                parts = content.split('</think>')
                # 只保留最后一个</think>后的内容
                filtered_content = parts[-1].strip()
            else:
                filtered_content = content

            # 过滤 R1 格式 (思考过程...\n\n\n最终回复)
            # 查找三个连续换行符
            triple_newline_match = re.search(r'\n\n\n', filtered_content)
            if triple_newline_match:
                # 只保留三个连续换行符后面的内容（最终回复）
                filtered_content = filtered_content[triple_newline_match.end():]

            # 过滤 DeepSeek 推理格式：移除中文圆括号包裹的思考内容 （...）
            # 匹配连续的 （思考...）块，只保留最后的实际回复
            filtered_content = re.sub(r'（[^）]*）\s*', '', filtered_content)

            return filtered_content.strip()
        except Exception as e:
            logger.error(f"过滤思考内容失败: {str(e)}")
            return content  # 如果处理失败，返回原始内容

    def _validate_response(self, response: dict) -> bool:
        """
        放宽检验
        API响应校验
        只要能获取到有效的回复内容或工具调用就返回True
        """
        try:
            # 调试：打印完整响应结构
            logger.debug(f"API响应结构: {json.dumps(response, default=str, indent=2)}")

            # 尝试获取回复内容
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        # 尝试不同的响应格式
                        # 格式1: choices[0].message.content
                        if isinstance(first_choice.get("message"), dict):
                            message = first_choice["message"]
                            content = message.get("content")
                            if content and isinstance(content, str):
                                return True
                            # 格式1b: DeepSeek推理模型 - content为空，回复在reasoning_content中
                            reasoning = message.get("reasoning_content")
                            if reasoning and isinstance(reasoning, str):
                                return True
                        # 格式2: choices[0].content
                        content = first_choice.get("content")
                        if content and isinstance(content, str):
                            return True

                        # 格式3: choices[0].text
                        text = first_choice.get("text")
                        if text and isinstance(text, str):
                            return True

            logger.warning("无法从响应中获取有效内容，完整响应: %s", json.dumps(response, default=str))
            return False

        except Exception as e:
            logger.error(f"验证响应时发生错误: {str(e)}")
            return False

    def _send_single_request(
        self,
        current_model: str,
        messages: List[Dict[str, str]],
        user_id: str = "unknown",
    ) -> str:
        """执行单次 API 请求，成功返回 raw_content，失败抛出异常。"""
        is_ollama = 'localhost:11434' in str(self.client.base_url)

        if is_ollama:
            from src.utils.version import get_current_version, get_version_identifier
            request_config = {
                "model": current_model.split('/')[-1],
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.config["temperature"], "max_tokens": self.config["max_token"]}
            }
            response = requests.post(
                str(self.client.base_url),
                json=request_config,
                headers={"Content-Type": "application/json", "User-Agent": get_version_identifier(), "X-AtriNexus-Version": get_current_version()},
                timeout=(5, 60)  # 连接5秒，读取60秒
            )
            response.raise_for_status()
            response_data = response.json()
            if response_data and "message" in response_data:
                # Ollama 通常不返回 token 统计，估算
                content = response_data["message"]["content"]
                estimated_tokens = max(1, len(content) // 4)
                token_monitor.record(
                    user_id=user_id,
                    model=current_model,
                    prompt_tokens=max(1, sum(len(m.get("content", "")) for m in messages) // 4),
                    completion_tokens=estimated_tokens,
                    request_type="chat",
                )
                return content
            raise ValueError(f"错误的API响应结构: {json.dumps(response_data, default=str)}")

        # 发送请求（带工具支持）
        response = self.client.chat.completions.create(
            model=current_model,
            messages=messages,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_token"],
        )
        
        if not self._validate_response(response.model_dump()):
            raise ValueError(f"错误的API响应结构: {json.dumps(response.model_dump(), default=str)}")
        
        message = response.choices[0].message
        
        # 记录本次请求的 token 使用
        usage = getattr(response, 'usage', None)
        if usage:
            token_monitor.record(
                user_id=user_id,
                model=current_model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                request_type="chat",
            )
        
        raw_content = message.content
        
        if not raw_content:
            reasoning = getattr(message, 'reasoning_content', None)
            if not reasoning and hasattr(message, 'model_dump'):
                reasoning = (message.model_dump() or {}).get('reasoning_content', '')
            if reasoning:
                logger.info("检测到推理模型格式：content为空，从reasoning_content提取回复")
                raw_content = reasoning
        
        return raw_content

    def _process_response_content(self, raw_content: str) -> str:
        """清理、过滤响应内容"""
        clean_content = self._sanitize_response(raw_content)
        return self._filter_thinking_content(clean_content)

    def chat(self, messages: list, **kwargs) -> str:
        """
        发送聊天请求并获取回复

        Args:
            messages: 消息列表，每个消息是包含 role 和 content 的字典
            **kwargs: 额外的参数配置，包括 model、temperature 等

        Returns:
            str: AI的回复内容
        """
        try:
            # 使用传入的model参数，如果没有则使用默认模型
            model = kwargs.get('model', self.config["model"])
            logger.info(f"使用模型: {model} 发送聊天请求")
            raw_content = self._send_single_request(model, messages)
            clean_content = self._sanitize_response(raw_content)
            return self._filter_thinking_content(clean_content) or ""

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            return USER_VISIBLE_LLM_ERROR
