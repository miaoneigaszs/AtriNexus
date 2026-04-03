"""
LLM AI 服务模块
提供与LLM API的完整交互实现，包含以下核心功能：
- API请求管理
- 上下文对话管理
- 响应安全处理
- 智能错误恢复
- Token 使用监测
"""


import logging
import re # 正则表达式处理
import os
import json
import requests
from typing import Dict, List, Optional, Any
from openai import OpenAI

import emoji
from src.services.ai.model_manager import ModelManager
from src.services.token_monitor import token_monitor
from src.services.tools import ToolRegistry
from src.services.tools.time_tool import TimeTool
from src.services.tools.search_tool import SearchTool

logger = logging.getLogger('main')
USER_VISIBLE_LLM_ERROR = "抱歉，我暂时无法处理你的消息，请稍后再试。"

# 不支持工具调用的模型列表
MODELS_WITHOUT_TOOL_SUPPORT = [
    'deepseek-reasoner',
    'deepseek-r1',
]


class LLMService:
    """
        强化版AI服务类，支持多模型自动切换、工具调用、响应安全处理和智能错误恢复
            - 自动模型切换：当启用 auto_model_switch 时，遇到错误会自动尝试预设的备用模型列表，增加请求成功率
            - 工具调用支持：内置时间查询工具和可选的网络搜索工具，允许模型在生成回复时调用外部工具获取实时信息
            - 响应安全处理：移除控制字符、标准化换行符、处理emoji表情，确保回复内容安全且格式正确
            - 智能错误恢复：针对不同类型的错误（如超时、API错误）进行分类处理，并根据配置决定是否重试或切换模型
            - Token使用监测：集成token_monitor模块，记录每次请求的token使用情况，支持后续分析和优化
    """
    
    def __init__(self, api_key: str, base_url: str, model: str,
                 max_token: int, temperature: float, max_groups: int,
                 auto_model_switch: bool = False, fallback_models: list = None):
        """
        强化版AI服务初始化

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
        self.chat_contexts: Dict[str, List[Dict]] = {}
        self.safe_pattern = re.compile(r'[\x00-\x1F\u202E\u200B]')
        
        # 使用 ModelManager 管理模型（传入用户配置的备用模型列表）
        self.model_manager = ModelManager(self.client, model, fallback_models=fallback_models or [])
        self.ollama_models = self.model_manager.get_ollama_models()
        self.available_models = self.model_manager.get_available_models()

        # 初始化工具注册表
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(TimeTool())
        # 搜索工具按需注册（启动时检查配置，运行时通过 _build_messages 动态判断）
        self._search_api_key: Optional[str] = None  # 缓存，_build_messages 中更新

    def _manage_context(self, user_id: str, message: str, role: str = "user"):
        """
        上下文管理器（支持动态记忆窗口）

        :param user_id: 用户唯一标识
        :param message: 消息内容
        :param role: 角色类型(user/assistant)
        """
        if user_id not in self.chat_contexts:
            self.chat_contexts[user_id] = []

        # 添加新消息
        self.chat_contexts[user_id].append({"role": role, "content": message})

        # 维护上下文窗口
        while len(self.chat_contexts[user_id]) > self.config["max_groups"] * 2:
            # 优先保留最近的对话组
            self.chat_contexts[user_id] = self.chat_contexts[user_id][-self.config["max_groups"]*2:]
    
    # 工具定义与执行逻辑已迁移至 src/services/tools/
    # TimeTool  → src/services/tools/time_tool.py
    # SearchTool → src/services/tools/search_tool.py
    # 工具注册由 self.tool_registry (ToolRegistry) 统一管理


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
                            # 格式1c: 工具调用 - content为空但有tool_calls
                            tool_calls = message.get("tool_calls")
                            if tool_calls and isinstance(tool_calls, list):
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

    def _build_messages(self, user_id: str, message: str, system_prompt: str,
                        core_memory: Optional[str] = None, kb_context: Optional[str] = None) -> Dict[str, Any]:
        """构建消息列表和请求参数（结构化拼接上下文，强化角色设定）"""
        
        # 采用块状结构拼接提示词，解决之前因直接连接带来的“语境断裂”和角色污染问题
        prompt_parts = []
        
        # 1. 强制写入核心人设（由用户配置提供）
        if system_prompt:
            prompt_parts.append(f"【角色设定】\n{system_prompt}")
            
        # 2. 注入核心记忆
        if core_memory:
            prompt_parts.append(f"【核心记忆】\n{core_memory}")
            
        # 3. 注入 RAG 知识库检索内容，并加入强制指令，防止脱离角色
        if kb_context:
            prompt_parts.append(f"【参考资料】\n{kb_context}\n必须结合上述参考资料，并严格保持你的【角色设定】来回答用户的问题。如果参考资料无相关性，请忽略资料，自然回复即可。")
            
        final_prompt = "\n\n".join(prompt_parts)

        chat_history = self.chat_contexts.get(user_id, [])[-self.config["max_groups"] * 2:]
        
        messages = [
            {"role": "system", "content": final_prompt},
            *chat_history,
            {"role": "user", "content": message}
        ]
        
        # 计算 token 详细分解（简单估算：中文约1.5字符/token，英文约4字符/token）
        def estimate_tokens(text: str) -> int:
            if not text:
                return 0
            # 简单估算：中文字符数/1.5 + 英文单词数
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4) + 1
        
        # 计算各部分的 token
        system_prompt_tokens = estimate_tokens(final_prompt)
        core_memory_tokens = estimate_tokens(core_memory) if core_memory else 0
        kb_context_tokens = estimate_tokens(kb_context) if kb_context else 0
        chat_history_tokens = sum(estimate_tokens(m.get("content", "")) for m in chat_history)
        user_message_tokens = estimate_tokens(message)
        
        # Ollama 需要特殊格式的消息
        ollama_message = {
            "role": "user",
            "content": f"{final_prompt}\n\n用户问题：{message}"
        }
        is_ollama = 'localhost:11434' in str(self.client.base_url)
        
        # 从 ToolRegistry 获取工具 schemas
        # 搜索工具：按需动态注册/更新
        from data.config import config
        search_cfg = config.network_search
        if search_cfg.search_enabled and search_cfg.api_key:
            if not self.tool_registry.has("web_search") or self._search_api_key != search_cfg.api_key:
                self.tool_registry.register(SearchTool(api_key=search_cfg.api_key))
                self._search_api_key = search_cfg.api_key
        elif self.tool_registry.has("web_search"):
            self.tool_registry.unregister("web_search")
        
        tools = self.tool_registry.get_schemas()
        
        # Token 分解信息
        token_breakdown = {
            "system_prompt_tokens": system_prompt_tokens,
            "core_memory_tokens": core_memory_tokens,
            "kb_context_tokens": kb_context_tokens,
            "chat_history_tokens": chat_history_tokens,
            "user_message_tokens": user_message_tokens,
        }
        
        return {
            "messages": messages,
            "ollama_message": ollama_message,
            "is_ollama": is_ollama,
            "tools": tools,
            "token_breakdown": token_breakdown,
        }

    def _send_single_request(self, current_model: str, built: Dict, user_id: str = "unknown") -> str:
        """执行单次 API 请求，支持工具调用，成功返回 raw_content，失败抛出异常"""
        messages = built["messages"]
        ollama_message = built["ollama_message"]
        is_ollama = built["is_ollama"]
        tools = built.get("tools", [])

        if is_ollama:
            from src.utils.version import get_current_version, get_version_identifier
            request_config = {
                "model": current_model.split('/')[-1],
                "messages": [ollama_message],
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
                estimated_tokens = len(content) // 4  # 粗略估算
                token_breakdown = built.get("token_breakdown", {})
                token_monitor.record(
                    user_id=user_id,
                    model=current_model,
                    prompt_tokens=estimated_tokens,
                    completion_tokens=estimated_tokens,
                    request_type="chat",
                    **token_breakdown
                )
                return content
            raise ValueError(f"错误的API响应结构: {json.dumps(response_data, default=str)}")

        # 发送请求（带工具支持）
        response = self.client.chat.completions.create(
            model=current_model,
            messages=messages,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_token"],
            tools=tools if tools else None,
            tool_choice="auto" if tools else None
        )
        
        if not self._validate_response(response.model_dump()):
            raise ValueError(f"错误的API响应结构: {json.dumps(response.model_dump(), default=str)}")
        
        message = response.choices[0].message
        
        # 记录本次请求的 token 使用
        usage = getattr(response, 'usage', None)
        if usage:
            token_breakdown = built.get("token_breakdown", {})
            token_monitor.record(
                user_id=user_id,
                model=current_model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                request_type="chat",
                **token_breakdown
            )
        
        # 检查是否有工具调用
        if message.tool_calls:
            # 通过 ToolRegistry 统一分发工具调用，新增工具无需修改此处
            tool_results = []
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments or "{}"
                tool_result = self.tool_registry.execute(fn_name, fn_args)
                logger.info(f"[Tool Call] 执行 {fn_name}: 结果长度={len(tool_result)}")
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result
                })
            
            if tool_results:
                # 将工具结果添加到消息列表，再次请求 LLM
                messages.append(message.model_dump())
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })
                
                # 再次调用 API 获取最终回复
                final_response = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_token"]
                )
                
                # 记录第二次请求的 token 使用
                final_usage = getattr(final_response, 'usage', None)
                if final_usage:
                    token_breakdown = built.get("token_breakdown", {})
                    token_monitor.record(
                        user_id=user_id,
                        model=current_model,
                        prompt_tokens=final_usage.prompt_tokens,
                        completion_tokens=final_usage.completion_tokens,
                        request_type="chat_tool",
                        **token_breakdown
                    )
                
                raw_content = final_response.choices[0].message.content
            else:
                raw_content = message.content
        else:
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

    def get_response(self, message: str, user_id: str, system_prompt: str, previous_context: List[Dict] = None, core_memory: str = None, kb_context: str = None) -> str:
        """
        完整请求处理流程（主入口，协调各子步骤）
        Args:
            message: 用户消息
            user_id: 用户ID
            system_prompt: 系统提示词（人设）
            previous_context: 历史上下文（可选，须为 List[Dict] 格式）
            core_memory: 核心记忆（可选）
            kb_context: 知识库检索上下文（可选，RAG 时注入）
        """
        if not message.strip():
            logger.warning("收到空消息，跳过 LLM 请求")
            return USER_VISIBLE_LLM_ERROR

        if previous_context and user_id not in self.chat_contexts:
            logger.info(f"程序启动初始化：为用户 {user_id} 加载历史上下文，共 {len(previous_context)} 条消息")
            self.chat_contexts[user_id] = previous_context.copy()
        self._manage_context(user_id, message)

        built = self._build_messages(user_id, message, system_prompt, core_memory, kb_context)
        max_retries = 3
        last_error = None
        current_model = self.config["model"]
        models_tried = []
        logger.info(f"准备发送API请求 - 用户: {user_id}, 模型: {self.config['model']}")

        for attempt in range(max_retries):
            try:
                models_tried.append(current_model)
                logger.info(f"开始API请求 - 尝试 {attempt+1}/{max_retries}, 模型: {current_model}")
                raw_content = self._send_single_request(current_model, built, user_id)
                filtered_content = self._process_response_content(raw_content)
                if filtered_content.strip().lower().startswith("error"):
                    raise ValueError(f"错误响应: {filtered_content}")
                self._manage_context(user_id, filtered_content, "assistant")
                if current_model != self.original_model:
                    logger.info(f"使用备用模型 {current_model} 成功获取响应")
                return filtered_content or ""
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                # 判断是否为超时错误
                if "timeout" in error_str.lower() or "timed out" in error_str.lower():
                    logger.warning(f"模型 {current_model} API请求超时 (尝试 {attempt+1}/{max_retries})")
                else:
                    logger.warning(f"模型 {current_model} API请求失败 (尝试 {attempt+1}/{max_retries}): {error_str}")
                if self.config["auto_model_switch"] and attempt < max_retries - 1:
                    next_model = self.model_manager.get_next_model(current_model)
                    # 检查工具是否启用
                    tools_enabled = built.get("tools", [])
                    # 跳过不支持工具调用的模型（当工具启用时）
                    while next_model and next_model not in models_tried:
                        if tools_enabled and any(m in next_model.lower() for m in MODELS_WITHOUT_TOOL_SUPPORT):
                            logger.info(f"跳过不支持工具调用的模型: {next_model}")
                            next_model = self.model_manager.get_next_model(next_model)
                        else:
                            break
                    if next_model and next_model not in models_tried:
                        logger.info(f"自动切换到模型: {next_model}")
                        current_model = next_model
                        continue
                if attempt < max_retries - 1:
                    continue

        if self.config.get("auto_model_switch", False):
            logger.error(f"所有模型 {models_tried} 均失败: {last_error}")
        else:
            logger.error(f"所有重试尝试均失败: {last_error}")
        return USER_VISIBLE_LLM_ERROR

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

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get('temperature', self.config["temperature"]),
                max_tokens=self.config["max_token"]
            )

            if not self._validate_response(response.model_dump()):
                error_msg = f"错误的API响应结构: {json.dumps(response.model_dump(), default=str)}"
                logger.error(error_msg)
                return USER_VISIBLE_LLM_ERROR

            raw_content = response.choices[0].message.content
            # 清理和过滤响应内容
            clean_content = self._sanitize_response(raw_content)
            filtered_content = self._filter_thinking_content(clean_content)

            return filtered_content or ""

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            return USER_VISIBLE_LLM_ERROR
