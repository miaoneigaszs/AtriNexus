"""Context-window management for a single agent run.

ContextEngine decides when model-bound messages are too large and compresses them
before the next provider call. It only manages the short-lived message window;
long-term memory remains in src/memory.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


logger = logging.getLogger("wecom")


# ── 基类 ─────────────────────────────────────────────────────────────────


class ContextEngine(ABC):
    """所有 context 引擎的最小接口。

    子类必须实现 should_compress / compress；其余 lifecycle 方法有合理默认。
    """

    name: str = "base"

    # 触发阈值：当 last_prompt_tokens / context_length 超过该比例时压缩
    threshold_percent: float = 0.75

    # 不可触碰的"头部 N 条"和"尾部 N 条"——保护 system prompt 与最近上下文
    protect_first_n: int = 3
    protect_last_n: int = 6

    def __init__(self, context_length: int = 32_000) -> None:
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.compression_count: int = 0
        self.context_length: int = max(1, int(context_length))
        self.threshold_tokens: int = int(self.context_length * self.threshold_percent)

    # -- Core ---------------------------------------------------------------

    @abstractmethod
    def should_compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> bool:
        """是否需要压缩。current_tokens 缺省时由实现自行估算。"""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """返回压缩后的消息列表（必须是合法的 OpenAI 风格序列）。"""

    # -- Optional preflight -------------------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """LLM 调用前的廉价估算。默认走 should_compress。"""
        return self.should_compress(messages)

    # -- Token bookkeeping --------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """收到 LLM 响应后更新 token 计数。usage 接受 OpenAI 风格字段。"""
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0))
        completion = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0))
        if prompt or completion:
            self.last_prompt_tokens = prompt
            self.last_completion_tokens = completion
            self.last_total_tokens = prompt + completion

    # -- Lifecycle ----------------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        """会话开启钩子。默认无操作。"""

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """会话真正结束（用户退出 / 重置）的钩子。默认无操作。"""

    def on_session_reset(self) -> None:
        """/reset 之类指令调用，重置 per-session 计数。"""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    def update_model(
        self,
        model: str,
        context_length: int,
    ) -> None:
        """切模型时刷新 context window 上限。"""
        del model
        self.context_length = max(1, int(context_length))
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    # -- Status -------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        usage_pct = 0.0
        if self.context_length > 0:
            usage_pct = min(100.0, self.last_prompt_tokens / self.context_length * 100.0)
        return {
            "engine": self.name,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_total_tokens": self.last_total_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": usage_pct,
            "compression_count": self.compression_count,
        }


# ── 默认实现：头尾保留 + 中段省略 ───────────────────────────────────────


class DefaultCompressor(ContextEngine):
    """无 LLM 依赖的轻量压缩：保留头 N 条 + 尾 M 条，中段替换为占位 system 消息。

    优点：完全确定性、零 token 成本、不会破坏对话连贯（最近上下文完整保留）。
    缺点：丢失中段的事实信息。需要 LLM 摘要质量时换实现，接口不变。
    """

    name = "compressor"

    OMITTED_PLACEHOLDER_TEMPLATE = (
        "[省略了 {dropped} 条较早消息以控制上下文长度，"
        "保留了开头 {head} 条与最近 {tail} 条]"
    )

    def __init__(
        self,
        context_length: int = 32_000,
        *,
        chars_per_token: int = 4,
    ) -> None:
        super().__init__(context_length=context_length)
        self.chars_per_token = max(1, chars_per_token)

    # -- Token estimation ---------------------------------------------------

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """字符长度粗估 token；与 platform_core/token_monitor 估算口径一致。"""
        if not messages:
            return 0
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        total_chars += len(item)
                    elif isinstance(item, dict):
                        text = item.get("text", "")
                        if isinstance(text, str):
                            total_chars += len(text)
        if total_chars <= 0:
            return 0
        return max(1, total_chars // self.chars_per_token)

    # -- Decisions ----------------------------------------------------------

    def should_compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> bool:
        if self.threshold_tokens <= 0 or not messages:
            return False
        if len(messages) <= self.protect_first_n + self.protect_last_n:
            return False
        tokens = current_tokens if current_tokens is not None else self.estimate_tokens(messages)
        return tokens >= self.threshold_tokens

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages
        if len(messages) <= self.protect_first_n + self.protect_last_n:
            return messages

        head = list(messages[: self.protect_first_n])
        tail = list(messages[-self.protect_last_n:])
        dropped = len(messages) - len(head) - len(tail)
        if dropped <= 0:
            return messages

        placeholder = {
            "role": "system",
            "content": self.OMITTED_PLACEHOLDER_TEMPLATE.format(
                dropped=dropped,
                head=len(head),
                tail=len(tail),
            ),
        }
        compressed = head + [placeholder] + tail
        self.compression_count += 1
        logger.info(
            "Context compressed: %s -> %s msgs (engine=%s)",
            len(messages),
            len(compressed),
            self.name,
        )
        return compressed
