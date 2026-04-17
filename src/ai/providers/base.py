"""所有 provider adapter 必须实现的最小接口。

对标 pi-mono `packages/ai/src/types.ts` 里的 stream 函数 + provider 适配。
单个方法 `stream`：接消息 + 工具 + 模型参数，吐 StreamEvent 异步迭代器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from src.ai.types import Message, StreamEvent, ToolSpec


@dataclass(frozen=True)
class ProviderRequest:
    """一次模型调用的全部输入。"""

    model: str
    messages: List[Message]
    tools: List[ToolSpec]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_headers: Optional[dict] = None


class ProviderAdapter(ABC):
    """Provider 抽象。每个具体 provider（OpenAI 兼容、Anthropic native 等）实现一份。"""

    name: str = "base"

    @abstractmethod
    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamEvent]:
        """流式发起一次调用，逐个 yield StreamEvent；最后一个事件应是 StreamDone。

        实现方约定：
        - 单次调用永不 raise（请求/网络/模型错误一律包成 StreamError 并紧跟 StreamDone）
        - 必须在 StreamDone 上挂 usage（如果 provider 没给 usage 字段，也要构造一个空 Usage）
        - 必须支持取消：caller 通过取消外层 task 关闭 generator，实现需在 yield 之间能感知
        """
        # 让类型检查器知道这是 async generator
        if False:
            yield  # pragma: no cover

    def supports_tools(self, model: str) -> bool:  # noqa: ARG002 — provider 子类可重写
        """该 provider 在该 model 下是否支持 tool calling。"""
        return True


__all__ = ["ProviderAdapter", "ProviderRequest"]
