"""框架中立的 AI 运行时类型。

这些 dataclass 描述 provider 层和 agent loop 使用的消息、工具调用、用量和流事件。它们可以转换为 OpenAI 兼容字典，同时不导入任何 SDK 专属类型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


# ── 内容块 ───────────────────────────────────────────────────────────


@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"

    def to_openai_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass
class ImageContent:
    """OpenAI 风格的 image_url 引用；data URL 也走这里。"""

    image_url: str
    detail: Literal["low", "high", "auto"] = "auto"
    type: Literal["image_url"] = "image_url"

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.image_url, "detail": self.detail},
        }


ContentBlock = Union[TextContent, ImageContent]


def _content_to_openai(content: Union[str, List[ContentBlock]]) -> Any:
    if isinstance(content, str):
        return content
    return [block.to_openai_dict() for block in content]


# ── 工具调用 ─────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """模型决定调用工具时的请求。args 已经是解析后的 dict（不是 JSON 字符串）。"""

    id: str
    name: str
    args: Dict[str, Any]

    def to_openai_dict(self) -> Dict[str, Any]:
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.args, ensure_ascii=False),
            },
        }


# ── 消息 ─────────────────────────────────────────────────────────────


@dataclass
class SystemMessage:
    content: str
    role: Literal["system"] = "system"

    def to_openai_dict(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.content}


@dataclass
class UserMessage:
    content: Union[str, List[ContentBlock]]
    role: Literal["user"] = "user"

    def to_openai_dict(self) -> Dict[str, Any]:
        return {"role": "user", "content": _content_to_openai(self.content)}


@dataclass
class AssistantMessage:
    content: Union[str, List[ContentBlock]] = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    # 模型提供方自报的元数据（finish_reason、cache 信息等）原样保留
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"role": "assistant"}
        if self.content:
            body["content"] = _content_to_openai(self.content)
        if self.tool_calls:
            body["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]
        return body


@dataclass
class ToolResultMessage:
    """工具执行结果。"""

    tool_call_id: str
    content: str
    is_error: bool = False
    name: Optional[str] = None
    role: Literal["tool"] = "tool"

    def to_openai_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
        if self.name:
            body["name"] = self.name
        return body


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]


# ── 用量 ─────────────────────────────────────────────────────────────


@dataclass
class Usage:
    """单次模型调用的 token 统计；兼容 OpenAI 与 Anthropic 两种字段命名。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


# ── 工具规格 ─────────────────────────────────────────────────────────


@dataclass
class ToolSpec:
    """送给模型的工具描述。parameters 是 JSONSchema dict。"""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ── 流事件 ───────────────────────────────────────────────────────────


@dataclass
class TextDelta:
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass
class ToolCallDelta:
    """流式中的 tool_call 累积片段；index 标识同一 assistant message 内第几个调用。"""

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    args_delta: str = ""
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass
class StreamDone:
    """流结束信号。stop_reason 来自 provider；usage 在最后一个 chunk 才到位。"""

    stop_reason: str = ""
    usage: Optional[Usage] = None
    type: Literal["done"] = "done"


@dataclass
class StreamError:
    message: str
    type: Literal["error"] = "error"


StreamEvent = Union[TextDelta, ToolCallDelta, StreamDone, StreamError]


# ── 辅助函数 ─────────────────────────────────────────────────────────


def messages_to_openai(messages: List[Message]) -> List[Dict[str, Any]]:
    return [m.to_openai_dict() for m in messages]


def tools_to_openai(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [t.to_openai_dict() for t in tools]


__all__ = [
    "TextContent",
    "ImageContent",
    "ContentBlock",
    "ToolCall",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    "Usage",
    "ToolSpec",
    "TextDelta",
    "ToolCallDelta",
    "StreamDone",
    "StreamError",
    "StreamEvent",
    "messages_to_openai",
    "tools_to_openai",
]
