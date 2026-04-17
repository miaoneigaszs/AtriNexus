"""Anthropic prompt caching 标记工具（system_and_3 策略）。

对长多轮对话应用 Anthropic 的 cache_control 断点，可把输入 token 成本降 ~75%。
放 4 个断点（Anthropic 上限）：
  1. system prompt（整会话稳定）
  2-4. 最近 3 条非 system 消息（滚动窗口）

纯函数，无类状态，无 AIAgent 依赖。参考 hermes-agent/agent/prompt_caching.py。

**当前整合状态**：AtriNexus 目前通过 LangChain ChatOpenAI 访问模型。LangChain 把
消息转成 `BaseMessage` 对象送入 provider SDK，cache_control 字段是否被透传取决于
具体 provider（OpenAI 兼容代理通常会透传，原生 Anthropic SDK 则需要 native_anthropic
模式）。本工具作为未来自建 provider 层（Phase 4，对标 pi-ai）的一部分预先就位，
在替代 LangChain 之后会成为 Anthropic 请求发送前的标准预处理步骤。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


# 认为支持 cache_control 的模型名前缀（Anthropic 原生 + 兼容代理）。
# OpenAI / DeepSeek 的服务端缓存不需要用户主动标记，所以这里留空。
_CACHE_CONTROL_MODEL_PREFIXES = (
    "claude-",
    "anthropic/",
)


def model_supports_cache_control(model_name: str) -> bool:
    """根据模型名判断是否需要主动写 cache_control。

    OpenAI 与 DeepSeek 的 prompt caching 在服务端自动生效，不需要主动标记。
    只有 Anthropic（含兼容代理）需要手动插入 cache_control 断点。
    """
    if not model_name:
        return False
    lowered = model_name.lower()
    return any(lowered.startswith(prefix) for prefix in _CACHE_CONTROL_MODEL_PREFIXES)


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """给单条消息打上 cache_control；容忍 str / list / None 三种 content 格式。"""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """对 dict 形式的消息列表应用 system_and_3 缓存策略。

    放置至多 4 个 cache_control 断点：system + 最近 3 条非 system 消息。

    Args:
        api_messages: OpenAI/Anthropic 风格的 dict 列表（role/content 字段）。
        cache_ttl: "5m" 或 "1h"。Anthropic 的两档 TTL。
        native_anthropic: True 表示走 Anthropic 原生 SDK（cache_control 直接挂消息上）；
            False 表示走 OpenAI 兼容代理（挂最后一个 content block 上）。

    Returns:
        深拷贝后的消息列表，已注入 cache_control 字段。
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker: Dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
