from __future__ import annotations

from typing import Any, Dict, List, Optional


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_MESSAGE_CHARS = 800


def extract_text(result: Any) -> str:
    """从 LangChain agent 结果中提取最终 AI 文本。"""

    if isinstance(result, dict):
        messages = result.get("messages") or []
        for message in reversed(messages):
            if getattr(message, "type", "") != "ai":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                return "\n".join(part for part in parts if part).strip()
    return str(result).strip()


def collect_usage_metadata(result: Any) -> Optional[Dict[str, int]]:
    """聚合 LangChain 返回中的 usage 元数据。"""

    if not isinstance(result, dict):
        return None

    total_prompt = 0
    total_completion = 0
    found_usage = False

    for message in result.get("messages") or []:
        if getattr(message, "type", "") != "ai":
            continue

        usage = normalize_usage_dict(message)
        if not usage:
            continue

        total_prompt += usage["prompt_tokens"]
        total_completion += usage["completion_tokens"]
        found_usage = True

    if not found_usage:
        return None

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
    }


def normalize_usage_dict(message: Any) -> Optional[Dict[str, int]]:
    """兼容不同模型返回口径的 token usage 字段。"""

    usage_sources = [
        getattr(message, "usage_metadata", None),
        getattr(message, "response_metadata", None),
    ]

    for usage_source in usage_sources:
        usage = extract_usage_fields(usage_source)
        if usage:
            return usage
    return None


def extract_usage_fields(source: Any) -> Optional[Dict[str, int]]:
    """从 usage source 中抽出 prompt/completion token 计数。"""

    if not isinstance(source, dict):
        return None

    if "prompt_tokens" in source or "completion_tokens" in source:
        return {
            "prompt_tokens": int(source.get("prompt_tokens", 0)),
            "completion_tokens": int(source.get("completion_tokens", 0)),
        }

    if "input_tokens" in source or "output_tokens" in source:
        return {
            "prompt_tokens": int(source.get("input_tokens", 0)),
            "completion_tokens": int(source.get("output_tokens", 0)),
        }

    token_usage = source.get("token_usage")
    if isinstance(token_usage, dict):
        return {
            "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(token_usage.get("completion_tokens", 0)),
        }

    return None


def estimate_tokens(text: Optional[str]) -> int:
    """用字符长度做粗粒度 token 估算。"""

    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: Optional[List[Dict[str, Any]]]) -> int:
    """估算最近历史消息消耗的 token。"""

    if not messages:
        return 0

    total_chars = 0
    for item in list(messages)[-MAX_HISTORY_MESSAGES:]:
        content = truncate_message_content(str(item.get("content", "")).strip())
        total_chars += len(content)

    if total_chars <= 0:
        return 0
    return max(1, total_chars // 4)


def truncate_message_content(content: str) -> str:
    """限制单条历史消息长度，避免旧内容淹没当前问题。"""

    if len(content) <= MAX_HISTORY_MESSAGE_CHARS:
        return content
    return content[: MAX_HISTORY_MESSAGE_CHARS - 17].rstrip() + "\n[内容已截断]"
