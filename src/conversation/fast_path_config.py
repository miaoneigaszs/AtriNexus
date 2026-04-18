"""FastPath 意图路由的配置与结果类型。

单独成文件是为了测试能不经 SessionService/数据库链导入这些纯结构。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


FAST_PATH_ENV_VAR = "ATRINEXUS_FAST_PATH_INTENT"
FAST_PATH_MODE_FULL = "full"
FAST_PATH_MODE_DISABLED = "disabled"
_FAST_PATH_MODES = {FAST_PATH_MODE_FULL, FAST_PATH_MODE_DISABLED}

INTENT_DISABLED = "disabled"
INTENT_NONE = "none"
INTENT_TOOL_OVERVIEW = "tool_overview"
INTENT_PROFILE_OVERVIEW = "profile_overview"
INTENT_PENDING_RESOLUTION = "pending_resolution"

_AGENT_PREFIX = "/agent"


def read_fast_path_mode() -> str:
    raw = os.getenv(FAST_PATH_ENV_VAR, "").strip().lower() or FAST_PATH_MODE_FULL
    if raw not in _FAST_PATH_MODES:
        return FAST_PATH_MODE_FULL
    return raw


@dataclass
class FastPathOutcome:
    """FastPath 路由的命中结果 + 标签。

    `reply is None` 表示未命中意图路由，调用方应继续走后续管线。
    `intent` 用于 trajectory 观测，无论命中与否都应有值。
    """

    reply: Optional[str]
    intent: str

    @classmethod
    def miss(cls, intent: str = INTENT_NONE) -> "FastPathOutcome":
        return cls(reply=None, intent=intent)

    @classmethod
    def hit(cls, reply: str, intent: str) -> "FastPathOutcome":
        return cls(reply=reply, intent=intent)


def strip_agent_prefix(content: Optional[str]) -> Tuple[Optional[str], bool]:
    """识别并剥离 `/agent` 前缀。

    返回 (payload, bypass)。bypass=True 时调用方应跳过 FastPath 意图路由。
    前缀必须紧跟空白或结尾，避免误伤以 `/agent…` 为内容的消息（如 `/agentic`）。
    大小写敏感。
    """
    stripped = (content or "").lstrip()
    if not stripped.startswith(_AGENT_PREFIX):
        return content, False
    rest = stripped[len(_AGENT_PREFIX):]
    if rest and not rest[0].isspace():
        return content, False
    return rest.lstrip(), True
