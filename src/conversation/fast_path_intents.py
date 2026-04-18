from __future__ import annotations


TOOL_OVERVIEW_HINTS = (
    "有哪些工具",
    "有什么工具",
    "能用什么工具",
    "可以用什么工具",
    "能做什么",
    "会什么",
    "能力有哪些",
)

PROFILE_OVERVIEW_HINTS = (
    "当前能力档位",
    "当前工具档位",
    "当前模式",
    "当前会话模式",
    "我现在是什么模式",
    "我现在是什么档位",
    "现在是什么模式",
    "现在是什么档位",
)


def is_tool_overview(message: str) -> bool:
    return any(hint in (message or "") for hint in TOOL_OVERVIEW_HINTS)


def is_profile_overview(message: str) -> bool:
    return any(hint in (message or "") for hint in PROFILE_OVERVIEW_HINTS)
