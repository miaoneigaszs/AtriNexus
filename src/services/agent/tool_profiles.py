from __future__ import annotations

PROFILE_ORDER = (
    "chat",
    "workspace_read",
    "workspace_edit",
    "workspace_exec",
    "full",
)


def normalize_tool_profile(profile: str | None) -> str:
    if not profile:
        return "chat"

    normalized = str(profile).strip().lower()
    if normalized in PROFILE_ORDER:
        return normalized
    return "chat"


def merge_tool_profile(current_profile: str | None, inferred_profile: str | None) -> str:
    current = normalize_tool_profile(current_profile)
    inferred = normalize_tool_profile(inferred_profile)

    if PROFILE_ORDER.index(inferred) > PROFILE_ORDER.index(current):
        return inferred
    return current


def should_enable_workspace_read(profile: str) -> bool:
    return normalize_tool_profile(profile) in {"workspace_read", "workspace_edit", "workspace_exec", "full"}


def should_enable_workspace_edit(profile: str) -> bool:
    return normalize_tool_profile(profile) in {"workspace_edit", "workspace_exec", "full"}


def should_enable_command(profile: str) -> bool:
    return normalize_tool_profile(profile) in {"workspace_exec", "full"}


def should_enable_web(profile: str) -> bool:
    return normalize_tool_profile(profile) in {"chat", "workspace_read", "workspace_edit", "workspace_exec", "full"}


def describe_tool_profile(profile: str | None) -> str:
    normalized = normalize_tool_profile(profile)
    descriptions = {
        "chat": "当前以对话、知识库、记忆和联网查询为主，不默认暴露工作区读写与命令执行能力。",
        "workspace_read": "当前允许读取目录、读取文件、搜索文本，适合查看代码、文档和配置。",
        "workspace_edit": "当前允许读取 workspace 并生成文件修改预览，适合改文档、改配置、改代码，但真正落盘仍需确认。",
        "workspace_exec": "当前允许读取 workspace、生成文件修改预览并执行命令，适合工作区排查和轻量执行任务。",
        "full": "当前暴露全部能力，包括对话、检索、工作区读写预览、命令执行与联网查询。",
    }
    return descriptions.get(normalized, descriptions["chat"])
