from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentRunContext:
    """单轮 agent 调用的动态上下文。"""

    persona_prompt: str
    core_memory: Optional[str]
    tool_profile: Optional[str]
    tool_profiles: list[str]
    tool_summary: str
