from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.services.agent.tool_profiles import describe_tool_profile, normalize_tool_profile

logger = logging.getLogger("wecom")


class PromptManager:
    """读取并组装系统 prompt 文件。"""

    MAX_RUNTIME_TOOL_SUMMARY_CHARS = 1200
    MAX_RUNTIME_CORE_MEMORY_CHARS = 1400
    MAX_RUNTIME_KB_CONTEXT_CHARS = 1800

    SYSTEM_FILE_ORDER = (
        ("身份定位", "identity.md"),
        ("人格基调", "soul.md"),
        ("运行规则", "rules.md"),
        ("工具原则", "tools.md"),
        ("记忆策略", "memory_policy.md"),
    )

    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        self.system_dir = self.root_dir / "src" / "base" / "prompts" / "system"

    def build_agent_static_prompt(self) -> str:
        """组装稳定前缀。

        这里故意只放长期稳定内容，避免把动态信息混进缓存友好的前缀里。
        """
        sections: List[str] = []
        for title, filename in self.SYSTEM_FILE_ORDER:
            content = self._read_markdown(self.system_dir / filename)
            if content:
                sections.append(f"【{title}】\n{content}")
        return "\n\n".join(sections)

    def build_agent_system_prompt(
        self,
        *,
        tool_profiles: List[str] | None = None,
        tool_summary: str | None = None,
    ) -> str:
        """兼容当前 agent 调用名。

        tool_profiles 和 tool_summary 继续保留在签名里，避免接线阶段改动过大。
        静态前缀不直接依赖它们。
        """
        return self.build_agent_static_prompt()

    def build_mode_prompt(self, current_mode: str) -> str:
        """构建当前会话模式提示。

        这里不再引入独立 avatar 人设文件，避免角色扮演压过工作型助手的主职责。
        """
        if current_mode == "companion":
            return (
                "当前处于陪伴模式。\n"
                "表达可以更温和、更有陪伴感，但仍然要实事求是，不要编造，不要夸张表演。"
            )

        return (
            "当前处于工作模式。\n"
            "表达应当直率、简洁、接地气，有啥说啥，先解决问题。"
            "可以保留少量温度，但不要主动卖萌、调侃、挖苦或角色表演。"
        )

    def build_persona_prompt(self, avatar_name: str, current_mode: str) -> str:
        return self.build_mode_prompt(current_mode)

    def build_runtime_prompt(
        self,
        *,
        avatar_prompt: str = "",
        persona_prompt: str = "",
        tool_profile: str | None = None,
        tool_profiles: List[str] | None = None,
        tool_summary: str = "",
        core_memory: str | None,
        kb_context: str | None,
    ) -> str:
        sections: List[str] = []
        current_persona_prompt = avatar_prompt or persona_prompt

        if tool_profile:
            normalized_profile = normalize_tool_profile(tool_profile)
            sections.append(f"【当前能力档位】\n{normalized_profile}")
            sections.append(f"【当前能力边界】\n{describe_tool_profile(normalized_profile)}")
        if tool_profiles:
            sections.append(f"【当前工具组】\n{', '.join(tool_profiles)}")
        if tool_summary:
            sections.append(
                f"【当前可用工具摘要】\n"
                f"{self._truncate_text(tool_summary, self.MAX_RUNTIME_TOOL_SUMMARY_CHARS)}"
            )
        if current_persona_prompt:
            sections.append(f"【当前会话风格】\n{current_persona_prompt}")

        if core_memory:
            sections.append(
                f"【核心记忆】\n"
                f"{self._truncate_text(core_memory, self.MAX_RUNTIME_CORE_MEMORY_CHARS)}"
            )
        if kb_context:
            sections.append(
                "【参考资料】\n"
                f"{self._truncate_text(kb_context, self.MAX_RUNTIME_KB_CONTEXT_CHARS)}\n"
                "必须结合上述参考资料，并严格保持当前会话风格来回答用户的问题。"
                "如果参考资料无相关性，请忽略资料，自然回复即可。"
            )

        return "\n\n".join(sections)

    def build_fast_path_rewrite_prompt(self) -> str:
        """构建快路径文档改写使用的固定系统壳。"""
        static_prompt = self.build_agent_static_prompt()
        rewrite_rules = (
            "【快路径任务】\n"
            "你现在只负责改写一个已经定位好的文档块。\n"
            "必须保留原有语言。\n"
            "只返回改写后的正文或标题，不要解释，不要加引号，不要输出额外说明。"
        )
        if static_prompt:
            return f"{static_prompt}\n\n{rewrite_rules}"
        return rewrite_rules

    def _read_markdown(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.debug("Prompt 文件不存在: %s", path)
            return ""
        except Exception as exc:
            logger.warning("读取 Prompt 文件失败 %s: %s", path, exc)
            return ""

    def _truncate_text(self, text: str, max_chars: int) -> str:
        normalized = (text or "").strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 17].rstrip() + "\n[内容已截断]"
