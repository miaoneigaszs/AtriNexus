from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.services.agent.tool_profiles import describe_tool_profile, normalize_tool_profile

logger = logging.getLogger("wecom")


class PromptManager:
    """读取并组装系统 prompt 文件。"""

    SYSTEM_FILE_ORDER = (
        ("身份定位", "identity.md"),
        ("人格基调", "soul.md"),
        ("运行规则", "rules.md"),
        ("工具原则", "tools.md"),
        ("记忆策略", "memory_policy.md"),
    )

    RUNTIME_FILE_ORDER = (
        ("用户黑板", "user.md"),
        ("偏好黑板", "preferences.md"),
        ("会话黑板", "session_memory.md"),
        ("工作备注", "working_notes.md"),
    )

    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        self.system_dir = self.root_dir / "src" / "base" / "prompts" / "system"
        self.runtime_dir = self.root_dir / "src" / "base" / "prompts" / "runtime"
        self.avatar_root = self.root_dir / "data" / "avatars"

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

    def build_avatar_prompt(self, avatar_name: str, current_mode: str) -> str:
        avatar_path = self.avatar_root / avatar_name / "avatar.md"
        avatar_prompt = self._read_markdown(avatar_path)
        if current_mode == "work" and avatar_prompt:
            avatar_prompt += (
                "\n\n【当前模式：工作模式】"
                "请以专业、简洁、高效的方式回应。"
                "保留少量角色感即可，不要主动使用轻佻、挖苦或冒犯性的调侃。"
                "不要使用“机器人保护法”式的玩笑威胁，不要把用户称作“小寂寞鬼”或类似称呼。"
                "默认先解决问题，再保留一点温和的人设表达。"
            )
        return avatar_prompt

    def build_persona_prompt(self, avatar_name: str, current_mode: str) -> str:
        return self.build_avatar_prompt(avatar_name, current_mode)

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
            sections.append(f"【当前可用工具摘要】\n{tool_summary}")
        if current_persona_prompt:
            sections.append(f"【当前角色设定】\n{current_persona_prompt}")

        blackboard = self._build_runtime_blackboard()
        if blackboard:
            sections.append(f"【可写黑板】\n{blackboard}")

        if core_memory:
            sections.append(f"【核心记忆】\n{core_memory}")
        if kb_context:
            sections.append(
                "【参考资料】\n"
                f"{kb_context}\n"
                "必须结合上述参考资料，并严格保持当前角色设定来回答用户的问题。"
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

    def _build_runtime_blackboard(self) -> str:
        sections: List[str] = []
        for title, filename in self.RUNTIME_FILE_ORDER:
            content = self._read_markdown(self.runtime_dir / filename)
            if not content or self._is_placeholder(content):
                continue
            sections.append(f"【{title}】\n{content}")
        return "\n\n".join(sections)

    def _read_markdown(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.debug("Prompt 文件不存在: %s", path)
            return ""
        except Exception as exc:
            logger.warning("读取 Prompt 文件失败 %s: %s", path, exc)
            return ""

    def _is_placeholder(self, content: str) -> bool:
        stripped = content.strip()
        return stripped.startswith("<!--") and stripped.endswith("-->")
