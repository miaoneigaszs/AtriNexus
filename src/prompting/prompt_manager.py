from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.agent_runtime.tool_profiles import describe_tool_profile, normalize_tool_profile

logger = logging.getLogger("wecom")


class PromptManager:
    """读取并组装系统 prompt 文件。"""

    MAX_RUNTIME_TOOL_SUMMARY_CHARS = 1200
    MAX_RUNTIME_CORE_MEMORY_CHARS = 1400

    SYSTEM_FILE_ORDER = (
        ("身份定位", "identity.md"),
        ("人格基调", "soul.md"),
        ("运行规则", "rules.md"),
        ("工具原则", "tools.md"),
        ("记忆策略", "memory_policy.md"),
    )

    MODE_FILE_MAP = {
        "work": "work.md",
        "companion": "companion.md",
    }
    DEFAULT_MODE = "work"

    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        self.system_dir = self.root_dir / "src" / "prompting" / "system"
        self.mode_dir = self.system_dir / "modes"

    def build_agent_static_prompt(self) -> str:
        """组装稳定前缀，只放长期稳定内容以保持缓存前缀一致。"""
        sections: List[str] = []
        for title, filename in self.SYSTEM_FILE_ORDER:
            content = self._read_markdown(self.system_dir / filename)
            if content:
                sections.append(f"【{title}】\n{content}")
        return "\n\n".join(sections)

    def build_mode_prompt(self, current_mode: str) -> str:
        """构建当前会话模式提示。

        模式文案直接从 markdown 文件读取，保持"所有静态提示词都在 prompts 目录"的单一事实来源。
        """
        mode_key = current_mode if current_mode in self.MODE_FILE_MAP else self.DEFAULT_MODE
        filename = self.MODE_FILE_MAP[mode_key]
        return self._read_markdown(self.mode_dir / filename)

    def build_persona_prompt(self, avatar_name: str, current_mode: str) -> str:
        # 保留签名兼容外部调用方；avatar_name 当前未使用。
        del avatar_name
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
        current_mode: str | None = None,
    ) -> str:
        sections: List[str] = []
        current_persona_prompt = avatar_prompt or persona_prompt

        capability_lines: List[str] = []
        if current_mode:
            capability_lines.append(f"当前模式：{current_mode}")
        if tool_profile:
            normalized_profile = normalize_tool_profile(tool_profile)
            capability_lines.append(f"当前能力档位：{normalized_profile}")
            capability_lines.append(f"当前能力边界：{describe_tool_profile(normalized_profile)}")
        if tool_profiles:
            capability_lines.append(f"当前工具组：{', '.join(tool_profiles)}")
        if tool_summary:
            capability_lines.append(
                "这些工具当前分别能做：\n"
                f"{self._truncate_text(tool_summary, self.MAX_RUNTIME_TOOL_SUMMARY_CHARS)}"
            )
        if capability_lines:
            sections.append("【你现在的能力】\n" + "\n".join(capability_lines))

        if current_persona_prompt:
            sections.append(f"【当前会话风格】\n{current_persona_prompt}")

        if core_memory:
            sections.append(
                f"【核心记忆】\n"
                f"{self._truncate_text(core_memory, self.MAX_RUNTIME_CORE_MEMORY_CHARS)}"
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
