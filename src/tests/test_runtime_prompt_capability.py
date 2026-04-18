"""PR16 — 概览能力下沉到系统 prompt 的行为测试。

验证 `PromptManager.build_runtime_prompt` 在收到 `current_mode` + tool 档位 +
详细工具摘要时，会拼出带有"你现在的能力"snapshot 的 runtime 段——
让 agent 被问到"有哪些工具"/"当前是什么模式"时能直接作答，不再依赖
hardcoded overview 分支。
"""

from __future__ import annotations

import unittest

from src.prompting.prompt_manager import PromptManager


class RuntimePromptCapabilitySnapshotTest(unittest.TestCase):
    def setUp(self):
        self.pm = PromptManager(root_dir=".")

    def test_capability_block_includes_mode_profile_and_tool_summary(self):
        prompt = self.pm.build_runtime_prompt(
            persona_prompt="",
            tool_profile="workspace_read",
            tool_profiles=["core", "workspace_read"],
            tool_summary="[基础] 读本地时间。\n[读文件] read_file / list_directory",
            core_memory=None,
            current_mode="work",
        )
        self.assertIn("【你现在的能力】", prompt)
        self.assertIn("当前模式：work", prompt)
        self.assertIn("当前能力档位：workspace_read", prompt)
        self.assertIn("当前工具组：core, workspace_read", prompt)
        self.assertIn("这些工具当前分别能做", prompt)
        self.assertIn("read_file", prompt)

    def test_capability_block_skipped_when_all_inputs_missing(self):
        prompt = self.pm.build_runtime_prompt(
            persona_prompt="",
            tool_profile=None,
            tool_profiles=None,
            tool_summary="",
            core_memory=None,
            current_mode=None,
        )
        self.assertNotIn("【你现在的能力】", prompt)

    def test_mode_alone_still_emits_capability_block(self):
        prompt = self.pm.build_runtime_prompt(
            persona_prompt="",
            tool_profile=None,
            tool_profiles=None,
            tool_summary="",
            core_memory=None,
            current_mode="companion",
        )
        self.assertIn("【你现在的能力】", prompt)
        self.assertIn("当前模式：companion", prompt)

    def test_capability_block_appears_before_persona_and_memory(self):
        prompt = self.pm.build_runtime_prompt(
            persona_prompt="风格示例",
            tool_profile="workspace_read",
            tool_profiles=["core"],
            tool_summary="工具摘要",
            core_memory="记忆内容",
            current_mode="work",
        )
        capability_idx = prompt.index("【你现在的能力】")
        persona_idx = prompt.index("【当前会话风格】")
        memory_idx = prompt.index("【核心记忆】")
        self.assertLess(capability_idx, persona_idx)
        self.assertLess(persona_idx, memory_idx)


if __name__ == "__main__":
    unittest.main()
