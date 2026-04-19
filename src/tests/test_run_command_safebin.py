"""PR20 — SAFE_BIN_READONLY: 只读命令 pipeline 直接放行（不需要确认）。

单元测试 `_build_command_plan` 的判定逻辑；不实际 subprocess.run 避免平台依赖。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent_runtime.runtime import (
    CommandExecutionPolicy,
    WorkspaceRuntime,
)


class ReadonlyPipelinePlanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runtime = WorkspaceRuntime(str(Path(self._tmp.name)))

    def _plan(self, command: str):
        return self.runtime._build_command_plan(command)

    def test_plain_readonly_bin_is_direct(self):
        plan = self._plan("find src -name '*.py'")
        self.assertEqual(plan.mode, "direct")

    def test_readonly_pipeline_is_shell_safe(self):
        plan = self._plan("find src -name '*.py' | wc -l")
        self.assertEqual(plan.mode, "shell-safe")
        self.assertEqual(plan.reason, "safe-readonly")

    def test_readonly_with_redirect_is_shell_safe(self):
        plan = self._plan("du -sh src > sizes.txt")
        self.assertEqual(plan.mode, "shell-safe")

    def test_readonly_chained_with_semicolon_is_shell_safe(self):
        plan = self._plan("which python; which pip")
        self.assertEqual(plan.mode, "shell-safe")

    def test_readonly_chained_with_and_is_shell_safe(self):
        plan = self._plan("stat src && stat src/tests")
        self.assertEqual(plan.mode, "shell-safe")

    def test_env_alone_is_readonly(self):
        plan = self._plan("env | wc -l")
        self.assertEqual(plan.mode, "shell-safe")

    def test_env_prefix_with_readonly_command_in_pipeline_passes(self):
        plan = self._plan("env FOO=bar find src | wc -l")
        self.assertEqual(plan.mode, "shell-safe")

    def test_env_prefix_with_nonreadonly_command_in_pipeline_requires_confirm(self):
        plan = self._plan("env FOO=bar make all | tee build.log")
        self.assertEqual(plan.mode, "confirm")
        self.assertEqual(plan.reason, "shell-operator")

    def test_env_prefix_without_pipeline_stays_direct(self):
        # No shell operators → direct argv path; the env-readonly check only
        # kicks in for pipelines. Non-pipeline commands keep their pre-PR20
        # behavior (argv exec with shell=False).
        plan = self._plan("env FOO=bar find src")
        self.assertEqual(plan.mode, "direct")

    def test_pipeline_with_nonreadonly_segment_requires_confirm(self):
        # tee writes output files — not in readonly list
        plan = self._plan("find src | tee out.txt")
        self.assertEqual(plan.mode, "confirm")

    def test_command_substitution_is_rejected(self):
        plan = self._plan("find $(pwd) -name '*.py'")
        self.assertEqual(plan.mode, "confirm")
        self.assertEqual(plan.reason, "shell-operator")

    def test_backtick_substitution_is_rejected(self):
        plan = self._plan("find `pwd` -name '*.py'")
        self.assertEqual(plan.mode, "confirm")

    def test_destructive_confirm_pattern_still_wins(self):
        # Even if wrapped in a readonly-looking pipeline, confirm_patterns trip first.
        plan = self._plan("find src && rm -rf build")
        self.assertEqual(plan.mode, "confirm")
        self.assertEqual(plan.reason, "confirm-pattern")

    def test_absolute_path_readonly_bin_is_accepted(self):
        plan = self._plan("/usr/bin/find src | wc -l")
        self.assertEqual(plan.mode, "shell-safe")

    def test_unknown_bin_is_still_direct_without_operators(self):
        # Without shell operators, we go the normal argv path, which returns `direct`.
        # Policy stays unchanged for non-pipeline commands.
        plan = self._plan("python -V")
        self.assertEqual(plan.mode, "direct")


class SafeBinReadonlyPolicyTest(unittest.TestCase):
    def test_default_policy_includes_all_roadmap_bins(self):
        policy = CommandExecutionPolicy()
        expected = {
            "find",
            "du",
            "wc",
            "stat",
            "file",
            "which",
            "env",
            "tree",
            "basename",
            "dirname",
            "realpath",
        }
        self.assertTrue(expected.issubset(set(policy.safe_bin_readonly)))


if __name__ == "__main__":
    unittest.main()
