"""只读命令规划测试。

单元测试覆盖 WorkspaceRuntime._build_command_plan，不实际运行 subprocess。
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
        # 没有 shell 操作符时走 direct 参数路径；env 只读检查只会
        # 在管道场景生效。非管道命令保持原有
        # 行为（使用 shell=False 的参数列表执行）。
        plan = self._plan("env FOO=bar find src")
        self.assertEqual(plan.mode, "direct")

    def test_pipeline_with_nonreadonly_segment_requires_confirm(self):
        # tee 会写输出文件，不在只读名单中
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
        # 即使包在看似只读的管道里，也优先命中确认规则。
        plan = self._plan("find src && rm -rf build")
        self.assertEqual(plan.mode, "confirm")
        self.assertEqual(plan.reason, "confirm-pattern")

    def test_absolute_path_readonly_bin_is_accepted(self):
        plan = self._plan("/usr/bin/find src | wc -l")
        self.assertEqual(plan.mode, "shell-safe")

    def test_unknown_bin_is_still_direct_without_operators(self):
        # 没有 shell 操作符时走普通参数路径，返回 `direct`。
        # 非管道命令的策略保持不变。
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

    def test_expansion_includes_common_inspection_bins(self):
        """常见巡检命令应视为只读。"""
        policy = CommandExecutionPolicy()
        expected_inspection_bins = {
            # 输出类
            "echo", "printf",
            # 文件读
            "cat", "head", "tail", "nl",
            # 系统状态
            "free", "uptime", "top", "ps", "df", "lsblk",
            # 硬件
            "nproc", "lscpu", "uname",
            # 身份
            "whoami", "hostname", "date", "id",
            # 路径
            "ls", "pwd",
            # 文本处理
            "grep", "egrep", "fgrep", "sort", "uniq", "cut",
            # 加速器
            "nvidia-smi", "rocm-smi", "vainfo",
        }
        self.assertTrue(expected_inspection_bins.issubset(set(policy.safe_bin_readonly)))

    def test_expansion_excludes_write_capable_bins(self):
        """具备写入能力的命令必须排除在只读白名单之外。"""
        policy = CommandExecutionPolicy()
        forbidden = {"sed", "awk", "xargs", "tee", "dd", "cp", "chmod", "chown"}
        self.assertEqual(set(policy.safe_bin_readonly) & forbidden, set())


class ExpandedReadonlyPipelineTest(unittest.TestCase):
    """常见只读巡检管道应使用 shell-safe 模式。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runtime = WorkspaceRuntime(str(Path(self._tmp.name)))

    def _plan(self, command: str):
        return self.runtime._build_command_plan(command)

    def test_system_inspection_chain_is_shell_safe(self):
        plan = self._plan("echo '== uptime =='; uptime; echo; free -h")
        self.assertEqual(plan.mode, "shell-safe")
        self.assertEqual(plan.reason, "safe-readonly")

    def test_memory_and_cpu_short_chain_is_shell_safe(self):
        plan = self._plan("free -h && uptime && cat /proc/loadavg")
        self.assertEqual(plan.mode, "shell-safe")

    def test_top_batch_piped_to_head_is_shell_safe(self):
        plan = self._plan("top -bn1 | head -n 8")
        self.assertEqual(plan.mode, "shell-safe")

    def test_gpu_query_is_shell_safe(self):
        plan = self._plan("nvidia-smi --query-gpu=name,memory.used --format=csv")
        # 无管道操作符时走 direct
        self.assertEqual(plan.mode, "direct")

    def test_grep_sort_pipeline_is_shell_safe(self):
        plan = self._plan("cat /proc/meminfo | grep MemFree | sort")
        self.assertEqual(plan.mode, "shell-safe")

    def test_sed_inplace_still_requires_confirm(self):
        """sed 绝不入白名单；即使和只读命令拼在一起也要过确认。"""
        plan = self._plan("cat foo | sed -i 's/a/b/' foo")
        self.assertEqual(plan.mode, "confirm")

    def test_tee_write_still_requires_confirm(self):
        plan = self._plan("uptime | tee uptime.log")
        self.assertEqual(plan.mode, "confirm")

    def test_rm_pattern_wins_over_readonly_chain(self):
        plan = self._plan("free -h && rm -rf build")
        self.assertEqual(plan.mode, "confirm")
        self.assertEqual(plan.reason, "confirm-pattern")


if __name__ == "__main__":
    unittest.main()
