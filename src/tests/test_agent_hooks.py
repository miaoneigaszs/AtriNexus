"""PR8 hook 化的聚焦测试。

覆盖：
- AgentHooks 协议与 NoopAgentHooks 的基本约定
- AgentToolGuard 的 before/after 流程（repair / validate / loop guard / shape）
- DefaultAgentHooks 的 transform_context（Anthropic 场景）和 on_response（rate limit 抓取）
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agent_runtime.agent_tool_guard import (
    MAX_TOOL_REPEAT_COUNT,
    AgentToolGuard,
)
from src.agent_runtime.default_hooks import DefaultAgentHooks
from src.agent_runtime.hooks import (
    AfterToolCallContext,
    AgentHooks,
    BeforeToolCallContext,
    NoopAgentHooks,
    OnResponseContext,
    TransformContextContext,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class _FakeCandidate:
    def __init__(self, path: str, is_file: bool = False, is_dir: bool = False):
        self.path = path
        self._is_file = is_file
        self._is_dir = is_dir

    def exists(self):
        return self._is_file or self._is_dir

    def is_file(self):
        return self._is_file

    def is_dir(self):
        return self._is_dir


class _FakeRuntime:
    """足够骗过 AgentToolGuard 的 runtime 替身。"""

    SKIP_DIRS = {".git", "__pycache__"}
    workspace_root = "."

    def resolve_path_or_error(self, raw_path):
        normalized = str(raw_path).replace('\\', '/').strip('/')
        if normalized == "README.md":
            return _FakeCandidate(normalized, is_file=True), None
        if normalized == "docs":
            return _FakeCandidate(normalized, is_dir=True), None
        return None, "not found"

    def iter_files(self, root):
        return [Path("README.md"), Path("docs/guide.md")]

    def to_relative(self, path):
        return str(path).replace('\\', '/')


class _FakeCatalog:
    def __init__(self):
        self.runtime = _FakeRuntime()


class NoopHooksProtocolTest(unittest.TestCase):
    def test_noop_satisfies_protocol(self):
        hooks = NoopAgentHooks()
        self.assertIsInstance(hooks, AgentHooks)

    def test_noop_returns_none_everywhere(self):
        hooks = NoopAgentHooks()
        before_ctx = BeforeToolCallContext(tool_name="x", args={}, call_id="c1")
        self.assertIsNone(_run(hooks.before_tool_call(before_ctx)))
        after_ctx = AfterToolCallContext(
            tool_name="x", args={}, call_id="c1", result_content="ok", is_error=False
        )
        self.assertIsNone(_run(hooks.after_tool_call(after_ctx)))
        self.assertIsNone(hooks.transform_context(TransformContextContext(messages=[], model="")))
        self.assertIsNone(hooks.on_response(OnResponseContext(model="", response=None)))


class AgentToolGuardHookTest(unittest.TestCase):
    def setUp(self):
        self.guard = AgentToolGuard(_FakeCatalog())

    def test_before_blocks_on_validation_failure(self):
        ctx = BeforeToolCallContext(tool_name="search_files", args={"query": ""}, call_id="c1")
        result = _run(self.guard.before_tool_call(ctx))
        self.assertIsNotNone(result)
        self.assertTrue(result.block)
        self.assertIn("search_files", result.reason)

    def test_before_repairs_args_without_blocking(self):
        ctx = BeforeToolCallContext(
            tool_name="run_command",
            args={"command": "  ls  "},
            call_id="c1",
        )
        result = _run(self.guard.before_tool_call(ctx))
        self.assertIsNotNone(result)
        self.assertFalse(result.block)
        self.assertEqual(result.repaired_args, {"command": "ls"})

    def test_before_passthrough_when_no_change(self):
        ctx = BeforeToolCallContext(
            tool_name="run_command",
            args={"command": "ls"},
            call_id="c1",
        )
        result = _run(self.guard.before_tool_call(ctx))
        self.assertIsNone(result)

    def test_before_repairs_workspace_path_with_shared_resolver(self):
        ctx = BeforeToolCallContext(
            tool_name="read_file",
            args={"path": "readme"},
            call_id="c1",
        )
        result = _run(self.guard.before_tool_call(ctx))
        self.assertIsNotNone(result)
        self.assertEqual(result.repaired_args, {"path": "README.md"})

    def test_loop_guard_blocks_after_repeats(self):
        token = self.guard.set_loop_state(self.guard.create_loop_state())
        try:
            ctx = BeforeToolCallContext(
                tool_name="run_command",
                args={"command": "git status"},
                call_id="c1",
            )
            # MAX_TOOL_REPEAT_COUNT 次允许，再多一次 block
            for _ in range(MAX_TOOL_REPEAT_COUNT):
                self.assertIsNone(_run(self.guard.before_tool_call(ctx)))
            blocked = _run(self.guard.before_tool_call(ctx))
            self.assertIsNotNone(blocked)
            self.assertTrue(blocked.block)
            self.assertIn("重复", blocked.reason)
        finally:
            self.guard.reset_loop_state(token)

    def test_after_shapes_and_truncates(self):
        long_dir = "\n".join(f"- [file] a{i}.txt" for i in range(100))
        ctx = AfterToolCallContext(
            tool_name="list_directory",
            args={"path": "."},
            call_id="c1",
            result_content=long_dir,
            is_error=False,
        )
        result = _run(self.guard.after_tool_call(ctx))
        self.assertIsNotNone(result)
        self.assertIn("其余", result.content)
        # 被截断，行数回到允许的上限以内（40 + 省略行）
        self.assertLessEqual(len(result.content.splitlines()), 41)

    def test_after_passthrough_when_result_short(self):
        ctx = AfterToolCallContext(
            tool_name="run_command",
            args={"command": "ls"},
            call_id="c1",
            result_content="short output",
            is_error=False,
        )
        result = _run(self.guard.after_tool_call(ctx))
        self.assertIsNone(result)


class DefaultAgentHooksTest(unittest.TestCase):
    def setUp(self):
        self.guard = AgentToolGuard(_FakeCatalog())
        self.hooks = DefaultAgentHooks(self.guard)

    def test_protocol_conformance(self):
        self.assertIsInstance(self.hooks, AgentHooks)

    def test_transform_context_activates_on_anthropic(self):
        ctx = TransformContextContext(
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
            ],
            model="claude-3-5-sonnet",
        )
        result = self.hooks.transform_context(ctx)
        self.assertIsNotNone(result)
        # system 消息被展平为 list 并带 cache_control
        sys_content = result.messages[0]["content"]
        self.assertIsInstance(sys_content, list)
        self.assertEqual(sys_content[0]["cache_control"]["type"], "ephemeral")

    def test_transform_context_noop_on_non_anthropic(self):
        ctx = TransformContextContext(
            messages=[{"role": "system", "content": "s"}],
            model="gpt-4o",
        )
        self.assertIsNone(self.hooks.transform_context(ctx))

    def test_on_response_records_rate_limit(self):
        from src.platform_core.rate_limit import get_latest_state, record_latest_state

        record_latest_state(None)  # 清掉残留
        ctx = OnResponseContext(
            model="any",
            response=None,
            response_metadata={
                "x-ratelimit-limit-requests": "50",
                "x-ratelimit-remaining-requests": "49",
                "x-ratelimit-reset-requests": "60",
            },
            duration_ms=120.0,
        )
        self.hooks.on_response(ctx)
        state = get_latest_state()
        self.assertIsNotNone(state)
        self.assertEqual(state.requests_min.limit, 50)

    def test_on_response_no_metadata_is_safe(self):
        ctx = OnResponseContext(model="any", response=None, response_metadata=None)
        # 不应抛异常
        self.hooks.on_response(ctx)


if __name__ == "__main__":
    unittest.main()
