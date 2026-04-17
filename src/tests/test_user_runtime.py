"""PR9 steering/abort/follow-up 队列聚焦测试。

覆盖：
- PendingMessageQueue 两种 drain 模式
- UserRuntimeRegistry claim_run / abort / queue_follow_up 的生命周期
- CURRENT_ABORT_EVENT contextvar 在 claim_run 内外的可见性
- AgentToolGuard.before_tool_call 在 abort 被置位时抛 CancelledError
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agent_runtime.agent_tool_guard import AgentToolGuard
from src.agent_runtime.hooks import BeforeToolCallContext
from src.agent_runtime.user_runtime import (
    CURRENT_ABORT_EVENT,
    PendingMessageQueue,
    UserRuntimeRegistry,
    abort_requested,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeRuntime:
    SKIP_DIRS = {".git", "__pycache__"}
    workspace_root = "."

    def resolve_path_or_error(self, raw_path):
        return None, "not found"

    def iter_files(self, root):
        return []

    def to_relative(self, path):
        return str(path)


class _FakeCatalog:
    def __init__(self):
        self.runtime = _FakeRuntime()


class PendingMessageQueueTest(unittest.TestCase):
    def test_drain_all_clears_queue(self):
        q = PendingMessageQueue("all")
        q.enqueue("a")
        q.enqueue("b")
        q.enqueue("c")
        self.assertEqual(q.drain(), ["a", "b", "c"])
        self.assertFalse(q.has_items())

    def test_drain_one_at_a_time_keeps_rest(self):
        q = PendingMessageQueue("one-at-a-time")
        q.enqueue("a")
        q.enqueue("b")
        self.assertEqual(q.drain(), ["a"])
        self.assertTrue(q.has_items())
        self.assertEqual(q.drain(), ["b"])
        self.assertFalse(q.has_items())

    def test_drain_when_empty_returns_empty(self):
        q = PendingMessageQueue("all")
        self.assertEqual(q.drain(), [])

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            PendingMessageQueue("bogus")


class UserRuntimeRegistryTest(unittest.TestCase):
    def test_claim_run_toggles_is_running(self):
        async def scenario():
            registry = UserRuntimeRegistry()
            self.assertFalse(await registry.is_running("u1"))
            async with registry.claim_run("u1"):
                self.assertTrue(await registry.is_running("u1"))
            self.assertFalse(await registry.is_running("u1"))

        _run(scenario())

    def test_cannot_claim_twice(self):
        async def scenario():
            registry = UserRuntimeRegistry()
            async with registry.claim_run("u1"):
                with self.assertRaises(RuntimeError):
                    async with registry.claim_run("u1"):
                        pass

        _run(scenario())

    def test_abort_sets_event_and_clears_on_next_run(self):
        async def scenario():
            registry = UserRuntimeRegistry()
            # 没活跃 run 时 abort 返回 False
            self.assertFalse(await registry.abort("u1"))

            captured = {}

            async def run_agent():
                async with registry.claim_run("u1"):
                    captured["event"] = CURRENT_ABORT_EVENT.get()
                    await asyncio.sleep(0.05)
                    captured["aborted_midrun"] = abort_requested()

            task = asyncio.create_task(run_agent())
            await asyncio.sleep(0.01)  # 等 claim 成功
            self.assertTrue(await registry.abort("u1"))
            await task

            self.assertIsNotNone(captured["event"])
            self.assertTrue(captured["event"].is_set())
            self.assertTrue(captured["aborted_midrun"])

            # 下一次 run 的 abort event 是全新的
            async with registry.claim_run("u1"):
                new_event = CURRENT_ABORT_EVENT.get()
                self.assertIsNotNone(new_event)
                self.assertFalse(new_event.is_set())

        _run(scenario())

    def test_follow_up_queue(self):
        async def scenario():
            registry = UserRuntimeRegistry()
            self.assertEqual(await registry.queue_follow_up("u1", "m1"), 1)
            self.assertEqual(await registry.queue_follow_up("u1", "m2"), 2)
            drained = await registry.drain_follow_up("u1")
            self.assertEqual(drained, ["m1", "m2"])
            # drain 后队列清空
            self.assertEqual(await registry.drain_follow_up("u1"), [])

        _run(scenario())

    def test_context_var_cleared_outside_run(self):
        self.assertIsNone(CURRENT_ABORT_EVENT.get())
        self.assertFalse(abort_requested())


class AgentToolGuardAbortTest(unittest.TestCase):
    def setUp(self):
        self.guard = AgentToolGuard(_FakeCatalog())

    def test_before_tool_call_raises_on_abort(self):
        async def scenario():
            registry = UserRuntimeRegistry()
            async with registry.claim_run("u1"):
                # 立刻置位取消
                await registry.abort("u1")
                ctx = BeforeToolCallContext(tool_name="run_command", args={"command": "ls"}, call_id="c1")
                with self.assertRaises(asyncio.CancelledError):
                    await self.guard.before_tool_call(ctx)

        _run(scenario())

    def test_before_tool_call_passes_when_not_aborted(self):
        async def scenario():
            registry = UserRuntimeRegistry()
            async with registry.claim_run("u1"):
                ctx = BeforeToolCallContext(tool_name="run_command", args={"command": "ls"}, call_id="c1")
                # 未取消 → 正常返回 None（无修复无阻塞）
                result = await self.guard.before_tool_call(ctx)
                self.assertIsNone(result)

        _run(scenario())


if __name__ == "__main__":
    unittest.main()
