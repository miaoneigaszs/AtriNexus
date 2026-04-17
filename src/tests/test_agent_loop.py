"""PR12 自建 agent loop 聚焦测试。

用 FakeProvider 模拟 provider.stream 的事件序列，覆盖：
- 单轮完成（无工具）
- 一轮工具调用 → 给出结果 → 二轮完成
- 工具不存在 → 错误结果消息
- before_tool_call 阻断
- 取消信号在工具边界生效
- max_iterations 截停
- consume_stream 正确累积 text + tool_calls + usage

不发任何真实 HTTP 请求。
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from typing import AsyncIterator, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agent_runtime.agent_loop import LoopResult, ToolEvent, run_agent_loop
from src.agent_runtime.context_engine import DefaultCompressor
from src.agent_runtime.hooks import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    NoopAgentHooks,
    OnResponseContext,
    TransformContextContext,
)
from src.agent_runtime.tool_catalog import RegisteredTool
from src.agent_runtime.user_runtime import UserRuntimeRegistry
from src.ai.providers.base import ProviderAdapter, ProviderRequest
from src.ai.types import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
    UserMessage,
    Usage,
)


def _run(coro):
    return asyncio.run(coro)


# ── 测试帮助 ────────────────────────────────────────────────────────────


class FakeProvider(ProviderAdapter):
    """每次 stream() 按 scripts 中下一段事件序列吐出去。"""

    def __init__(self, scripts: List[List[StreamEvent]]):
        self.scripts = list(scripts)
        self.requests: List[ProviderRequest] = []

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if not self.scripts:
            raise RuntimeError("FakeProvider: scripts 用尽，agent loop 多调了一次")
        events = self.scripts.pop(0)
        for event in events:
            yield event


def _text_then_done(text: str, prompt: int = 10, completion: int = 5) -> List[StreamEvent]:
    return [
        TextDelta(text=text),
        StreamDone(stop_reason="stop", usage=Usage(prompt_tokens=prompt, completion_tokens=completion)),
    ]


def _tool_call_then_done(call_id: str, name: str, args_json: str) -> List[StreamEvent]:
    return [
        ToolCallDelta(index=0, id=call_id, name=name, args_delta=args_json),
        StreamDone(stop_reason="tool_calls", usage=Usage(prompt_tokens=20, completion_tokens=4)),
    ]


def _make_tool(name: str, return_value: str = "OK") -> RegisteredTool:
    async def handler(args: Dict) -> str:
        return f"{return_value}: {args}"

    return RegisteredTool(
        spec=ToolSpec(
            name=name,
            description=f"fake tool {name}",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        handler=handler,
    )


# ── 测试 ────────────────────────────────────────────────────────────────


class AgentLoopHappyPathTest(unittest.TestCase):
    def test_single_turn_no_tools(self):
        provider = FakeProvider([_text_then_done("Hello!", prompt=12, completion=3)])

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="deepseek-ai/DeepSeek-V3",
                system_prompt="sys",
                initial_messages=[UserMessage(content="hi")],
                tools=[],
                hooks=NoopAgentHooks(),
                max_iterations=3,
            )

        result = _run(scenario())
        self.assertEqual(result.text, "Hello!")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual(result.usage.prompt_tokens, 12)
        self.assertEqual(result.usage.completion_tokens, 3)
        self.assertEqual(len(provider.requests), 1)

    def test_tool_call_then_completion(self):
        tool = _make_tool("get_time", return_value="11:11")
        provider = FakeProvider([
            _tool_call_then_done("c1", "get_time", "{}"),
            _text_then_done("现在是 11:11", prompt=5, completion=4),
        ])

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="deepseek-ai/DeepSeek-V3",
                system_prompt="sys",
                initial_messages=[UserMessage(content="时间")],
                tools=[tool],
                hooks=NoopAgentHooks(),
                max_iterations=3,
            )

        result = _run(scenario())
        self.assertEqual(result.text, "现在是 11:11")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.tool_events), 1)
        self.assertEqual(result.tool_events[0].status, "ok")
        self.assertIn("11:11", result.tool_events[0].result)
        # 第二轮请求里应该有 4 条消息：sys + user + assistant + tool
        second_request = provider.requests[1]
        roles = [m.role for m in second_request.messages]
        self.assertEqual(roles[-2:], ["assistant", "tool"])

    def test_unknown_tool_returns_error_to_model(self):
        provider = FakeProvider([
            _tool_call_then_done("c1", "no_such_tool", "{}"),
            _text_then_done("已知工具不足", prompt=3, completion=4),
        ])

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="deepseek-ai/DeepSeek-V3",
                system_prompt="sys",
                initial_messages=[UserMessage(content="x")],
                tools=[],
                hooks=NoopAgentHooks(),
                max_iterations=3,
            )

        result = _run(scenario())
        self.assertEqual(len(result.tool_events), 1)
        self.assertEqual(result.tool_events[0].status, "error")
        self.assertIn("不存在", result.tool_events[0].result)


class AgentLoopHookTest(unittest.TestCase):
    class BlockingHooks(NoopAgentHooks):
        async def before_tool_call(self, ctx):
            if ctx.tool_name == "blocked":
                return BeforeToolCallResult(block=True, reason="不允许调用 blocked")
            return None

    def test_before_tool_blocks_execution(self):
        tool = _make_tool("blocked", return_value="should not run")
        provider = FakeProvider([
            _tool_call_then_done("c1", "blocked", "{}"),
            _text_then_done("OK 我不调用了", prompt=2, completion=2),
        ])

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="m",
                system_prompt="sys",
                initial_messages=[UserMessage(content="x")],
                tools=[tool],
                hooks=self.BlockingHooks(),
                max_iterations=3,
            )

        result = _run(scenario())
        self.assertEqual(result.tool_events[0].status, "blocked")
        self.assertEqual(result.tool_events[0].result, "不允许调用 blocked")

    class TruncatingHooks(NoopAgentHooks):
        async def after_tool_call(self, ctx):
            return AfterToolCallResult(content="[shaped]")

    def test_after_tool_overrides_content(self):
        tool = _make_tool("any", return_value="long output text " * 10)
        provider = FakeProvider([
            _tool_call_then_done("c1", "any", "{}"),
            _text_then_done("done", prompt=1, completion=1),
        ])

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="m",
                system_prompt="sys",
                initial_messages=[UserMessage(content="x")],
                tools=[tool],
                hooks=self.TruncatingHooks(),
                max_iterations=3,
            )

        result = _run(scenario())
        self.assertEqual(result.tool_events[0].result, "[shaped]")


class AgentLoopAbortTest(unittest.TestCase):
    def test_abort_before_first_invoke_returns_cancelled(self):
        provider = FakeProvider([_text_then_done("never")])

        async def scenario():
            registry = UserRuntimeRegistry()
            async with registry.claim_run("u1"):
                await registry.abort("u1")
                return await run_agent_loop(
                    provider=provider,
                    model="m",
                    system_prompt="sys",
                    initial_messages=[UserMessage(content="x")],
                    tools=[],
                    hooks=NoopAgentHooks(),
                    max_iterations=3,
                )

        result = _run(scenario())
        self.assertTrue(result.cancelled)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(len(provider.requests), 0)

    def test_abort_between_tool_calls(self):
        # 第一轮 → tool call；abort 在 tool 执行前，因此第二轮不发生
        tool = _make_tool("any")
        provider = FakeProvider([
            _tool_call_then_done("c1", "any", "{}"),
        ])

        registry = UserRuntimeRegistry()

        class AbortingHooks(NoopAgentHooks):
            async def before_tool_call(self, ctx):
                # 在工具开始前置位 abort
                await registry.abort("u1")
                return None

        async def scenario():
            async with registry.claim_run("u1"):
                return await run_agent_loop(
                    provider=provider,
                    model="m",
                    system_prompt="sys",
                    initial_messages=[UserMessage(content="x")],
                    tools=[tool],
                    hooks=AbortingHooks(),
                    max_iterations=3,
                )

        result = _run(scenario())
        self.assertTrue(result.cancelled)


class AgentLoopBoundaryTest(unittest.TestCase):
    def test_max_iterations_hits_safety_cap(self):
        # 反复让模型调用工具，永不收敛 → 被 max_iterations 卡住
        tool = _make_tool("any")
        scripts = []
        # max_iterations=3 → loop 内消耗 3 个 tool_call 脚本
        for _ in range(3):
            scripts.append(_tool_call_then_done("c1", "any", "{}"))
        # 收尾轮（max_iterations 后会再发一次给模型机会）
        scripts.append(_text_then_done("最终回复", prompt=1, completion=1))
        provider = FakeProvider(scripts)

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="m",
                system_prompt="sys",
                initial_messages=[UserMessage(content="x")],
                tools=[tool],
                hooks=NoopAgentHooks(),
                max_iterations=3,
            )

        result = _run(scenario())
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(result.iterations, 3)
        # 被强制结束后还有一次收尾调用
        self.assertEqual(result.text, "最终回复")


class AgentLoopContextEngineTest(unittest.TestCase):
    def test_context_engine_receives_usage(self):
        engine = DefaultCompressor(context_length=10_000)
        provider = FakeProvider([_text_then_done("ok", prompt=300, completion=20)])

        async def scenario():
            return await run_agent_loop(
                provider=provider,
                model="m",
                system_prompt="sys",
                initial_messages=[UserMessage(content="x")],
                tools=[],
                hooks=NoopAgentHooks(),
                context_engine=engine,
                max_iterations=2,
            )

        _run(scenario())
        self.assertEqual(engine.last_prompt_tokens, 300)
        self.assertEqual(engine.last_completion_tokens, 20)


if __name__ == "__main__":
    unittest.main()
