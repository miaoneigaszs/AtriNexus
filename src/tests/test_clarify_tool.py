"""PR22 — clarify 工具行为 + agent_loop 集成。

直接测 `_run_clarify_tool` handler + clarify_store contextvar 流。agent_loop
集成用 stub provider 跑一次 loop，保证 clarify 调用后 stop_reason == "clarify"
且 text == 用户看到的澄清文本。
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agent_runtime.agent_loop import run_agent_loop
from src.agent_runtime.clarify_store import (
    CLARIFY_PENDING,
    mark_clarify,
    reset_clarify,
    take_clarify,
)
from src.agent_runtime.hooks import AgentHooks
from src.agent_runtime.tool_catalog import RegisteredTool, _run_clarify_tool
from src.ai.providers.base import ProviderAdapter, ProviderRequest
from src.ai.types import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
    Usage,
)


TOOL_CATALOG_PATH = Path(__file__).resolve().parents[1] / "agent_runtime" / "tool_catalog.py"


# ── 单元测试：handler + contextvar ────────────────────────────────────────


class ClarifyHandlerTest(unittest.TestCase):
    def setUp(self):
        reset_clarify()
        self.addCleanup(reset_clarify)

    def test_plain_question_marks_clarify(self):
        reply = _run_clarify_tool({"question": "要改哪个文件？"})
        self.assertEqual(reply, "要改哪个文件？")
        self.assertEqual(CLARIFY_PENDING.get(), "要改哪个文件？")

    def test_question_with_options_renders_numbered_list(self):
        reply = _run_clarify_tool(
            {"question": "选哪个？", "options": ["README.md", "docs/intro.md"]}
        )
        self.assertIn("选哪个？", reply)
        self.assertIn("1. README.md", reply)
        self.assertIn("2. docs/intro.md", reply)
        self.assertEqual(CLARIFY_PENDING.get(), reply)

    def test_empty_question_returns_error_and_does_not_mark(self):
        reply = _run_clarify_tool({"question": ""})
        self.assertIn("失败", reply)
        self.assertIsNone(CLARIFY_PENDING.get())

    def test_whitespace_only_options_are_stripped_out(self):
        reply = _run_clarify_tool(
            {"question": "q", "options": ["", "   ", "real", None]}
        )
        self.assertIn("real", reply)
        self.assertNotIn("1. \n", reply)

    def test_options_non_list_ignored_gracefully(self):
        reply = _run_clarify_tool({"question": "q", "options": "not a list"})
        self.assertEqual(reply, "q")


class ClarifyStoreTest(unittest.TestCase):
    def setUp(self):
        reset_clarify()

    def test_take_returns_none_when_unmarked(self):
        self.assertIsNone(take_clarify())

    def test_mark_then_take_clears_value(self):
        mark_clarify("hello")
        self.assertEqual(take_clarify(), "hello")
        self.assertIsNone(take_clarify())

    def test_reset_clears(self):
        mark_clarify("hi")
        reset_clarify()
        self.assertIsNone(take_clarify())


# ── 集成测试：agent_loop 看到 CLARIFY 后终止 ────────────────────────────


class _StubProvider(ProviderAdapter):
    """按预设脚本返回流式 StreamEvent 序列：第 n 次调用 stream 返回 scripts[n]。"""

    def __init__(self, scripts: List[List[StreamEvent]]):
        self.scripts = scripts
        self.calls = 0

    def stream(self, request: ProviderRequest):
        idx = self.calls
        self.calls += 1
        script = self.scripts[idx] if idx < len(self.scripts) else []

        async def gen():
            for event in script:
                yield event

        return gen()


def _text_event(text: str) -> StreamEvent:
    return TextDelta(text=text)


def _tool_call_event(call_id: str, name: str, args: Dict[str, Any]) -> StreamEvent:
    import json

    return ToolCallDelta(
        index=0,
        id=call_id,
        name=name,
        args_delta=json.dumps(args, ensure_ascii=False),
    )


def _stop_event(reason: str = "stop") -> StreamEvent:
    return StreamDone(stop_reason=reason, usage=Usage())


class _NoopHooks(AgentHooks):
    pass


class ClarifyLoopIntegrationTest(unittest.TestCase):
    def setUp(self):
        reset_clarify()
        self.addCleanup(reset_clarify)

    def _clarify_tool(self) -> RegisteredTool:
        async def handler(args: Dict[str, Any]) -> str:
            return _run_clarify_tool(args)

        return RegisteredTool(
            spec=ToolSpec(
                name="clarify",
                description="ask clarifying question",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question"],
                },
            ),
            handler=handler,
        )

    def test_clarify_call_terminates_loop_with_stop_reason_clarify(self):
        # Script: agent first responds with a clarify tool call. No second turn needed.
        script = [
            [
                _tool_call_event(
                    "c1",
                    "clarify",
                    {"question": "改哪个文件？", "options": ["README.md", "CHANGELOG.md"]},
                ),
                _stop_event("tool_calls"),
            ],
        ]
        provider = _StubProvider(script)

        result = asyncio.run(
            run_agent_loop(
                provider=provider,
                model="stub",
                system_prompt="sys",
                initial_messages=[],
                tools=[self._clarify_tool()],
                hooks=_NoopHooks(),
                max_iterations=4,
            )
        )

        self.assertEqual(result.stop_reason, "clarify")
        self.assertIn("改哪个文件？", result.text)
        self.assertIn("1. README.md", result.text)
        # Did NOT go past 1 iteration (no second provider call needed).
        self.assertEqual(provider.calls, 1)
        # Contextvar cleared after being taken.
        self.assertIsNone(CLARIFY_PENDING.get())

    def test_non_clarify_tool_call_does_not_terminate(self):
        # Script: first turn calls `other_tool`; second turn stops with plain text.
        async def other_handler(args: Dict[str, Any]) -> str:
            return "ok"

        other_tool = RegisteredTool(
            spec=ToolSpec(
                name="other_tool",
                description="no-op",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            handler=other_handler,
        )
        script = [
            [
                _tool_call_event("x1", "other_tool", {}),
                _stop_event("tool_calls"),
            ],
            [
                _text_event("done"),
                _stop_event("stop"),
            ],
        ]
        provider = _StubProvider(script)

        result = asyncio.run(
            run_agent_loop(
                provider=provider,
                model="stub",
                system_prompt="sys",
                initial_messages=[],
                tools=[self._clarify_tool(), other_tool],
                hooks=_NoopHooks(),
                max_iterations=4,
            )
        )

        self.assertNotEqual(result.stop_reason, "clarify")
        self.assertEqual(result.text, "done")
        self.assertEqual(provider.calls, 2)


# ── 工具目录源码级断言 ─────────────────────────────────────────────────────


class ClarifyToolRegistrationTest(unittest.TestCase):
    def test_core_section_registers_clarify_tool(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn('name="clarify"', source)
        self.assertIn("- clarify:", source)
        self.assertIn("Ask the user a clarifying question", source)


if __name__ == "__main__":
    unittest.main()
