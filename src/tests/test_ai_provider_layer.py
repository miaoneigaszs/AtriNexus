"""PR11 自建 ai/ provider 层聚焦测试。

覆盖：
- types: dataclass to_openai_dict 形态、tool_call args JSON 化、Usage merge
- registry: 已登记 model 命中、前缀匹配、未命中走默认
- stream: SSE 行解析、StreamAccumulator 聚合 chunk → text + tool_call + usage
- openai_compat: payload 构造、HTTP 错误转 StreamError、流式 happy path（mock httpx）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from typing import AsyncIterator, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.ai.providers.base import ProviderRequest
from src.ai.providers.openai_compat import OpenAICompatProvider
from src.ai.registry import (
    DEFAULT_CAPABILITIES,
    ModelCapabilities,
    all_models,
    get_capabilities,
    register,
)
from src.ai.stream import StreamAccumulator, iter_sse_lines, stream_openai_chunks
from src.ai.types import (
    AssistantMessage,
    ImageContent,
    StreamDone,
    StreamError,
    SystemMessage,
    TextContent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
    messages_to_openai,
    tools_to_openai,
)


def _run(coro):
    return asyncio.run(coro)


# ── 测试帮助 ────────────────────────────────────────────────────────────


async def _async_iter(items: List[bytes]) -> AsyncIterator[bytes]:
    for item in items:
        yield item


# ── types ────────────────────────────────────────────────────────────────


class TypesSerializationTest(unittest.TestCase):
    def test_user_message_string_content(self):
        msg = UserMessage(content="hi")
        self.assertEqual(msg.to_openai_dict(), {"role": "user", "content": "hi"})

    def test_user_message_multimodal(self):
        msg = UserMessage(content=[
            TextContent(text="describe this"),
            ImageContent(image_url="https://x.png", detail="low"),
        ])
        out = msg.to_openai_dict()
        self.assertEqual(out["content"][0]["type"], "text")
        self.assertEqual(out["content"][1]["image_url"]["url"], "https://x.png")
        self.assertEqual(out["content"][1]["image_url"]["detail"], "low")

    def test_assistant_message_with_tool_calls(self):
        msg = AssistantMessage(
            content="let me check",
            tool_calls=[ToolCall(id="c1", name="read_file", args={"path": "a.md"})],
        )
        out = msg.to_openai_dict()
        self.assertEqual(out["content"], "let me check")
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(out["tool_calls"][0]["function"]["arguments"]),
            {"path": "a.md"},
        )

    def test_tool_result_message(self):
        msg = ToolResultMessage(tool_call_id="c1", content="ok", name="read_file")
        out = msg.to_openai_dict()
        self.assertEqual(out["role"], "tool")
        self.assertEqual(out["tool_call_id"], "c1")
        self.assertEqual(out["name"], "read_file")

    def test_messages_to_openai_helper(self):
        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
        ]
        out = messages_to_openai(msgs)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["role"], "user")

    def test_tool_spec_dict(self):
        spec = ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        out = spec.to_openai_dict()
        self.assertEqual(out["function"]["name"], "read_file")
        self.assertIn("path", out["function"]["parameters"]["properties"])

    def test_usage_merge(self):
        a = Usage(prompt_tokens=10, completion_tokens=2)
        b = Usage(prompt_tokens=5, completion_tokens=3, cache_read_tokens=8)
        merged = a.merge(b)
        self.assertEqual(merged.prompt_tokens, 15)
        self.assertEqual(merged.completion_tokens, 5)
        self.assertEqual(merged.total_tokens, 20)
        self.assertEqual(merged.cache_read_tokens, 8)


# ── registry ────────────────────────────────────────────────────────────


class RegistryTest(unittest.TestCase):
    def test_known_model_hit(self):
        cap = get_capabilities("deepseek-ai/DeepSeek-V3.2")
        self.assertEqual(cap.provider, "openai_compat")
        self.assertGreaterEqual(cap.context_length, 64_000)

    def test_unknown_model_falls_back_to_default(self):
        cap = get_capabilities("totally-made-up-model-x")
        self.assertEqual(cap, DEFAULT_CAPABILITIES)

    def test_prefix_match(self):
        # 注册一个简单 prefix 然后用更长的 id 查
        register(ModelCapabilities(model_id="my-prefix-test", provider="openai_compat", context_length=4_000))
        cap = get_capabilities("my-prefix-test-v9-finetune")
        self.assertEqual(cap.context_length, 4_000)

    def test_reasoning_model_blocks_tools(self):
        cap = get_capabilities("deepseek-reasoner")
        self.assertFalse(cap.supports_tools)
        self.assertTrue(cap.is_reasoning)

    def test_anthropic_supports_cache_control(self):
        cap = get_capabilities("claude-3-5-sonnet")
        self.assertTrue(cap.supports_cache_control)

    def test_all_models_returns_dict(self):
        models = all_models()
        self.assertIsInstance(models, dict)
        self.assertIn("deepseek-ai/deepseek-v3", models)


# ── stream parsing ──────────────────────────────────────────────────────


class StreamSseLineTest(unittest.TestCase):
    def test_lines_split_across_chunks(self):
        async def collect():
            chunks = [b"data: ", b"hello\n\ndata: world\n", b"\n"]
            return [line async for line in iter_sse_lines(_async_iter(chunks))]
        lines = _run(collect())
        self.assertEqual(lines, ["data: hello", "data: world"])

    def test_comments_skipped(self):
        async def collect():
            chunks = [b": ping\n\ndata: real\n\n"]
            return [line async for line in iter_sse_lines(_async_iter(chunks))]
        lines = _run(collect())
        self.assertEqual(lines, ["data: real"])


class StreamAccumulatorTest(unittest.TestCase):
    def test_text_chunk_accumulation(self):
        acc = StreamAccumulator()
        acc.feed_chunk({"choices": [{"delta": {"content": "Hel"}}]})
        acc.feed_chunk({"choices": [{"delta": {"content": "lo"}}]})
        acc.feed_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        self.assertEqual(acc.collected_text(), "Hello")
        self.assertEqual(acc.stop_reason, "stop")

    def test_tool_call_accumulation_then_finalization(self):
        acc = StreamAccumulator()
        acc.feed_chunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{\"pa"}}
        ]}}]})
        acc.feed_chunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "th\":\"a.md\"}"}}
        ]}}]})
        acc.feed_chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        calls = acc.collected_tool_calls()
        self.assertEqual(len(calls), 1)
        call_id, name, args = calls[0]
        self.assertEqual(call_id, "c1")
        self.assertEqual(name, "read_file")
        self.assertEqual(args, {"path": "a.md"})

    def test_two_concurrent_tool_calls_by_index(self):
        acc = StreamAccumulator()
        # 两个 tool call 交替送
        acc.feed_chunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "f1", "arguments": "{}"}},
            {"index": 1, "id": "b", "function": {"name": "f2", "arguments": "{\"x\":1}"}},
        ]}}]})
        acc.feed_chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        calls = acc.collected_tool_calls()
        self.assertEqual([c[1] for c in calls], ["f1", "f2"])
        self.assertEqual(calls[1][2], {"x": 1})

    def test_tool_call_invalid_json_dropped(self):
        acc = StreamAccumulator()
        acc.feed_chunk({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "x", "function": {"name": "f", "arguments": "{not json"}}
        ]}}]})
        self.assertEqual(acc.collected_tool_calls(), [])

    def test_usage_extracted_from_chunk(self):
        acc = StreamAccumulator()
        acc.feed_chunk({"choices": [], "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "prompt_cache_hit_tokens": 80,
        }})
        done = acc.finalize()
        self.assertEqual(done.usage.prompt_tokens, 100)
        self.assertEqual(done.usage.completion_tokens, 30)
        self.assertEqual(done.usage.cache_read_tokens, 80)

    def test_finalize_without_usage_uses_empty(self):
        acc = StreamAccumulator()
        done = acc.finalize()
        self.assertIsInstance(done, StreamDone)
        self.assertIsInstance(done.usage, Usage)
        self.assertEqual(done.usage.total_tokens, 0)


class StreamDriverTest(unittest.TestCase):
    def test_full_pipeline_text_only(self):
        async def collect():
            payload = (
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
                b"data: {\"choices\":[{\"delta\":{\"content\":\" you\"},\"finish_reason\":\"stop\"}]}\n\n"
                b"data: [DONE]\n\n"
            )
            return [e async for e in stream_openai_chunks(_async_iter([payload]))]
        events = _run(collect())
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        done = next(e for e in events if isinstance(e, StreamDone))
        self.assertEqual(text, "Hi you")
        self.assertEqual(done.stop_reason, "stop")

    def test_provider_error_chunk_emits_stream_error(self):
        async def collect():
            payload = (
                b"data: {\"error\":{\"message\":\"rate_limited\"}}\n\n"
                b"data: [DONE]\n\n"
            )
            return [e async for e in stream_openai_chunks(_async_iter([payload]))]
        events = _run(collect())
        self.assertTrue(any(isinstance(e, StreamError) and "rate_limited" in e.message for e in events))
        self.assertTrue(any(isinstance(e, StreamDone) for e in events))


# ── openai_compat provider ──────────────────────────────────────────────


class _FakeAsyncResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    async def aread(self):
        return self._body

    async def aiter_bytes(self):
        yield self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncClient:
    def __init__(self, response: _FakeAsyncResponse):
        self.response = response
        self.last_request = None

    def stream(self, method, url, *, json=None, headers=None):
        self.last_request = {"method": method, "url": url, "json": json, "headers": headers}
        return self.response

    async def aclose(self):
        return None


class OpenAICompatProviderTest(unittest.TestCase):
    def _request(self, model="deepseek-ai/DeepSeek-V3"):
        return ProviderRequest(
            model=model,
            messages=[UserMessage(content="hi")],
            tools=[],
            temperature=0.7,
            max_tokens=512,
        )

    def test_payload_includes_required_fields(self):
        body = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n"
            b"data: [DONE]\n\n"
        )
        client = _FakeAsyncClient(_FakeAsyncResponse(200, body))
        provider = OpenAICompatProvider(api_key="k", base_url="https://api.test/v1", client=client)

        async def consume():
            return [e async for e in provider.stream(self._request())]

        events = _run(consume())
        self.assertTrue(any(isinstance(e, TextDelta) for e in events))
        self.assertTrue(any(isinstance(e, StreamDone) for e in events))
        # 验证 url + headers + payload
        sent = client.last_request
        self.assertEqual(sent["method"], "POST")
        self.assertEqual(sent["url"], "https://api.test/v1/chat/completions")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer k")
        self.assertTrue(sent["json"]["stream"])
        self.assertEqual(sent["json"]["model"], "deepseek-ai/DeepSeek-V3")
        self.assertEqual(sent["json"]["temperature"], 0.7)
        self.assertEqual(sent["json"]["max_tokens"], 512)
        self.assertNotIn("tools", sent["json"])  # 没传 tools 时不带

    def test_http_error_becomes_stream_error(self):
        body = b'{"error": {"message": "invalid api key"}}'
        client = _FakeAsyncClient(_FakeAsyncResponse(401, body))
        provider = OpenAICompatProvider(api_key="bad", base_url="https://api.test/v1", client=client)

        async def consume():
            return [e async for e in provider.stream(self._request())]

        events = _run(consume())
        errors = [e for e in events if isinstance(e, StreamError)]
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid api key", errors[0].message)
        self.assertTrue(any(isinstance(e, StreamDone) for e in events))

    def test_supports_tools_reads_registry(self):
        provider = OpenAICompatProvider(api_key="k", base_url="https://api.test/v1", client=_FakeAsyncClient(
            _FakeAsyncResponse(200, b"data: [DONE]\n\n")
        ))
        self.assertTrue(provider.supports_tools("deepseek-ai/DeepSeek-V3"))
        self.assertFalse(provider.supports_tools("deepseek-reasoner"))

    def test_constructor_validates_inputs(self):
        with self.assertRaises(ValueError):
            OpenAICompatProvider(api_key="", base_url="x")
        with self.assertRaises(ValueError):
            OpenAICompatProvider(api_key="x", base_url="")


if __name__ == "__main__":
    unittest.main()
