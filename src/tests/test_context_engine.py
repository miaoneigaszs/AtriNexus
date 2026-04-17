"""PR10 ContextEngine + DefaultCompressor 聚焦测试。

覆盖：
- DefaultCompressor 的 token 估算
- 阈值未到不压缩 / 超过阈值压缩
- 头尾保留 + 中段占位符
- 消息数过少（<= protect_first + protect_last）时不动
- update_from_response 兼容多种 usage 字段命名
- on_session_reset 重置计数
- update_model 切上下文长度
- get_status 字段齐备
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agent_runtime.context_engine import ContextEngine, DefaultCompressor


def _make_messages(n: int, chars: int = 100, role: str = "user") -> list:
    return [{"role": role, "content": "x" * chars} for _ in range(n)]


class DefaultCompressorTokenEstimateTest(unittest.TestCase):
    def test_empty_returns_zero(self):
        engine = DefaultCompressor()
        self.assertEqual(engine.estimate_tokens([]), 0)

    def test_string_content_estimated(self):
        engine = DefaultCompressor(chars_per_token=4)
        msgs = [{"role": "user", "content": "abcd" * 25}]  # 100 chars
        self.assertEqual(engine.estimate_tokens(msgs), 25)

    def test_list_content_aggregated(self):
        engine = DefaultCompressor(chars_per_token=4)
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "abcd" * 10}, "extra"]},
        ]
        # 40 chars + 5 chars = 45 → 11 tokens
        self.assertEqual(engine.estimate_tokens(msgs), 11)


class DefaultCompressorTriggerTest(unittest.TestCase):
    def test_below_threshold_no_compress(self):
        engine = DefaultCompressor(context_length=10_000)
        msgs = _make_messages(20, chars=50)
        # 估算 ~ 250 tokens，远低于 7500 (=10000*0.75) 阈值
        self.assertFalse(engine.should_compress(msgs))

    def test_above_threshold_triggers(self):
        engine = DefaultCompressor(context_length=200)
        msgs = _make_messages(20, chars=200)
        # 估算 ~ 1000 tokens，超过 150 阈值
        self.assertTrue(engine.should_compress(msgs))

    def test_too_few_messages_no_compress(self):
        engine = DefaultCompressor(context_length=200)  # threshold=150
        # 默认 protect_first_n=3 + protect_last_n=6 = 9；只给 9 条
        msgs = _make_messages(9, chars=500)
        self.assertFalse(engine.should_compress(msgs))


class DefaultCompressorCompressTest(unittest.TestCase):
    def test_compress_keeps_head_and_tail(self):
        engine = DefaultCompressor(context_length=200)
        engine.protect_first_n = 2
        engine.protect_last_n = 3
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ]
        compressed = engine.compress(msgs)
        # 保留前 2 + 占位符 1 + 后 3 = 6
        self.assertEqual(len(compressed), 6)
        # 头部
        self.assertEqual(compressed[0]["role"], "system")
        self.assertEqual(compressed[1]["content"], "u1")
        # 占位符
        self.assertEqual(compressed[2]["role"], "system")
        self.assertIn("省略", compressed[2]["content"])
        # 尾部
        self.assertEqual(compressed[-1]["content"], "a4")
        # compression_count 增加
        self.assertEqual(engine.compression_count, 1)

    def test_compress_returns_unchanged_when_too_few(self):
        engine = DefaultCompressor()
        msgs = _make_messages(5)
        compressed = engine.compress(msgs)
        self.assertIs(compressed, msgs)
        self.assertEqual(engine.compression_count, 0)

    def test_compress_empty_safe(self):
        engine = DefaultCompressor()
        self.assertEqual(engine.compress([]), [])


class ContextEngineLifecycleTest(unittest.TestCase):
    def test_update_from_response_openai_style(self):
        engine = DefaultCompressor()
        engine.update_from_response({"prompt_tokens": 500, "completion_tokens": 100})
        self.assertEqual(engine.last_prompt_tokens, 500)
        self.assertEqual(engine.last_completion_tokens, 100)
        self.assertEqual(engine.last_total_tokens, 600)

    def test_update_from_response_anthropic_style(self):
        engine = DefaultCompressor()
        engine.update_from_response({"input_tokens": 800, "output_tokens": 50})
        self.assertEqual(engine.last_prompt_tokens, 800)
        self.assertEqual(engine.last_completion_tokens, 50)

    def test_update_from_response_zero_skipped(self):
        engine = DefaultCompressor()
        engine.last_prompt_tokens = 999  # 已有非零值
        engine.update_from_response({"prompt_tokens": 0, "completion_tokens": 0})
        # 0/0 不覆盖
        self.assertEqual(engine.last_prompt_tokens, 999)

    def test_session_reset_clears_counters(self):
        engine = DefaultCompressor()
        engine.last_prompt_tokens = 100
        engine.compression_count = 5
        engine.on_session_reset()
        self.assertEqual(engine.last_prompt_tokens, 0)
        self.assertEqual(engine.compression_count, 0)

    def test_update_model_recalculates_threshold(self):
        engine = DefaultCompressor(context_length=10_000)
        self.assertEqual(engine.threshold_tokens, 7_500)
        engine.update_model("new-model", 200_000)
        self.assertEqual(engine.context_length, 200_000)
        self.assertEqual(engine.threshold_tokens, 150_000)

    def test_status_has_required_fields(self):
        engine = DefaultCompressor(context_length=1000)
        engine.update_from_response({"prompt_tokens": 500})
        status = engine.get_status()
        for field in (
            "engine",
            "last_prompt_tokens",
            "threshold_tokens",
            "context_length",
            "usage_percent",
            "compression_count",
        ):
            self.assertIn(field, status)
        self.assertEqual(status["engine"], "compressor")
        self.assertEqual(status["usage_percent"], 50.0)


class ContextEngineSubclassContractTest(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            ContextEngine()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
