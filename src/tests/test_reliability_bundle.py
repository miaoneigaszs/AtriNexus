"""PR7 的四个新工具（retry / prompt_cache / rate_limit / trajectory）聚焦测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.agent_runtime.prompt_cache import (
    apply_anthropic_cache_control,
    model_supports_cache_control,
)
from src.agent_runtime.trajectory import (
    build_trajectory_entry,
    save_trajectory,
    trajectory_enabled,
)
from src.platform_core.rate_limit import (
    format_rate_limit_compact,
    format_rate_limit_display,
    get_latest_state,
    parse_rate_limit_headers,
    record_latest_state,
)
from src.platform_core.retry_utils import jittered_backoff


class JitteredBackoffTest(unittest.TestCase):
    def test_first_attempt_in_expected_range(self):
        for _ in range(20):
            delay = jittered_backoff(1, base_delay=5.0, max_delay=120.0, jitter_ratio=0.5)
            self.assertGreaterEqual(delay, 5.0)
            self.assertLessEqual(delay, 5.0 + 2.5)

    def test_exponential_growth_with_cap(self):
        # attempt=5 的基础值 5 * 2^4 = 80；加抖动 <= 80 * 1.5
        for _ in range(20):
            delay = jittered_backoff(5, base_delay=5.0, max_delay=120.0)
            self.assertGreaterEqual(delay, 80.0)
            self.assertLessEqual(delay, 120.0 + 60.0)

    def test_cap_applied_at_high_attempts(self):
        # attempt=50 远超 cap；base_delay * 2^49 远超 max_delay，等于 max_delay + jitter
        for _ in range(5):
            delay = jittered_backoff(50, base_delay=5.0, max_delay=120.0)
            self.assertGreaterEqual(delay, 120.0)
            self.assertLessEqual(delay, 120.0 + 60.0)


class PromptCacheTest(unittest.TestCase):
    def test_claude_detected(self):
        self.assertTrue(model_supports_cache_control("claude-3-5-sonnet"))
        self.assertTrue(model_supports_cache_control("anthropic/claude-opus-4"))

    def test_non_anthropic_models_skip(self):
        self.assertFalse(model_supports_cache_control("gpt-4o"))
        self.assertFalse(model_supports_cache_control("deepseek-v3"))
        self.assertFalse(model_supports_cache_control(""))

    def test_cache_markers_on_system_and_last_three(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
        out = apply_anthropic_cache_control(messages)
        # system 被展平为 list 并带 cache_control
        system_content = out[0]["content"]
        self.assertIsInstance(system_content, list)
        self.assertEqual(system_content[0]["cache_control"]["type"], "ephemeral")

        # 非 system 消息的最后 3 条都应带 cache_control（位置 3/4/5）
        for idx in (3, 4, 5):
            tail = out[idx]["content"]
            self.assertTrue(any("cache_control" in str(x) for x in tail))

        # 前面的旧消息（位置 1/2）不带 cache_control
        for idx in (1, 2):
            self.assertNotIn("cache_control", str(out[idx]))

    def test_ttl_1h_marker(self):
        out = apply_anthropic_cache_control([{"role": "system", "content": "x"}], cache_ttl="1h")
        self.assertEqual(out[0]["content"][0]["cache_control"]["ttl"], "1h")

    def test_empty_input(self):
        self.assertEqual(apply_anthropic_cache_control([]), [])


class RateLimitTest(unittest.TestCase):
    def test_parse_none_when_no_headers(self):
        self.assertIsNone(parse_rate_limit_headers({}))
        self.assertIsNone(parse_rate_limit_headers({"content-type": "application/json"}))

    def test_parse_minute_window(self):
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "42",
            "x-ratelimit-reset-requests": "30",
        }
        state = parse_rate_limit_headers(headers, provider="demo")
        self.assertIsNotNone(state)
        self.assertTrue(state.has_data)
        self.assertEqual(state.requests_min.limit, 100)
        self.assertEqual(state.requests_min.remaining, 42)
        self.assertEqual(state.requests_min.used, 58)
        self.assertAlmostEqual(state.requests_min.usage_pct, 58.0)

    def test_latest_state_roundtrip(self):
        record_latest_state(None)  # reset not supported; just verify idempotency
        state = parse_rate_limit_headers(
            {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "900"}
        )
        record_latest_state(state)
        self.assertIs(get_latest_state(), state)

    def test_format_produces_ascii_bar(self):
        state = parse_rate_limit_headers(
            {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "50"}
        )
        display = format_rate_limit_display(state)
        self.assertIn("Requests/min", display)
        self.assertIn("50.0%", display)

        compact = format_rate_limit_compact(state)
        self.assertIn("RPM", compact)


class TrajectoryTest(unittest.TestCase):
    def test_disabled_by_default(self):
        # 取决于环境变量没设
        saved = os.environ.pop("ATRINEXUS_TRAJECTORY_PATH", None)
        try:
            self.assertFalse(trajectory_enabled())
        finally:
            if saved is not None:
                os.environ["ATRINEXUS_TRAJECTORY_PATH"] = saved

    def test_entry_structure(self):
        entry = build_trajectory_entry(
            user_id="u1",
            user_message="hi",
            assistant_reply="hello",
            model="claude-3",
            system_prompt="sys",
            tool_events=[
                {"name": "read_file", "args": {"path": "a.md"}, "result": "content"},
            ],
        )
        # 第一条 system，紧跟 human
        self.assertEqual(entry["conversations"][0]["from"], "system")
        self.assertEqual(entry["conversations"][1]["from"], "human")
        # 工具调用展开为 function_call + observation 两条
        froms = [c["from"] for c in entry["conversations"]]
        self.assertIn("function_call", froms)
        self.assertIn("observation", froms)
        # 最后一条 gpt
        self.assertEqual(entry["conversations"][-1]["from"], "gpt")
        self.assertEqual(entry["conversations"][-1]["value"], "hello")

    def test_save_appends_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "traj.jsonl")
            entry = build_trajectory_entry(
                user_id="u1", user_message="q", assistant_reply="a", model="m"
            )
            save_trajectory(entry, path=path)
            save_trajectory(entry, path=path)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(line) for line in lines]
            self.assertTrue(all(item["user_id"] == "u1" for item in parsed))


if __name__ == "__main__":
    unittest.main()
