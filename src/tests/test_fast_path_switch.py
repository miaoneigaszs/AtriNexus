"""PR13 的行为测试：env 总开关、/agent 前缀、trajectory 观测字段。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_config.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("fast_path_config_under_test", CONFIG_PATH)
CONFIG_MODULE = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC and CONFIG_SPEC.loader
sys.modules["fast_path_config_under_test"] = CONFIG_MODULE
CONFIG_SPEC.loader.exec_module(CONFIG_MODULE)

FAST_PATH_ENV_VAR = CONFIG_MODULE.FAST_PATH_ENV_VAR
FAST_PATH_MODE_DISABLED = CONFIG_MODULE.FAST_PATH_MODE_DISABLED
FAST_PATH_MODE_FULL = CONFIG_MODULE.FAST_PATH_MODE_FULL
FastPathOutcome = CONFIG_MODULE.FastPathOutcome
INTENT_DISABLED = CONFIG_MODULE.INTENT_DISABLED
INTENT_NONE = CONFIG_MODULE.INTENT_NONE
INTENT_PENDING_RESOLUTION = CONFIG_MODULE.INTENT_PENDING_RESOLUTION
read_fast_path_mode = CONFIG_MODULE.read_fast_path_mode
strip_agent_prefix = CONFIG_MODULE.strip_agent_prefix


from src.agent_runtime.trajectory import (
    build_trajectory_entry,
    record_fast_path_turn,
    record_turn,
)


ROUTER_PATH = Path(__file__).resolve().parents[1] / "conversation" / "fast_path_router.py"
MESSAGE_HANDLER_PATH = Path(__file__).resolve().parents[1] / "conversation" / "message_handler.py"


class FastPathOutcomeTest(unittest.TestCase):
    def test_miss_default_intent_is_none(self):
        outcome = FastPathOutcome.miss()
        self.assertIsNone(outcome.reply)
        self.assertEqual(outcome.intent, INTENT_NONE)

    def test_miss_preserves_disabled_intent(self):
        outcome = FastPathOutcome.miss(INTENT_DISABLED)
        self.assertIsNone(outcome.reply)
        self.assertEqual(outcome.intent, INTENT_DISABLED)

    def test_hit_carries_reply_and_intent(self):
        outcome = FastPathOutcome.hit("hello", INTENT_PENDING_RESOLUTION)
        self.assertEqual(outcome.reply, "hello")
        self.assertEqual(outcome.intent, INTENT_PENDING_RESOLUTION)


class ReadFastPathModeTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(FAST_PATH_ENV_VAR, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[FAST_PATH_ENV_VAR] = self._saved
        else:
            os.environ.pop(FAST_PATH_ENV_VAR, None)

    def test_default_is_full(self):
        self.assertEqual(read_fast_path_mode(), FAST_PATH_MODE_FULL)

    def test_disabled_recognized(self):
        os.environ[FAST_PATH_ENV_VAR] = "disabled"
        self.assertEqual(read_fast_path_mode(), FAST_PATH_MODE_DISABLED)

    def test_disabled_case_insensitive(self):
        os.environ[FAST_PATH_ENV_VAR] = "DISABLED"
        self.assertEqual(read_fast_path_mode(), FAST_PATH_MODE_DISABLED)

    def test_unknown_value_falls_back_to_full(self):
        os.environ[FAST_PATH_ENV_VAR] = "nonsense"
        self.assertEqual(read_fast_path_mode(), FAST_PATH_MODE_FULL)

    def test_blank_falls_back_to_full(self):
        os.environ[FAST_PATH_ENV_VAR] = "   "
        self.assertEqual(read_fast_path_mode(), FAST_PATH_MODE_FULL)


class StripAgentPrefixTest(unittest.TestCase):
    def test_bare_message_untouched(self):
        payload, bypass = strip_agent_prefix("看看 src")
        self.assertEqual(payload, "看看 src")
        self.assertFalse(bypass)

    def test_slash_agent_with_payload_strips_and_flags(self):
        payload, bypass = strip_agent_prefix("/agent 看看 src")
        self.assertEqual(payload, "看看 src")
        self.assertTrue(bypass)

    def test_slash_agent_alone_empties_payload(self):
        payload, bypass = strip_agent_prefix("/agent")
        self.assertEqual(payload, "")
        self.assertTrue(bypass)

    def test_leading_whitespace_before_slash_tolerated(self):
        payload, bypass = strip_agent_prefix("  /agent 帮我改一下")
        self.assertEqual(payload, "帮我改一下")
        self.assertTrue(bypass)

    def test_non_prefix_slash_agent_as_content_ignored(self):
        payload, bypass = strip_agent_prefix("/agentic behaviour")
        self.assertEqual(payload, "/agentic behaviour")
        self.assertFalse(bypass)

    def test_none_input_safe(self):
        payload, bypass = strip_agent_prefix(None)
        self.assertIsNone(payload)
        self.assertFalse(bypass)


class RouterEnvGateSourceTest(unittest.TestCase):
    """try_handle 的 env 短路应立即返回——源码文本断言。"""

    def test_disabled_branch_returns_disabled_outcome(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("read_fast_path_mode() == FAST_PATH_MODE_DISABLED", source)
        self.assertIn("return FastPathOutcome.miss(INTENT_DISABLED)", source)

    def test_try_handle_has_no_resolver_interaction(self):
        # PR17 之后 try_handle 不再触碰 path_resolver（normalize_request_text 已删）；
        # 所有非空消息都交给 agent loop。
        source = ROUTER_PATH.read_text(encoding="utf-8")
        try_handle_start = source.index("def try_handle(")
        pending_start = source.index("def try_handle_pending_resolution(")
        try_handle_body = source[try_handle_start:pending_start]
        self.assertNotIn("self.path_resolver.begin", try_handle_body)
        self.assertNotIn("self.path_resolver.normalize_request_text", try_handle_body)

    def test_try_handle_returns_fast_path_outcome(self):
        source = ROUTER_PATH.read_text(encoding="utf-8")
        self.assertIn("def try_handle(self, user_id: str, message: str) -> FastPathOutcome:", source)


class MessageHandlerAgentPrefixSourceTest(unittest.TestCase):
    """message_handler.process_message 必须在 FastPath 调用前检测 /agent 前缀。"""

    def test_strip_prefix_called_before_fast_path_try_handle(self):
        source = MESSAGE_HANDLER_PATH.read_text(encoding="utf-8")
        strip_idx = source.index("strip_agent_prefix(content_trim)")
        fast_path_idx = source.index("self.fast_path_router.try_handle")
        self.assertLess(strip_idx, fast_path_idx)

    def test_bypass_flag_gates_fast_path_call(self):
        source = MESSAGE_HANDLER_PATH.read_text(encoding="utf-8")
        self.assertIn("if not bypass_fast_path:", source)

    def test_trajectory_recorded_for_pending_confirmation(self):
        source = MESSAGE_HANDLER_PATH.read_text(encoding="utf-8")
        self.assertIn("intent=INTENT_PENDING_RESOLUTION", source)

    def test_trajectory_recorded_for_fast_path_hit(self):
        source = MESSAGE_HANDLER_PATH.read_text(encoding="utf-8")
        self.assertIn("intent=outcome.intent", source)


class TrajectoryExtraFieldsTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("ATRINEXUS_TRAJECTORY_PATH", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["ATRINEXUS_TRAJECTORY_PATH"] = self._saved
        else:
            os.environ.pop("ATRINEXUS_TRAJECTORY_PATH", None)

    def test_build_entry_preserves_explicit_extra(self):
        entry = build_trajectory_entry(
            user_id="u1",
            user_message="q",
            assistant_reply="a",
            model="m",
            extra={"k": "v"},
        )
        self.assertEqual(entry["extra"], {"k": "v"})

    def test_record_turn_merges_fast_path_fields_into_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "traj.jsonl")
            os.environ["ATRINEXUS_TRAJECTORY_PATH"] = path
            record_turn(
                user_id="u1",
                user_message="q",
                assistant_reply="a",
                model="m",
                fast_path_hit=False,
                intent="none",
            )
            with open(path, "r", encoding="utf-8") as f:
                entry = json.loads(f.readline())
        self.assertEqual(entry["extra"]["fast_path_hit"], False)
        self.assertEqual(entry["extra"]["intent"], "none")

    def test_record_fast_path_turn_marks_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "traj.jsonl")
            os.environ["ATRINEXUS_TRAJECTORY_PATH"] = path
            record_fast_path_turn(
                user_id="u1",
                user_message="看看 src",
                assistant_reply="已列出 src 目录",
                intent="browse_list",
            )
            with open(path, "r", encoding="utf-8") as f:
                entry = json.loads(f.readline())
        self.assertEqual(entry["extra"]["fast_path_hit"], True)
        self.assertEqual(entry["extra"]["intent"], "browse_list")
        self.assertEqual(entry["model"], "fast_path")

    def test_record_turn_caller_extra_takes_precedence_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "traj.jsonl")
            os.environ["ATRINEXUS_TRAJECTORY_PATH"] = path
            record_turn(
                user_id="u1",
                user_message="q",
                assistant_reply="a",
                model="m",
                extra={"intent": "explicit", "fast_path_hit": True, "other": 1},
                fast_path_hit=False,
                intent="default",
            )
            with open(path, "r", encoding="utf-8") as f:
                entry = json.loads(f.readline())
        self.assertEqual(entry["extra"]["intent"], "explicit")
        self.assertEqual(entry["extra"]["fast_path_hit"], True)
        self.assertEqual(entry["extra"]["other"], 1)

    def test_record_turn_skips_when_trajectory_disabled(self):
        os.environ.pop("ATRINEXUS_TRAJECTORY_PATH", None)
        record_fast_path_turn(
            user_id="u1",
            user_message="q",
            assistant_reply="a",
            intent="none",
        )


if __name__ == "__main__":
    unittest.main()
