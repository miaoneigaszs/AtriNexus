import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


MODULE_PATH = Path(__file__).resolve().parents[1] / "conversation" / "pending_confirmation_handler.py"
MODULE_SPEC = importlib.util.spec_from_file_location("pending_confirmation_handler", MODULE_PATH)
PENDING_CONFIRMATION_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC and MODULE_SPEC.loader
MODULE_SPEC.loader.exec_module(PENDING_CONFIRMATION_MODULE)
PendingConfirmationHandler = PENDING_CONFIRMATION_MODULE.PendingConfirmationHandler


async def _fake_run_sync(func, *args):
    return func(*args)


class _FakeReplyService:
    def __init__(self):
        self.latest_change_id = None
        self.latest_command_id = None
        self.calls = []

    def get_latest_pending_change_id(self, user_id):
        self.calls.append(("get_latest_pending_change_id", user_id))
        return self.latest_change_id

    def get_latest_pending_command_id(self, user_id):
        self.calls.append(("get_latest_pending_command_id", user_id))
        return self.latest_command_id

    def apply_pending_change(self, change_id, user_id):
        self.calls.append(("apply_pending_change", change_id, user_id))
        return f"apply:{change_id}:{user_id}"

    def discard_pending_change(self, change_id, user_id):
        self.calls.append(("discard_pending_change", change_id, user_id))
        return f"discard-change:{change_id}:{user_id}"

    def confirm_pending_command(self, command_id, user_id):
        self.calls.append(("confirm_pending_command", command_id, user_id))
        return f"confirm:{command_id}:{user_id}"

    def discard_pending_command(self, command_id, user_id):
        self.calls.append(("discard_pending_command", command_id, user_id))
        return f"discard-command:{command_id}:{user_id}"


class _FakeFastPathRouter:
    def __init__(self):
        self.reply = None
        self.calls = []

    def try_handle_pending_resolution(self, user_id, content):
        self.calls.append((user_id, content))
        return self.reply


class PendingConfirmationHandlerTest(unittest.TestCase):
    def setUp(self):
        self.reply_service = _FakeReplyService()
        self.fast_path_router = _FakeFastPathRouter()
        self.handler = PendingConfirmationHandler(
            reply_service=self.reply_service,
            fast_path_router=self.fast_path_router,
            run_sync_func=_fake_run_sync,
        )

    def _handle(self, content: str, user_id: str = "u1"):
        return asyncio.run(self.handler.handle(user_id, content))

    def test_generic_approval_prefers_latest_change(self):
        self.reply_service.latest_change_id = "chg-1"
        self.reply_service.latest_command_id = "cmd-1"

        result = self._handle("确认")

        self.assertEqual(result, "apply:chg-1:u1")
        self.assertIn(("apply_pending_change", "chg-1", "u1"), self.reply_service.calls)
        self.assertNotIn(("confirm_pending_command", "cmd-1", "u1"), self.reply_service.calls)

    def test_generic_rejection_falls_back_to_command(self):
        self.reply_service.latest_command_id = "cmd-2"

        result = self._handle("取消")

        self.assertEqual(result, "discard-command:cmd-2:u1")
        self.assertIn(("discard_pending_command", "cmd-2", "u1"), self.reply_service.calls)

    def test_explicit_confirm_command_id(self):
        result = self._handle("确认执行 cmd_42")
        self.assertEqual(result, "confirm:cmd_42:u1")

    def test_explicit_apply_change_id(self):
        result = self._handle("确定修改 change-9")
        self.assertEqual(result, "apply:change-9:u1")

    def test_explicit_discard_command_id(self):
        result = self._handle("取消执行 cmd-7")
        self.assertEqual(result, "discard-command:cmd-7:u1")

    def test_explicit_discard_change_id(self):
        result = self._handle("取消修改 change_5")
        self.assertEqual(result, "discard-change:change_5:u1")

    def test_no_pending_command_reply_is_preserved(self):
        result = self._handle("确认执行")
        self.assertEqual(result, "当前没有待确认执行的命令。")

    def test_no_pending_change_reply_is_preserved(self):
        result = self._handle("确认修改")
        self.assertEqual(result, "当前没有待确认的修改。")

    def test_generic_approval_without_pending_items_preserves_reply(self):
        result = self._handle("同意")
        self.assertEqual(result, "当前没有待审批的命令或修改。")

    def test_workspace_resolution_fallback_runs_last(self):
        self.fast_path_router.reply = "workspace-choice"

        result = self._handle("不是确认指令")

        self.assertEqual(result, "workspace-choice")
        self.assertEqual(self.fast_path_router.calls, [("u1", "不是确认指令")])

    def test_numeric_choice_prefers_workspace_resolution(self):
        self.reply_service.latest_change_id = "chg-1"
        self.fast_path_router.reply = "workspace-choice"

        result = self._handle("1")

        self.assertEqual(result, "workspace-choice")
        self.assertNotIn(("apply_pending_change", "chg-1", "u1"), self.reply_service.calls)

    def test_confirmation_word_still_prefers_pending_change_before_workspace(self):
        self.reply_service.latest_change_id = "chg-1"
        self.fast_path_router.reply = "workspace-choice"

        result = self._handle("确认")

        self.assertEqual(result, "apply:chg-1:u1")


if __name__ == "__main__":
    unittest.main()
