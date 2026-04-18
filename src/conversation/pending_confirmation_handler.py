from __future__ import annotations

from typing import Awaitable, Callable, Optional


GENERIC_APPROVAL_WORDS = {"审批通过", "通过", "确认", "同意", "确定"}
GENERIC_REJECTION_WORDS = {"拒绝", "不同意", "不通过", "取消", "算了"}


class PendingConfirmationHandler:
    """Handle pending command/change confirmations before the normal message flow."""

    def __init__(
        self,
        *,
        reply_service,
        fast_path_router,
        run_sync_func: Callable[..., Awaitable[str]],
    ) -> None:
        self.reply_service = reply_service
        self.fast_path_router = fast_path_router
        self._run_sync = run_sync_func

    async def handle(self, user_id: str, content: str) -> Optional[str]:
        workspace_resolution_reply = await self._run_sync(
            self.fast_path_router.try_handle_pending_resolution,
            user_id,
            content,
        )
        if content in {"1", "2"} and workspace_resolution_reply is not None:
            return workspace_resolution_reply

        if content in GENERIC_APPROVAL_WORDS:
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await self._run_sync(self.reply_service.apply_pending_change, latest_change_id, user_id)
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await self._run_sync(self.reply_service.confirm_pending_command, latest_command_id, user_id)
            if workspace_resolution_reply is not None:
                return workspace_resolution_reply
            return "当前没有待审批的命令或修改。"

        if content in GENERIC_REJECTION_WORDS:
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await self._run_sync(self.reply_service.discard_pending_change, latest_change_id, user_id)
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await self._run_sync(self.reply_service.discard_pending_command, latest_command_id, user_id)
            if workspace_resolution_reply is not None:
                return workspace_resolution_reply
            return "当前没有待处理的命令或修改。"

        confirm_command_id = self._extract_confirmation_id(content, prefixes=("确认执行", "确定执行"))
        if confirm_command_id:
            return await self._run_sync(self.reply_service.confirm_pending_command, confirm_command_id, user_id)

        if content in {"确认执行", "确定执行"}:
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await self._run_sync(self.reply_service.confirm_pending_command, latest_command_id, user_id)
            return "当前没有待确认执行的命令。"

        discard_command_id = self._extract_confirmation_id(content, prefixes=("取消执行",))
        if discard_command_id:
            return await self._run_sync(self.reply_service.discard_pending_command, discard_command_id, user_id)

        if content == "取消执行":
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await self._run_sync(self.reply_service.discard_pending_command, latest_command_id, user_id)
            return "当前没有待取消的命令。"

        apply_change_id = self._extract_confirmation_id(content, prefixes=("确认修改", "确定修改"))
        if apply_change_id:
            return await self._run_sync(self.reply_service.apply_pending_change, apply_change_id, user_id)

        if content in {"确认修改", "确定修改"}:
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await self._run_sync(self.reply_service.apply_pending_change, latest_change_id, user_id)
            return "当前没有待确认的修改。"

        discard_change_id = self._extract_confirmation_id(content, prefixes=("取消修改",))
        if discard_change_id:
            return await self._run_sync(self.reply_service.discard_pending_change, discard_change_id, user_id)

        if content == "取消修改":
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await self._run_sync(self.reply_service.discard_pending_change, latest_change_id, user_id)
            return "当前没有待取消的修改。"

        if workspace_resolution_reply is not None:
            return workspace_resolution_reply

        return None

    def _extract_confirmation_id(self, content: str, prefixes: tuple[str, ...]) -> str | None:
        normalized = (content or "").strip()
        for prefix in prefixes:
            if not normalized.startswith(prefix):
                continue
            suffix = normalized[len(prefix) :].strip()
            if suffix and all(ch.isalnum() or ch in {"_", "-"} for ch in suffix):
                return suffix
        return None
