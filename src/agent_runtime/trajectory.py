"""每轮对话的轨迹落盘。

把当轮 user/assistant/tool 交互以 ShareGPT 风格的 JSONL 追加到文件，
用于离线 eval、debug 与将来的微调数据。参考 hermes-agent/agent/trajectory.py。

默认关闭；把环境变量 `ATRINEXUS_TRAJECTORY_PATH` 指向文件路径即启用。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger("wecom")

_ENV_VAR = "ATRINEXUS_TRAJECTORY_PATH"


def trajectory_enabled() -> bool:
    return bool(os.getenv(_ENV_VAR, "").strip())


def _resolve_path() -> Optional[str]:
    raw = os.getenv(_ENV_VAR, "").strip()
    return raw or None


def build_trajectory_entry(
    *,
    user_id: str,
    user_message: str,
    assistant_reply: str,
    model: str,
    system_prompt: Optional[str] = None,
    tool_events: Optional[List[Dict[str, Any]]] = None,
    completed: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组装一条 ShareGPT 风格的 trajectory 记录。

    `tool_events` 为可选的工具调用序列，每项形如
    `{"name": str, "args": dict, "result": str, "status": "ok"|"error"}`。
    """
    conversations: List[Dict[str, str]] = []
    if system_prompt:
        conversations.append({"from": "system", "value": system_prompt})
    conversations.append({"from": "human", "value": user_message})

    if tool_events:
        for event in tool_events:
            conversations.append({
                "from": "function_call",
                "value": json.dumps(
                    {
                        "name": event.get("name", ""),
                        "args": event.get("args", {}),
                    },
                    ensure_ascii=False,
                ),
            })
            conversations.append({
                "from": "observation",
                "value": str(event.get("result", ""))[:4000],
            })

    conversations.append({"from": "gpt", "value": assistant_reply})

    entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "model": model,
        "completed": completed,
        "conversations": conversations,
    }
    if extra:
        entry["extra"] = extra
    return entry


def save_trajectory(entry: Dict[str, Any], path: Optional[str] = None) -> None:
    """追加一条 trajectory 到 JSONL；path 缺省读环境变量。"""
    target = path or _resolve_path()
    if not target:
        return

    try:
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("写 trajectory 失败 %s: %s", target, exc)


def record_turn(
    *,
    user_id: str,
    user_message: str,
    assistant_reply: str,
    model: str,
    system_prompt: Optional[str] = None,
    tool_events: Optional[List[Dict[str, Any]]] = None,
    completed: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """业务侧便捷入口：未启用时直接返回，启用时落盘一条。"""
    if not trajectory_enabled():
        return
    entry = build_trajectory_entry(
        user_id=user_id,
        user_message=user_message,
        assistant_reply=assistant_reply,
        model=model,
        system_prompt=system_prompt,
        tool_events=tool_events,
        completed=completed,
        extra=extra,
    )
    save_trajectory(entry)
