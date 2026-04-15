from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


Message = dict[str, Any]
ContentBlock = dict[str, Any]


def now_ms() -> int:
    return int(time.time() * 1000)


def extract_tool_uses_from_assistant(message: Message) -> list[tuple[str, str | None]]:
    if message.get("role") != "assistant" or isinstance(message.get("content"), str):
        return []

    calls: list[tuple[str, str | None]] = []
    for block in message["content"]:
        if block.get("type") == "tool_use" and block.get("id"):
            calls.append((block["id"], block.get("name")))
    return calls


def extract_tool_result_ids(message: Message) -> list[str]:
    if message.get("role") != "user" or isinstance(message.get("content"), str):
        return []

    ids: list[str] = []
    for block in message["content"]:
        if block.get("type") == "tool_result" and block.get("tool_use_id"):
            ids.append(block["tool_use_id"])
    return ids


def make_missing_tool_result(tool_call_id: str, tool_name: str | None = None) -> ContentBlock:
    return {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "name": tool_name,
        "content": (
            "[openclaw-python-edition] missing tool result in session history; "
            "inserted synthetic error result for transcript repair."
        ),
    }


@dataclass
class ToolResultGuard:
    _original_append: Any
    _pending_by_session: dict[str, dict[str, str | None]]

    async def flush_pending_tool_results(self, session_key: str) -> None:
        pending = self._pending_by_session.get(session_key)
        if not pending:
            return

        results = [
            make_missing_tool_result(tool_use_id, tool_name)
            for tool_use_id, tool_name in pending.items()
        ]
        pending.clear()

        await self._original_append(
            session_key,
            {
                "role": "user",
                "content": results,
                "timestamp": now_ms(),
            },
        )

    def get_pending_ids(self, session_key: str) -> list[str]:
        return list(self._pending_by_session.get(session_key, {}).keys())


def install_session_tool_result_guard(session_manager: Any) -> ToolResultGuard:
    if getattr(session_manager, "_tool_result_guard_installed", False):
        return session_manager._tool_result_guard

    original_append = session_manager.append
    pending_by_session: dict[str, dict[str, str | None]] = defaultdict(dict)
    guard = ToolResultGuard(
        _original_append=original_append,
        _pending_by_session=pending_by_session,
    )

    async def guarded_append(session_key: str, message: Message) -> None:
        pending = pending_by_session[session_key]
        result_ids = extract_tool_result_ids(message)

        if result_ids:
            for tool_use_id in result_ids:
                pending.pop(tool_use_id, None)
            await original_append(session_key, message)
            return

        tool_calls = extract_tool_uses_from_assistant(message)

        if pending and not tool_calls:
            await guard.flush_pending_tool_results(session_key)

        if pending and tool_calls:
            await guard.flush_pending_tool_results(session_key)

        await original_append(session_key, message)

        for tool_use_id, tool_name in tool_calls:
            pending[tool_use_id] = tool_name

    session_manager.append = guarded_append
    session_manager._tool_result_guard_installed = True
    session_manager._tool_result_guard = guard
    return guard


__all__ = [
    "ToolResultGuard",
    "extract_tool_result_ids",
    "extract_tool_uses_from_assistant",
    "install_session_tool_result_guard",
    "make_missing_tool_result",
]
