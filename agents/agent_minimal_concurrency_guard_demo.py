from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, TypeVar


T = TypeVar("T")
Message = dict[str, Any]
ContentBlock = dict[str, Any]


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class AgentConfig:
    max_concurrent_runs: int | None = None


@dataclass
class QueueEntry:
    task: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]
    enqueued_at: float


@dataclass
class LaneState:
    lane: str
    active: int = 0
    max_concurrent: int = 1
    queue: Deque[QueueEntry] = field(default_factory=deque)


class SessionManager:
    """最小会话存储，仅用于演示 append/get 行为。"""

    def __init__(self) -> None:
        self._messages: dict[str, list[Message]] = defaultdict(list)

    async def append(self, session_key: str, message: Message) -> None:
        self._messages[session_key].append(message)

    def get(self, session_key: str) -> list[Message]:
        return list(self._messages.get(session_key, []))


_lanes: dict[str, LaneState] = {}


def get_lane_state(lane: str) -> LaneState:
    state = _lanes.get(lane)
    if state is None:
        state = LaneState(lane=lane)
        _lanes[lane] = state
    return state


async def _run_lane_entry(lane: str, entry: QueueEntry) -> None:
    try:
        result = await entry.task()
        entry.future.set_result(result)
    except Exception as exc:
        entry.future.set_exception(exc)
    finally:
        state = get_lane_state(lane)
        state.active -= 1
        drain_lane(lane)


def drain_lane(lane: str) -> None:
    state = get_lane_state(lane)

    if state.active == 0 and not state.queue and lane.startswith("session:"):
        _lanes.pop(lane, None)
        return

    while state.active < state.max_concurrent and state.queue:
        entry = state.queue.popleft()
        state.active += 1
        asyncio.create_task(_run_lane_entry(lane, entry))


def set_lane_concurrency(lane: str, max_concurrent: int) -> None:
    state = get_lane_state(lane)
    state.max_concurrent = max(1, int(max_concurrent))
    drain_lane(lane)


async def enqueue_in_lane(lane: str, task: Callable[[], Awaitable[T]]) -> T:
    state = get_lane_state(lane)
    future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
    state.queue.append(
        QueueEntry(
            task=task,
            future=future,
            enqueued_at=time.perf_counter(),
        )
    )
    drain_lane(lane)
    return await future


def resolve_session_lane(session_key: str) -> str:
    cleaned = session_key.strip() or "main"
    return cleaned if cleaned.startswith("session:") else f"session:{cleaned}"


def resolve_global_lane(lane: str | None = None) -> str:
    cleaned = (lane or "").strip()
    return cleaned or "main"


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


def install_session_tool_result_guard(session_manager: SessionManager) -> Any:
    """给 SessionManager.append 打补丁，自动补缺失的 tool_result。"""

    if getattr(session_manager, "_tool_result_guard_installed", False):
        return session_manager._tool_result_guard

    original_append = session_manager.append
    pending_by_session: dict[str, dict[str, str | None]] = defaultdict(dict)

    async def flush_pending_tool_results(session_key: str) -> None:
        pending = pending_by_session.get(session_key)
        if not pending:
            return

        synthetic_results = [
            make_missing_tool_result(tool_use_id, tool_name)
            for tool_use_id, tool_name in pending.items()
        ]
        pending.clear()

        await original_append(
            session_key,
            {
                "role": "user",
                "content": synthetic_results,
                "timestamp": now_ms(),
            },
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
            await flush_pending_tool_results(session_key)

        if pending and tool_calls:
            await flush_pending_tool_results(session_key)

        await original_append(session_key, message)

        for tool_use_id, tool_name in tool_calls:
            pending[tool_use_id] = tool_name

    session_manager.append = guarded_append  # type: ignore[method-assign]

    guard = type(
        "ToolResultGuard",
        (),
        {
            "flush_pending_tool_results": flush_pending_tool_results,
            "get_pending_ids": lambda self, session_key: list(
                pending_by_session.get(session_key, {}).keys()
            ),
        },
    )()

    session_manager._tool_result_guard_installed = True
    session_manager._tool_result_guard = guard
    return guard


class MiniAgentDemo:
    """
    对齐 TypeScript 的两行核心初始化:

        global_lane = resolve_global_lane()
        set_lane_concurrency(global_lane, config.max_concurrent_runs ?? 4)
        self.tool_result_guard = install_session_tool_result_guard(self.sessions)
    """

    def __init__(self, config: AgentConfig) -> None:
        self.sessions = SessionManager()

        global_lane = resolve_global_lane()
        set_lane_concurrency(
            global_lane,
            4 if config.max_concurrent_runs is None else config.max_concurrent_runs,
        )

        self.tool_result_guard = install_session_tool_result_guard(self.sessions)

    async def run_in_lanes(
        self,
        session_key: str,
        label: str,
        duration_seconds: float,
        started_at: float,
    ) -> str:
        async def actual_task() -> str:
            elapsed = time.perf_counter() - started_at
            print(f"{elapsed:05.2f}s start {label:<12} session={session_key}")
            print(f"{elapsed:05.2f}s sta_run_lane_entryrt {label:<12} session={session_key}")
            await asyncio.sleep(duration_seconds)
            elapsed = time.perf_counter() - started_at
            print(f"{elapsed:05.2f}s end   {label:<12} session={session_key}")
            return label

        session_lane = resolve_session_lane(session_key)
        global_lane = resolve_global_lane()
        return await enqueue_in_lane(
            session_lane,
            lambda: enqueue_in_lane(global_lane, actual_task),
        )


def format_message(message: Message) -> str:
    role = message["role"]
    content = message["content"]

    if isinstance(content, str):
        return f"{role}: {content}"

    parts: list[str] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "tool_use":
            parts.append(
                f"tool_use(id={block.get('id')}, name={block.get('name')}, input={block.get('input')})"
            )
        elif block_type == "tool_result":
            parts.append(
                "tool_result("
                f"tool_use_id={block.get('tool_use_id')}, "
                f"name={block.get('name')}, "
                f"content={block.get('content')!r})"
            )
        elif block_type == "text":
            parts.append(f"text({block.get('text')!r})")
        else:
            parts.append(repr(block))
    return f"{role}: [{', '.join(parts)}]"


async def demo_global_lane() -> None:
    print("\n=== Demo 1: global lane 并发 + session lane 串行 ===")
    agent = MiniAgentDemo(AgentConfig(max_concurrent_runs=2))
    started_at = time.perf_counter()

    await asyncio.gather(
        agent.run_in_lanes("session-a", "A-1", 1.0, started_at),
        agent.run_in_lanes("session-a", "A-2", 0.3, started_at),
        agent.run_in_lanes("session-b", "B-1", 0.8, started_at),
        agent.run_in_lanes("session-c", "C-1", 0.2, started_at),
    )

    print("观察点:")
    print("- A-1 和 A-2 属于同一 session，所以它们不会同时运行。")
    print("- 全局并发被设成 2，所以最多只有两个任务同时执行。")


async def demo_tool_result_guard() -> None:
    print("\n=== Demo 2: tool result guard 自动补缺失结果 ===")
    agent = MiniAgentDemo(AgentConfig())
    session_key = "tool-guard-demo"

    await agent.sessions.append(
        session_key,
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我准备调用 read_file 工具。"},
                {
                    "type": "tool_use",
                    "id": "toolu_demo_1",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
            ],
            "timestamp": now_ms(),
        },
    )

    print("pending tool ids:", agent.tool_result_guard.get_pending_ids(session_key))

    await agent.sessions.append(
        session_key,
        {
            "role": "assistant",
            "content": "工具结果还没回来，但 assistant 又继续说话了。",
            "timestamp": now_ms(),
        },
    )

    print("session transcript:")
    for index, message in enumerate(agent.sessions.get(session_key), start=1):
        print(f"{index}. {format_message(message)}")

    print("观察点:")
    print("- 第二条 assistant 消息到来前，guard 发现上一条 tool_use 还没有配套的 tool_result。")
    print("- guard 会先插入一条 synthetic tool_result，再追加新的 assistant 消息。")


async def main() -> None:
    await demo_global_lane()
    await demo_tool_result_guard()


if __name__ == "__main__":
    asyncio.run(main())
