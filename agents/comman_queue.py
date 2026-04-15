import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TypeVar, Awaitable, Callable, Any, Deque

T = TypeVar("T")

@dataclass()
class QueueEntry:
    task: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]
    enqueued_at: float
    warn_after_ms: int
    on_wait: Callable[[int, int], None] | None = None


@dataclass
class LaneState:
    lane: str
    active: int = 0
    max_concurrent: int = 1
    queue: Deque[QueueEntry] = field(default_factory=deque)

_lanes: dict[str, LaneState] = {}


def get_lane_state(lane: str) -> LaneState:
    state = _lanes.get(lane)
    if state is None:
        state = LaneState(lane=lane)
        _lanes[lane] = state

    return state


def drain_Lane(lane: str) -> None:
    state = get_lane_state(lane)
    if state.active == 0 and not state.queue and lane.startswith("session:"):
        _lanes.pop(lane, None)
        return

    while state.active < state.max_concurrent and state.queue:
        entry = state.queue.popleft()
        state.active += 1

        wait_ms = int((time.perf_counter() - entry.enqueued_at) * 1000)
        if wait_ms > entry.warn_after_ms and entry.on_wait is not None:
            entry.on_wait(wait_ms, len(state.queue))

        asyncio.create_task(_run_lane_entry(lane, entry))


async def _run_lane_entry(lane: str, entry: QueueEntry) -> None:
    try:
        result = await entry.task()
        entry.future.set_result(result)
    except Exception as exc:
        entry.future.set_exception(exc)
    finally:
        state = get_lane_state(lane)
        state.active -= 1
        drain_Lane(lane)

async def enqueue_in_lane(
        lane: str,
        task: Callable[[], Awaitable[T]],
        *,
        warn_after_ms: int = 2000,
        on_wait: Callable[[int, int], None] | None = None,
) -> T:
    state = get_lane_state(lane)
    future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
    state.queue.append(
        QueueEntry(
            task = task,
            future = future,
            enqueued_at=time.perf_counter(),
            warn_after_ms=warn_after_ms,
            on_wait=on_wait,
        )
    )
    drain_Lane(lane)
    return await future

def set_lane_concurrency(lane: str, max_concurrent: int) -> None:
    state  = get_lane_state(lane)
    state.max_concurrent = max(1, int(max_concurrent))
    drain_Lane(lane)


def resolve_session_lane(session_key: str) -> str:
    cleaned =  session_key.strip() or "main"
    return cleaned if cleaned.startswith("session:") else f"session:{cleaned}"

def delete_lane(lane: str) -> bool:
    state = _lanes.get(lane)
    if state is None:
        return False
    if state.active > 0 or state.queue:
        return False
    del  _lanes[lane]
    return True

def resolve_global_lane(lane: str | None = None) -> str:
    cleaned = (lane or "").strip()
    return cleaned or "main"