import asyncio
from typing import TypeVar, Optional, Generic, AsyncIterator, Callable

T = TypeVar("T")
R = TypeVar("R")
_SENTINEL = object()
class EventStream(Generic[T, R]):

    def __init__(self, is_complete:Callable[[T], bool], extract_result: Callable[[T], R]):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False
        self._is_complete: Callable[[T], bool] = is_complete
        self._extract_result: Callable[[T], R] = extract_result
        self._result_future: asyncio.Future = asyncio.get_running_loop().create_future()


    async def push(self, event: T) -> None:
        if self._done:
            raise RuntimeError('Event stream is already ended')
        await self._queue.put(event)

    async def end(self, result: Optional[R] = None) -> None:
        if self._done:
            return
        self._done = True
        if not self._result_future.done():
            self._result_future.set_result(result)
        await self._queue.put(_SENTINEL)

    async def result(self) -> R:
        return await self._result_future

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item


