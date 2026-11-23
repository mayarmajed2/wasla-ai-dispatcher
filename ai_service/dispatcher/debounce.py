"""3-Second Async Debounce Buffer for Fragmented WhatsApp Messages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class MessageDebouncer:
    """Aggregates rapid consecutive messages from the same phone_number into one turn."""

    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = window_seconds
        self._buffers: dict[str, list[str]] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._futures: dict[str, list[asyncio.Future[Any]]] = {}

    async def enqueue(
        self,
        phone_number: str,
        message: str,
        handler: Callable[[str, str], Awaitable[Any]],
    ) -> Any:
        self._buffers.setdefault(phone_number, []).append(message.strip())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._futures.setdefault(phone_number, []).append(fut)

        existing = self._tasks.get(phone_number)
        if existing and not existing.done():
            existing.cancel()

        async def _flush() -> None:
            try:
                await asyncio.sleep(self.window_seconds)
                msgs = self._buffers.pop(phone_number, [])
                waiters = self._futures.pop(phone_number, [])
                self._tasks.pop(phone_number, None)
                combined = "\n".join(m for m in msgs if m)
                result = await handler(phone_number, combined)
                for w in waiters:
                    if not w.done():
                        w.set_result(result)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                waiters = self._futures.pop(phone_number, [])
                self._buffers.pop(phone_number, None)
                self._tasks.pop(phone_number, None)
                for w in waiters:
                    if not w.done():
                        w.set_exception(exc)

        self._tasks[phone_number] = asyncio.create_task(_flush())
        return await fut
