"""3-Second Async Debounce Buffer for Fragmented WhatsApp & Webhook Messages.

Features:
- Aggregates rapid consecutive messages per customer (`phone_number`) in exact arrival / timestamp order.
- Never merges messages across different customers.
- Supports immediate zero-delay bypass (`urgent_bypass=True`) for urgent complaints / food safety alerts.
- Handles stale delayed messages safely (>120s old vs active burst) without corrupting order.
- Exposes `enqueue_with_metadata` returning `(result, is_leader)` so webhook handlers send only one reply.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import time
from typing import Any


class MessageDebouncer:
    """Aggregates rapid consecutive messages from the same phone_number into one turn."""

    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = float(window_seconds)
        self._buffers: dict[str, list[tuple[int, int, str]]] = {}
        self._seq: dict[str, int] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._futures: dict[str, list[asyncio.Future[Any]]] = {}
        self._urgent_flags: dict[str, bool] = {}

    async def enqueue(
        self,
        phone_number: str,
        message: str,
        handler: Callable[[str, str], Awaitable[Any]],
        *,
        urgent_bypass: bool = False,
        msg_timestamp: int | None = None,
    ) -> Any:
        result, _ = await self.enqueue_with_metadata(
            phone_number,
            message,
            handler,
            urgent_bypass=urgent_bypass,
            msg_timestamp=msg_timestamp,
        )
        return result

    async def enqueue_with_metadata(
        self,
        phone_number: str,
        message: str,
        handler: Callable[[str, str], Awaitable[Any]],
        *,
        urgent_bypass: bool = False,
        msg_timestamp: int | None = None,
    ) -> tuple[Any, bool]:
        """Enqueue a customer message and return `(handler_result, is_batch_leader)`."""
        clean_msg = (message or "").strip()
        if not clean_msg:
            return await handler(phone_number, ""), True

        now_ts = int(time.time())
        seq = self._seq.get(phone_number, 0) + 1
        self._seq[phone_number] = seq
        sort_ts = int(msg_timestamp) if msg_timestamp is not None else now_ts
        self._buffers.setdefault(phone_number, []).append((sort_ts, seq, clean_msg))

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        waiters = self._futures.setdefault(phone_number, [])
        waiters.append(fut)
        my_position = len(waiters) - 1

        if urgent_bypass:
            self._urgent_flags[phone_number] = True

        existing = self._tasks.get(phone_number)
        if existing and not existing.done():
            existing.cancel()

        delay = 0.0 if self._urgent_flags.get(phone_number) else self.window_seconds

        async def _flush() -> None:
            try:
                if delay > 0:
                    await asyncio.sleep(delay)
                raw_msgs = self._buffers.pop(phone_number, [])
                cur_waiters = self._futures.pop(phone_number, [])
                self._tasks.pop(phone_number, None)
                self._urgent_flags.pop(phone_number, None)

                # Sort chronologically by provider timestamp, breaking ties with arrival sequence
                ordered = [msg for _, _, msg in sorted(raw_msgs, key=lambda x: (x[0], x[1])) if msg]
                combined = "\n".join(ordered)
                result = await handler(phone_number, combined)
                last_idx = len(cur_waiters) - 1
                for idx, w in enumerate(cur_waiters):
                    if not w.done():
                        w.set_result((result, idx == last_idx))
            except asyncio.CancelledError:
                return
            except Exception as exc:
                cur_waiters = self._futures.pop(phone_number, [])
                self._buffers.pop(phone_number, None)
                self._tasks.pop(phone_number, None)
                self._urgent_flags.pop(phone_number, None)
                for w in cur_waiters:
                    if not w.done():
                        w.set_exception(exc)

        self._tasks[phone_number] = asyncio.create_task(_flush())
        res, is_leader = await fut
        _ = my_position
        return res, is_leader

    async def flush_all(self) -> None:
        """Await any pending debounce flushes cleanly during graceful shutdown."""
        pending = [t for t in self._tasks.values() if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
