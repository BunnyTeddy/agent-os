"""Status-reaction lifecycle on the failure path.

Regression coverage for #1560: ``failed()`` appended its token through
``_add_state`` and nothing ever reclaimed it, because ``completed()`` was the
only method that popped ``_active``. On the ``TaskQueueFullError`` path --
``received(msg)`` then ``failed(msg)``, then return -- the message was left
carrying a contradictory ✅/❌ pair forever, and the ``_active`` entry leaked
once per rejected message for the adapter's lifetime.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentos.channels._reactions import _BaseStatusReactor
from agentos.channels.types import IncomingMessage


class _RecordingReactor(_BaseStatusReactor):
    def __init__(self) -> None:
        super().__init__("test", _SilentLog())
        self.added: list[str] = []
        self.removed: list[str] = []

    async def _add(self, message: IncomingMessage, state: str) -> Any:
        self.added.append(state)
        return state

    async def _remove(self, token: Any) -> None:
        self.removed.append(str(token))


class _SilentLog:
    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None


def _message(ts: str = "1700000000.000100") -> IncomingMessage:
    return IncomingMessage(
        sender_id="U1",
        channel_id="C1",
        content="hi",
        metadata={"ts": ts},
    )


@pytest.mark.asyncio
async def test_failed_clears_the_progress_marks_and_keeps_the_failure_mark() -> None:
    reactor = _RecordingReactor()
    message = _message()

    await reactor.received(message)
    await reactor.running(message)
    await reactor.failed(message)

    assert reactor.added == ["received", "running", "failed"]
    # The ✅/👀 pair is cleared; ❌ stays as the outcome marker.
    assert reactor.removed == ["received", "running"]


@pytest.mark.asyncio
async def test_failed_does_not_leave_a_tracked_entry_behind() -> None:
    reactor = _RecordingReactor()
    message = _message()

    await reactor.received(message)
    await reactor.failed(message)

    assert not reactor._active


@pytest.mark.asyncio
async def test_repeated_rejections_do_not_accumulate_entries() -> None:
    reactor = _RecordingReactor()

    for index in range(5):
        message = _message(ts=f"1700000000.00{index:04d}")
        await reactor.received(message)
        await reactor.failed(message)

    assert not reactor._active


@pytest.mark.asyncio
async def test_completed_after_failed_does_not_remove_the_failure_mark() -> None:
    reactor = _RecordingReactor()
    message = _message()

    await reactor.received(message)
    await reactor.failed(message)
    await reactor.completed(message)

    assert reactor.removed == ["received"]


@pytest.mark.asyncio
async def test_completed_still_clears_every_tracked_mark() -> None:
    reactor = _RecordingReactor()
    message = _message()

    await reactor.received(message)
    await reactor.running(message)
    await reactor.completed(message)

    assert reactor.removed == ["received", "running"]
    assert not reactor._active


@pytest.mark.asyncio
async def test_failed_is_skipped_once_reactions_are_disabled() -> None:
    reactor = _RecordingReactor()
    message = _message()
    await reactor.received(message)
    reactor._disable("test")

    await reactor.failed(message)

    # The tracked mark is still reclaimed, but no new reaction is attempted.
    assert reactor.removed == ["received"]
    assert reactor.added == ["received"]
