"""Unit tests for SessionDeltaTracker, in particular snapshot()/consume().

``consume()`` replaces a bare ``reset()`` at the one call site in
``MemorySyncManager.sync()`` specifically so that delta recorded
concurrently with an in-flight sync -- via ``notify_message()`` from the
poll loop or a request handler, both running as independent asyncio tasks
with no lock between them -- is not silently discarded. These tests cover
the tracker in isolation; ``test_memory_sync_manager_search_delta.py``
covers the same guarantee at the ``MemorySyncManager.sync()`` level.
"""

from __future__ import annotations

from agentos.memory.sync_manager import SessionDeltaTracker


def test_snapshot_reflects_pending_state_at_the_time_it_was_taken() -> None:
    tracker = SessionDeltaTracker()
    tracker.record(byte_count=10, message_count=1)
    snapshot = tracker.snapshot()
    assert snapshot == (10, 1)

    # Recording after the snapshot must not retroactively change it.
    tracker.record(byte_count=5, message_count=1)
    assert snapshot == (10, 1)


def test_consume_clears_exactly_the_snapshotted_amount() -> None:
    tracker = SessionDeltaTracker()
    tracker.record(byte_count=10, message_count=1)
    snapshot = tracker.snapshot()

    tracker.consume(snapshot)

    assert tracker.has_pending() is False


def test_consume_preserves_activity_recorded_after_the_snapshot() -> None:
    """The exact scenario consume() exists for: new delta arrives mid-sync."""
    tracker = SessionDeltaTracker()
    tracker.record(byte_count=10_000, message_count=1)
    snapshot = tracker.snapshot()

    # New messages land while the sync that took `snapshot` is still running.
    tracker.record(byte_count=90_000, message_count=2)

    tracker.consume(snapshot)

    assert tracker._pending_bytes == 90_000
    assert tracker._pending_messages == 2
    assert tracker.has_pending() is True


def test_consume_floors_at_zero_and_never_goes_negative() -> None:
    """A snapshot larger than current state (e.g. after an external reset())
    must not push the counters negative."""
    tracker = SessionDeltaTracker()
    tracker.record(byte_count=100, message_count=5)
    snapshot = tracker.snapshot()

    tracker.reset()

    tracker.consume(snapshot)

    assert tracker._pending_bytes == 0
    assert tracker._pending_messages == 0
    assert tracker.has_pending() is False


def test_reset_still_clears_everything_unconditionally() -> None:
    tracker = SessionDeltaTracker()
    tracker.record(byte_count=100, message_count=5)
    tracker.reset()
    assert tracker.has_pending() is False
