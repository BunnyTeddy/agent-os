"""Regression tests ensuring scheduler timer and heartbeat nudges are not swallowed."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentos.scheduler.heartbeat_loop import HeartbeatLoop
from agentos.scheduler.persistence import JobStore
from agentos.scheduler.timer import SchedulerTimer
from agentos.scheduler.types import CronJob, JobStatus, ScheduleKind, SessionTarget


def _immediate_job(job_id: str) -> CronJob:
    now = datetime.now(UTC)
    return CronJob(
        id=job_id,
        name=job_id,
        cron_expr=now.isoformat(),
        schedule_raw=now.isoformat(),
        handler_key="agent_run",
        payload={"kind": "agent_turn", "task": "noop", "agent_id": "main"},
        session_target=SessionTarget.ISOLATED,
        schedule_kind=ScheduleKind.AT,
        next_run_at=now,
        status=JobStatus.PENDING,
    )


@pytest.mark.asyncio
async def test_scheduler_timer_nudge_during_tick_wakes_immediately() -> None:
    """A nudge received while the tick is executing must not be cleared and dropped."""
    tick_count = 0
    first_tick_done = asyncio.Event()
    second_tick_done = asyncio.Event()

    async with JobStore(":memory:") as store:
        timer = SchedulerTimer(store, handlers={})

        original_tick = timer._tick

        async def custom_tick() -> None:
            nonlocal tick_count
            tick_count += 1
            if tick_count == 1:
                # Nudge while tick is executing (simulates concurrent job addition)
                timer.nudge()
                first_tick_done.set()
            elif tick_count == 2:
                second_tick_done.set()
            await original_tick()

        timer._tick = custom_tick  # type: ignore[method-assign]
        timer.MAX_TIMER_DELAY = 10.0  # Large delay to ensure test fails if nudge is swallowed

        await timer.start()
        try:
            # Wake the first tick
            timer.nudge()
            await asyncio.wait_for(first_tick_done.wait(), timeout=2.0)

            # The second tick must wake up immediately from the nudge fired during tick 1
            # without waiting for MAX_TIMER_DELAY (10s)
            start_wait = time.monotonic()
            await asyncio.wait_for(second_tick_done.wait(), timeout=2.0)
            elapsed = time.monotonic() - start_wait
            assert elapsed < 2.0
            assert tick_count >= 2
        finally:
            await timer.stop()


@pytest.mark.asyncio
async def test_scheduler_timer_preexisting_nudge_skips_wait() -> None:
    """If _nudge_event was already set before the wait check, it executes immediately."""
    async with JobStore(":memory:") as store:
        timer = SchedulerTimer(store, handlers={})
        timer.MAX_TIMER_DELAY = 10.0
        ticked = asyncio.Event()

        async def mark_tick() -> None:
            ticked.set()

        timer._tick = mark_tick  # type: ignore[method-assign]

        # Nudge before starting
        timer.nudge()
        assert timer._nudge_event.is_set()

        start = time.monotonic()
        await timer.start()
        try:
            await asyncio.wait_for(ticked.wait(), timeout=2.0)
            assert time.monotonic() - start < 2.0
            # Nudge event should now be cleared after consumption
            assert not timer._nudge_event.is_set()
        finally:
            await timer.stop()


@pytest.mark.asyncio
async def test_heartbeat_loop_nudge_during_tick_wakes_immediately() -> None:
    """A nudge to HeartbeatLoop during tick execution must not be lost."""
    tick_count = 0
    first_tick = asyncio.Event()
    second_tick = asyncio.Event()

    config = MagicMock()
    config.heartbeat.enabled = True
    config.heartbeat.interval_ms = 60000  # 60s interval
    config.heartbeat.prompt = "ping"
    config.heartbeat.config_path = None
    config.workspace_dir = None
    config.workspace_strict = False

    service = AsyncMock()
    service.run_once.return_value = MagicMock(status="ok")

    loop = HeartbeatLoop(config=config, heartbeat_service=service)

    async def custom_tick() -> None:
        nonlocal tick_count
        tick_count += 1
        if tick_count == 1:
            loop.nudge()
            first_tick.set()
        elif tick_count == 2:
            second_tick.set()

    loop._tick = custom_tick  # type: ignore[method-assign]

    await loop.start()
    try:
        # First tick starts after loop is nudged
        loop.nudge()
        await asyncio.wait_for(first_tick.wait(), timeout=2.0)

        # Second tick should happen immediately due to nudge, not after 60 seconds
        start_wait = time.monotonic()
        await asyncio.wait_for(second_tick.wait(), timeout=2.0)
        assert tick_count >= 2
        assert time.monotonic() - start_wait < 2.0
    finally:
        await loop.stop()
