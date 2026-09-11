"""Manual "run now" must execute the reserved row, not a pre-reservation snapshot.

Regression coverage for #1555: ``run_job_now`` read the job, reserved it, then
ran ``execute_with_timeout(job, handler)`` against the row it read *before* the
reservation. ``reservation.job`` -- the freshly-read row the reservation itself
returned, and what ``timer._run_single`` already uses -- was ignored, so an
``update()`` landing between the two reads ran the payload, prompt, timeout or
handler the operator had just replaced.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agentos.scheduler.engine import SchedulerEngine
from agentos.scheduler.persistence import JobStore
from agentos.scheduler.types import (
    CronJob,
    JobStatus,
    ManualRunStatus,
    ScheduleKind,
    SessionTarget,
)

_JOB_ID = "job-run-now"


def _cron_job(handler_key: str = "agent_run", payload: dict | None = None) -> CronJob:
    return CronJob(
        id=_JOB_ID,
        name=_JOB_ID,
        cron_expr="* * * * *",
        handler_key=handler_key,
        payload=payload or {"kind": "agent_turn", "task": "original", "agent_id": "main"},
        session_target=SessionTarget.ISOLATED,
        schedule_kind=ScheduleKind.CRON,
        next_run_at=datetime.now(UTC) + timedelta(minutes=5),
        status=JobStatus.PENDING,
    )


def _update_during_reservation(store: JobStore, **changes: Any) -> None:
    """Land an ``update()`` in the window run_job_now leaves open.

    ``run_job_now`` reads the job, then reserves it. Mutating inside the
    reservation call reproduces exactly that window: the outer snapshot is
    stale, while ``reservation.job`` carries the new row.
    """
    original = store.reserve_manual_job

    async def reserve_after_update(job_id: str, *args: Any, **kwargs: Any):
        current = await store.get(job_id)
        assert current is not None
        for field, value in changes.items():
            setattr(current, field, value)
        await store.save(current)
        return await original(job_id, *args, **kwargs)

    store.reserve_manual_job = reserve_after_update  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_run_now_executes_the_payload_from_the_reserved_row() -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job())
        engine = SchedulerEngine(store)
        seen: list[dict] = []

        async def handler(job: CronJob) -> str:
            seen.append(dict(job.payload))
            return "ok"

        engine._timer._handlers["agent_run"] = handler
        _update_during_reservation(
            store,
            payload={"kind": "agent_turn", "task": "updated", "agent_id": "main"},
        )

        result = await engine.run_job_now(_JOB_ID)

        assert result.status is ManualRunStatus.ACCEPTED
        assert seen == [{"kind": "agent_turn", "task": "updated", "agent_id": "main"}]


@pytest.mark.asyncio
async def test_run_now_dispatches_to_the_handler_key_from_the_reserved_row() -> None:
    # ``handler_key`` is derived from the payload kind, so switching a job from
    # an agent turn to a reminder moves it from ``agent_run`` to
    # ``static_message`` -- an ordinary edit that repoints the handler.
    async with JobStore(":memory:") as store:
        await store.save(_cron_job())
        engine = SchedulerEngine(store)
        called: list[str] = []

        async def agent_handler(job: CronJob) -> str:
            called.append("agent_run")
            return "ok"

        async def reminder_handler(job: CronJob) -> str:
            called.append("static_message")
            return "ok"

        engine._timer._handlers["agent_run"] = agent_handler
        engine._timer._handlers["static_message"] = reminder_handler
        _update_during_reservation(
            store,
            handler_key="static_message",
            payload={"kind": "reminder", "text": "updated reminder", "agent_id": "main"},
        )

        result = await engine.run_job_now(_JOB_ID)

        assert result.status is ManualRunStatus.ACCEPTED
        assert called == ["static_message"]


@pytest.mark.asyncio
async def test_run_now_releases_the_reservation_when_the_updated_handler_is_missing() -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job())
        engine = SchedulerEngine(store)

        async def handler(job: CronJob) -> str:
            return "ok"

        # Only the agent-turn handler is registered, so an edit to a reminder
        # leaves the reserved row pointing at a handler that does not exist.
        engine._timer._handlers["agent_run"] = handler
        _update_during_reservation(
            store,
            handler_key="static_message",
            payload={"kind": "reminder", "text": "updated reminder", "agent_id": "main"},
        )

        result = await engine.run_job_now(_JOB_ID)

        assert result.status is ManualRunStatus.NO_HANDLER
        assert result.error is not None
        assert "static_message" in result.error
        # The reservation must not be left held on a job that will never run.
        stored = await store.get(_JOB_ID)
        assert stored is not None
        assert not stored.reservation_token


@pytest.mark.asyncio
async def test_run_now_still_runs_normally_without_a_concurrent_update() -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job())
        engine = SchedulerEngine(store)
        seen: list[dict] = []

        async def handler(job: CronJob) -> str:
            seen.append(dict(job.payload))
            return "ok"

        engine._timer._handlers["agent_run"] = handler

        result = await engine.run_job_now(_JOB_ID)

        assert result.status is ManualRunStatus.ACCEPTED
        assert seen == [{"kind": "agent_turn", "task": "original", "agent_id": "main"}]


@pytest.mark.asyncio
async def test_run_now_reports_no_handler_without_reserving_when_none_is_registered() -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job())
        engine = SchedulerEngine(store)

        result = await engine.run_job_now(_JOB_ID)

        assert result.status is ManualRunStatus.NO_HANDLER
        stored = await store.get(_JOB_ID)
        assert stored is not None
        assert not stored.reservation_token
        assert stored.status is JobStatus.PENDING
