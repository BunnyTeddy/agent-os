"""An unrelated ops edit must not revert a reservation that landed mid-flight.

Regression coverage for #1537: ``ops.update``/``pause``/``resume`` each do
``get(job_id)`` -> mutate a field -> ``save(job)``, and the upsert wrote the
reservation columns unconditionally. An atomic ``reserve_due_job`` landing in
the gap was therefore reverted by a caller that only meant to flip ``status``,
which silently discarded the run's result (``apply_reserved_result`` no longer
recognises the token) and left the row free for the next tick to run the same
job again, concurrently with the run still in flight.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentos.scheduler.ops import SchedulerOps
from agentos.scheduler.persistence import JobStore
from agentos.scheduler.types import (
    CronJob,
    JobExecution,
    JobReservation,
    JobStatus,
    ScheduleKind,
    SessionTarget,
)

_JOB_ID = "job-reservation-race"


def _cron_job(next_run_at: datetime) -> CronJob:
    return CronJob(
        id=_JOB_ID,
        name=_JOB_ID,
        cron_expr="* * * * *",
        handler_key="agent_run",
        payload={"kind": "agent_turn", "task": "work", "agent_id": "main"},
        session_target=SessionTarget.ISOLATED,
        schedule_kind=ScheduleKind.CRON,
        next_run_at=next_run_at,
        status=JobStatus.PENDING,
    )


async def _reserve(store: JobStore) -> JobReservation:
    reservation = await store.reserve_due_job(
        _JOB_ID,
        datetime.now(UTC),
        source="timer",
        owner="scheduler-timer",
    )
    assert isinstance(reservation, JobReservation)
    return reservation


class _ReserveMidEdit:
    """Land an atomic reservation inside an ops read-modify-write.

    ``ops`` reads the job, mutates a field, then saves it back. Reserving at
    the top of that ``save`` reproduces the exact window: the row is reserved,
    but the job object about to be written still carries the pre-reservation
    snapshot of the reservation columns.
    """

    def __init__(self, store: JobStore) -> None:
        self._store = store
        self._original = store.save
        self.reservation: JobReservation | None = None
        store.save = self  # type: ignore[method-assign]

    async def __call__(self, job: CronJob, **kwargs: object) -> None:
        if self.reservation is None:
            self.reservation = await _reserve(self._store)
        await self._original(job, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["pause", "resume", "update"])
async def test_ops_edit_does_not_revert_a_concurrent_reservation(operation: str) -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job(datetime.now(UTC) - timedelta(seconds=1)))
        ops = SchedulerOps(store)
        racer = _ReserveMidEdit(store)

        if operation == "update":
            await ops.update(_JOB_ID, name="renamed")
        else:
            await getattr(ops, operation)(_JOB_ID)

        assert racer.reservation is not None
        current = await store.get(_JOB_ID)
        assert current is not None
        assert current.reservation_token == racer.reservation.token
        assert current.reserved_by == "scheduler-timer"


@pytest.mark.asyncio
async def test_result_of_the_in_flight_run_still_applies_after_a_concurrent_pause() -> None:
    """The end-to-end consequence: the run's result must not be discarded."""
    from agentos.scheduler.jobs import apply_reserved_result

    async with JobStore(":memory:") as store:
        await store.save(_cron_job(datetime.now(UTC) - timedelta(seconds=1)))
        ops = SchedulerOps(store)
        racer = _ReserveMidEdit(store)

        await ops.pause(_JOB_ID)

        assert racer.reservation is not None
        execution = JobExecution(job_id=_JOB_ID, started_at=datetime.now(UTC), success=True)
        applied = await apply_reserved_result(_JOB_ID, racer.reservation.token, execution, store)

        assert applied is True


@pytest.mark.asyncio
async def test_ops_edit_still_writes_the_fields_it_owns() -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job(datetime.now(UTC) - timedelta(seconds=1)))
        ops = SchedulerOps(store)
        _ReserveMidEdit(store)

        await ops.update(_JOB_ID, name="renamed")
        await ops.pause(_JOB_ID)

        current = await store.get(_JOB_ID)
        assert current is not None
        assert current.name == "renamed"
        assert current.status is JobStatus.PAUSED


@pytest.mark.asyncio
async def test_reservation_protocol_can_still_clear_the_reservation() -> None:
    async with JobStore(":memory:") as store:
        await store.save(_cron_job(datetime.now(UTC) - timedelta(seconds=1)))
        reservation = await _reserve(store)

        released = await store.release_reservation(_JOB_ID, reservation.token)

        assert released is True
        current = await store.get(_JOB_ID)
        assert current is not None
        assert current.reservation_token == ""
        assert current.reserved_by == ""


@pytest.mark.asyncio
async def test_creating_a_job_still_persists_its_columns() -> None:
    async with JobStore(":memory:") as store:
        ops = SchedulerOps(store)

        created = await ops.add(
            "fresh",
            schedule_kind=ScheduleKind.CRON,
            schedule_value="* * * * *",
            payload={"kind": "agent_turn", "task": "work", "agent_id": "main"},
        )

        stored = await store.get(created.id)
        assert stored is not None
        assert stored.name == "fresh"
        assert stored.reservation_token == ""
