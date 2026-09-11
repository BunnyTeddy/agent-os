"""Issue #1538: the final drain loop in _drain_task_runtime_for_session must
check every active task, not stop at the first one that times out.

The earlier "settle" loop in the same function already nests its
try/except TimeoutError inside the for, so one slow task doesn't stop the
loop from checking the rest. The final drain loop used to wrap the whole
for in one outer try/except TimeoutError instead, so the first task that
exceeded the drain timeout aborted the loop -- every remaining active task
was never waited on, and reset/delete proceeded to touch session storage
while those tasks could still be running against it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.gateway import rpc_sessions
from agentos.gateway.rpc_sessions import _drain_task_runtime_for_session

_SESSION_KEY = "agent:main:drain-loop-test"


class _Row:
    def __init__(self, task_id: str, status: str) -> None:
        self.task_id = task_id
        self.status = status


class _FakeTaskRuntime:
    """list()/wait()/cancel() double. wait() sleeps past the drain timeout
    for task ids named in ``slow``, so asyncio.wait_for genuinely times out
    rather than a mocked exception standing in for one."""

    def __init__(self, rows: list[_Row], slow: set[str]) -> None:
        self._rows = rows
        self._slow = slow
        self.waited: list[str] = []

    async def list(self, session_key: str) -> list[_Row]:
        return self._rows

    async def wait(self, task_id: str) -> None:
        self.waited.append(task_id)
        if task_id in self._slow:
            await asyncio.sleep(10)

    async def cancel(self, session_key: str, source: str = "", reason: str = "") -> int:
        return 0


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real asyncio.wait_for timeouts, just short ones, so tests stay fast.
    monkeypatch.setattr(rpc_sessions, "_RESET_RUNTIME_SETTLE_SECONDS", 0.02)
    monkeypatch.setattr(rpc_sessions, "_RESET_RUNTIME_CANCEL_DRAIN_SECONDS", 0.05)


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        rpc_sessions.log, "warning", lambda event, **kw: calls.append((event, kw))
    )
    return calls


async def test_a_slow_first_task_does_not_stop_the_rest_from_being_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"queued" status skips the earlier settle loop (which only waits on
    "running" tasks) so these waits come from the final drain loop alone."""
    _capture_warnings(monkeypatch)
    rows = [_Row("t1", "queued"), _Row("t2", "queued"), _Row("t3", "queued")]
    runtime = _FakeTaskRuntime(rows, slow={"t1"})

    await _drain_task_runtime_for_session(
        runtime, _SESSION_KEY, source="sessions_reset", reason="session_reset", op="reset"
    )

    # t1 times out, but t2 and t3 must still have been waited on.
    assert runtime.waited == ["t1", "t2", "t3"]


async def test_drain_timeout_warning_carries_the_undrained_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_warnings(monkeypatch)
    rows = [_Row("t1", "queued"), _Row("t2", "queued"), _Row("t3", "queued")]
    runtime = _FakeTaskRuntime(rows, slow={"t1", "t3"})

    await _drain_task_runtime_for_session(
        runtime, _SESSION_KEY, source="sessions_reset", reason="session_reset", op="reset"
    )

    drain_timeout_calls = [c for c in calls if c[0] == "sessions.reset.task_runtime_drain_timeout"]
    assert len(drain_timeout_calls) == 1
    _, kwargs = drain_timeout_calls[0]
    assert kwargs["session_key"] == _SESSION_KEY
    assert kwargs["undrained_count"] == 2


async def test_no_warning_when_every_active_task_drains_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_warnings(monkeypatch)
    rows = [_Row("t1", "queued"), _Row("t2", "queued")]
    runtime = _FakeTaskRuntime(rows, slow=set())

    await _drain_task_runtime_for_session(
        runtime, _SESSION_KEY, source="sessions_reset", reason="session_reset", op="reset"
    )

    assert runtime.waited == ["t1", "t2"]
    assert not any(c[0] == "sessions.reset.task_runtime_drain_timeout" for c in calls)


async def test_inactive_and_terminal_tasks_are_never_waited_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A "running" task is waited on twice by design — once by the earlier
    settle loop, once by the final drain loop; only completed/failed rows
    must never be waited on at all."""
    _capture_warnings(monkeypatch)
    rows = [_Row("done", "completed"), _Row("failed", "failed"), _Row("live", "running")]
    runtime = _FakeTaskRuntime(rows, slow=set())

    await _drain_task_runtime_for_session(
        runtime, _SESSION_KEY, source="sessions_reset", reason="session_reset", op="reset"
    )

    assert runtime.waited == ["live", "live"]


async def test_list_failure_still_logs_drain_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_warnings(monkeypatch)

    class _BoomRuntime(_FakeTaskRuntime):
        async def list(self, session_key: str) -> list[_Row]:
            raise RuntimeError("boom")

    runtime = _BoomRuntime([], slow=set())

    await _drain_task_runtime_for_session(
        runtime, _SESSION_KEY, source="sessions_reset", reason="session_reset", op="reset"
    )

    assert any(c[0] == "sessions.reset.task_runtime_drain_failed" for c in calls)


async def test_no_listing_support_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_runtime without list()/wait() (has_runtime_listing=False) must
    not be touched by the drain loop at all."""
    calls = _capture_warnings(monkeypatch)

    async def cancel(session_key: str, source: str = "", reason: str = "") -> int:
        return 0

    runtime = SimpleNamespace(cancel=cancel)

    await _drain_task_runtime_for_session(
        runtime, _SESSION_KEY, source="sessions_reset", reason="session_reset", op="reset"
    )

    assert calls == []
