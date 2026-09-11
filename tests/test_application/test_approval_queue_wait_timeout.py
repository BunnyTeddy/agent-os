from __future__ import annotations

import time

import pytest

from agentos.application.approval_queue import ApprovalQueue


@pytest.mark.asyncio
async def test_wait_per_call_timeout_leaves_approval_pending_when_lifetime_not_expired(
    tmp_path,
) -> None:
    db_path = tmp_path / "approval_queue.sqlite"
    queue = ApprovalQueue(default_timeout=300.0, db_path=str(db_path), poll_interval=0.01)
    approval_id = queue.request("exec", {"toolName": "exec_command"})

    result = await queue.wait(approval_id, timeout=0.05)

    assert result is False
    entry = queue.get(approval_id)
    assert entry.resolved is False
    assert entry.approved is False

    # A human operator resolving after the caller's bounded wait returned
    # must still succeed -- it must not have been permanently denied.
    queue.resolve(approval_id, True)
    resolved_entry = queue.get(approval_id)
    assert resolved_entry.resolved is True
    assert resolved_entry.approved is True
    queue.close()


@pytest.mark.asyncio
async def test_wait_denies_once_the_overall_approval_lifetime_has_expired(tmp_path) -> None:
    db_path = tmp_path / "approval_queue.sqlite"
    queue = ApprovalQueue(default_timeout=300.0, db_path=str(db_path), poll_interval=0.01)
    approval_id = queue.request("exec", {"toolName": "exec_command"})

    # Simulate the approval having been created long ago, so its overall
    # lifespan (created_at + default_timeout) has already elapsed.
    queue._conn.execute(
        "UPDATE approval_queue SET created_at = ? WHERE approval_id = ?",
        (time.time() - 1000.0, approval_id),
    )
    queue._conn.commit()

    result = await queue.wait(approval_id, timeout=0.05)

    assert result is False
    entry = queue.get(approval_id)
    assert entry.resolved is True
    assert entry.approved is False

    with pytest.raises(ValueError, match="already resolved"):
        queue.resolve(approval_id, True)
    queue.close()


@pytest.mark.asyncio
async def test_wait_returns_immediately_once_resolved_during_the_call(tmp_path) -> None:
    db_path = tmp_path / "approval_queue.sqlite"
    queue = ApprovalQueue(default_timeout=300.0, db_path=str(db_path), poll_interval=0.01)
    approval_id = queue.request("exec", {"toolName": "exec_command"})
    queue.resolve(approval_id, True)

    result = await queue.wait(approval_id, timeout=5.0)

    assert result is True
    queue.close()
