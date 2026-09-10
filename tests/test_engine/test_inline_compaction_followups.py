"""Inline compaction must preserve durable messages outside the running turn."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from agentos.engine.agent import Agent
from agentos.engine.runtime import TurnRunner
from agentos.engine.turn_runner.harness import _TurnRunnerCompactionPersistAdapter
from agentos.engine.turn_runner.stream_consumer_stage import _CompactionHandler
from agentos.engine.types import CompactionEvent
from agentos.session.manager import SessionManager
from agentos.session.models import TranscriptEntry
from agentos.session.storage import SessionStorage

KEY = "agent:main:compaction-test"


@pytest_asyncio.fixture
async def context():
    storage = SessionStorage(":memory:")
    await storage.connect()
    manager = SessionManager(storage, inject_time_prefix=False)
    await manager.create(KEY)
    runner = TurnRunner(provider_selector=MagicMock(), session_manager=manager)
    agent = Agent(provider=MagicMock())
    try:
        yield manager, runner, agent
    finally:
        await storage.close()


async def _load_history(manager, runner, agent):
    for role, content in [
        ("user", "original request"),
        ("assistant", "answer"),
        ("user", "current request"),
    ]:
        await manager.append_message(KEY, role, content)
    await runner._load_history(agent, KEY, trim_last_user=False)
    return await manager.get_transcript(KEY)


async def _compact(runner, agent, kept, compaction_id="inline-test"):
    handler = _CompactionHandler(
        persist=_TurnRunnerCompactionPersistAdapter(runner),
        memory_snapshot=MagicMock(),
        system_prompt=MagicMock(),
    )
    await handler.handle(
        CompactionEvent(
            summary="Earlier conversation", kept_entries=kept, compaction_id=compaction_id
        ),
        SimpleNamespace(
            agent=agent,
            session_key=KEY,
            agent_id="main",
            session_manager_present=True,
            private_memory_allowed=False,
            tool_defs=[],
            bootstrap_context_mode="full",
        ),
    )


@pytest.mark.asyncio
async def test_inline_compaction_preserves_queued_followups_and_metadata(context):
    manager, runner, agent = context
    original = await _load_history(manager, runner, agent)
    # Include identical text so content matching cannot identify the snapshot.
    queued = [
        await manager.append_message(KEY, "user", "current request", token_count=7),
        await manager.append_message(
            KEY,
            "user",
            '{"text":"follow-up","attachments":[{"name":"sample.txt"}]}',
            provenance={"kind": "test"},
            token_count=11,
        ),
    ]
    await _compact(
        runner,
        agent,
        [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "current request"},
        ],
    )

    live = await manager.get_transcript(KEY)
    canonical = await manager.get_canonical_transcript(KEY)
    assert [entry.message_id for entry in live] == [
        entry.message_id for entry in [*original[1:], *queued]
    ]
    assert [entry.message_id for entry in canonical] == [
        entry.message_id for entry in [*original, *queued]
    ]
    for expected, actual in zip(queued, live[-2:], strict=True):
        assert actual.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    summaries = await manager.get_summaries(KEY)
    assert len(summaries) == 1
    assert summaries[0].removed_count == 1
    assert summaries[0].kept_count == 2
    assert summaries[0].covered_through_id == original[0].id


@pytest.mark.asyncio
async def test_snapshot_validation_uses_transcript_order_instead_of_insert_order(context):
    manager, runner, agent = context
    node = await manager.storage.get_session(KEY)
    original = []
    for role, content, created_at in [
        ("user", "current request", 3000),
        ("user", "original request", 1000),
        ("assistant", "answer", 2000),
    ]:
        entry = TranscriptEntry(
            session_id=node.session_id,
            session_key=KEY,
            role=role,
            content=content,
            created_at=created_at,
        )
        await manager.storage.append_transcript_entry(entry)
        original.append(entry)
    await runner._load_history(agent, KEY, trim_last_user=False)
    queued = await manager.append_message(KEY, "user", "follow-up")
    await _compact(
        runner,
        agent,
        [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "current request"},
        ],
    )

    assert [entry.message_id for entry in await manager.get_transcript(KEY)] == [
        original[2].message_id,
        original[0].message_id,
        queued.message_id,
    ]
    canonical = await manager.get_canonical_transcript(KEY)
    assert [entry.message_id for entry in canonical] == [
        entry.message_id for entry in [original[1], original[2], original[0], queued]
    ]


@pytest.mark.asyncio
async def test_repeated_compaction_never_adopts_pending_followups(context):
    manager, runner, agent = context
    original = await _load_history(manager, runner, agent)
    first = await manager.append_message(KEY, "user", "first follow-up")
    await _compact(
        runner,
        agent,
        [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "current request"},
        ],
        "inline-first",
    )
    second = await manager.append_message(KEY, "user", "second follow-up")
    await _compact(
        runner,
        agent,
        [
            {"role": "user", "content": "current request"},
        ],
        "inline-second",
    )

    live = await manager.get_transcript(KEY)
    assert [entry.message_id for entry in live] == [
        original[-1].message_id,
        first.message_id,
        second.message_id,
    ]
    canonical = await manager.get_canonical_transcript(KEY)
    assert [entry.message_id for entry in canonical] == [
        entry.message_id for entry in [*original, first, second]
    ]
    assert agent.compaction_source_message_ids == [original[-1].message_id]
    assert [summary.removed_count for summary in await manager.get_summaries(KEY)] == [1, 1]


@pytest.mark.asyncio
async def test_empty_kept_tail_preserves_pending_followup(context):
    manager, runner, agent = context
    original = await _load_history(manager, runner, agent)
    queued = await manager.append_message(KEY, "user", "follow-up")
    await _compact(runner, agent, [])

    assert [entry.message_id for entry in await manager.get_transcript(KEY)] == [queued.message_id]
    canonical = await manager.get_canonical_transcript(KEY)
    assert [entry.message_id for entry in canonical] == [
        entry.message_id for entry in [*original, queued]
    ]
    assert agent.compaction_source_message_ids == []


@pytest.mark.parametrize("change", ["remove", "reset"])
@pytest.mark.asyncio
async def test_stale_inline_snapshot_cannot_rewrite_changed_session(context, change):
    manager, runner, agent = context
    original = await _load_history(manager, runner, agent)
    if change == "reset":
        await manager.delete(KEY)
        await manager.create(KEY)
        await manager.append_message(KEY, "user", "new conversation")
    else:
        await manager.remove_message(KEY, original[0].message_id)
    before = await manager.get_transcript(KEY)
    await _compact(runner, agent, [{"role": "user", "content": "current request"}])

    assert await manager.get_transcript(KEY) == before
    assert await manager.get_summaries(KEY) == []


@pytest.mark.asyncio
async def test_write_after_snapshot_read_is_rejected_atomically(context, monkeypatch):
    manager, runner, agent = context
    original = await _load_history(manager, runner, agent)
    storage = manager.storage
    rewrite = storage.rewrite_compacted_session
    appended = []

    async def racing_rewrite(**kwargs):
        appended.append(await manager.append_message(KEY, "user", "late writer"))
        await rewrite(**kwargs)

    monkeypatch.setattr(storage, "rewrite_compacted_session", racing_rewrite)
    await _compact(runner, agent, [{"role": "user", "content": "current request"}])

    assert [entry.message_id for entry in await manager.get_transcript(KEY)] == [
        entry.message_id for entry in [*original, *appended]
    ]
    assert await manager.get_summaries(KEY) == []
    assert await manager.get_context_states(KEY) == []
    assert not storage.conn.in_transaction


@pytest.mark.asyncio
async def test_missing_inline_snapshot_never_falls_back_to_latest_row_count(context):
    manager, runner, agent = context
    await manager.append_message(KEY, "user", "message outside loaded history")
    before = await manager.get_transcript(KEY)
    await _compact(runner, agent, [])
    assert await manager.get_transcript(KEY) == before
    assert await manager.get_summaries(KEY) == []
