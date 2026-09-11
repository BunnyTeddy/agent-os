"""Slack replies must never inherit another conversation's thread.

Regression coverage for #1543: ``SlackChannel`` kept a single
``_last_thread_ts`` for the whole account, overwritten by every inbound event
carrying a ``thread_ts``, and ``send`` fell back to it whenever an outgoing
message had no anchor of its own. The thread a message landed in was therefore
decided by whichever conversation spoke most recently -- so a scheduled
delivery, heartbeat or other proactive send could be posted into an unrelated
conversation's thread.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from agentos.channels.slack import SlackChannel
from agentos.channels.types import IncomingMessage, OutgoingMessage

_REQUEST = httpx.Request("POST", "https://slack.test/api")


def _ok_response() -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "ts": "1.0"}, request=_REQUEST)


def _channel(**kwargs: Any) -> SlackChannel:
    channel = SlackChannel(token="xoxb-test", slack_channel_id="C123", **kwargs)
    channel.bot_user_id = "UBOT"
    return channel


def _attach(channel: SlackChannel) -> AsyncMock:
    client = AsyncMock()
    client.post = AsyncMock(return_value=_ok_response())
    channel._client = client
    return client.post


def _inbound_threaded_event(channel: str, thread_ts: str) -> dict[str, Any]:
    return {
        "user": "U1",
        "channel": channel,
        "text": "hello",
        "ts": thread_ts,
        "thread_ts": thread_ts,
    }


@pytest.mark.asyncio
async def test_send_does_not_inherit_a_thread_from_another_conversation() -> None:
    channel = _channel(reply_in_thread=True)
    post = _attach(channel)
    # Conversation A speaks in a thread...
    channel.parse_event(_inbound_threaded_event("C_AAA", "1700000000.000100"))

    # ...then an unrelated proactive message goes out to conversation B.
    await channel.send(OutgoingMessage(content="scheduled report", reply_to="C_BBB"))

    payload = post.await_args.kwargs["json"]
    assert payload["channel"] == "C_BBB"
    assert "thread_ts" not in payload


@pytest.mark.asyncio
async def test_send_without_an_anchor_posts_unthreaded_in_the_same_conversation() -> None:
    channel = _channel(reply_in_thread=True)
    post = _attach(channel)
    channel.parse_event(_inbound_threaded_event("C_AAA", "1700000000.000100"))

    # Even back into the same channel, an anchorless send belongs in the
    # channel: the adapter cannot know which of its threads was meant.
    await channel.send(OutgoingMessage(content="heartbeat", reply_to="C_AAA"))

    payload = post.await_args.kwargs["json"]
    assert payload["channel"] == "C_AAA"
    assert "thread_ts" not in payload


@pytest.mark.asyncio
async def test_send_still_honours_an_explicit_thread_anchor() -> None:
    channel = _channel(reply_in_thread=True)
    post = _attach(channel)

    await channel.send(
        OutgoingMessage(
            content="in-thread reply",
            reply_to="C_AAA",
            metadata={"channel": "C_AAA", "thread_ts": "1700000000.000100"},
        )
    )

    payload = post.await_args.kwargs["json"]
    assert payload["thread_ts"] == "1700000000.000100"


@pytest.mark.asyncio
async def test_reply_built_for_an_inbound_message_still_threads_to_it() -> None:
    channel = _channel(reply_in_thread=True)
    post = _attach(channel)
    # A different conversation speaks last, so a shared anchor would win here.
    channel.parse_event(_inbound_threaded_event("C_BBB", "1700000000.000999"))
    inbound = IncomingMessage(
        sender_id="U1",
        channel_id="C_AAA",
        content="question",
        metadata={"ts": "1700000000.000200", "thread_ts": "1700000000.000100"},
    )

    await channel.send(channel.build_reply_message("answer", inbound))

    payload = post.await_args.kwargs["json"]
    assert payload["channel"] == "C_AAA"
    assert payload["thread_ts"] == "1700000000.000100"


@pytest.mark.asyncio
async def test_reply_to_a_bare_thread_ts_is_still_supported() -> None:
    channel = _channel(reply_in_thread=True)
    post = _attach(channel)

    await channel.send(
        OutgoingMessage(content="threaded", reply_to="1700000000.000100"),
    )

    payload = post.await_args.kwargs["json"]
    assert payload["channel"] == "C123"
    assert payload["thread_ts"] == "1700000000.000100"
