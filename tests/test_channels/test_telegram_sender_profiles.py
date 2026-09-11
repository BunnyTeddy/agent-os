"""Regression tests for the removed Telegram sender-profile map (#1542).

``_known_sender_profiles`` was written on every inbound update (``enqueue``)
and for unpaired-DM pairing requests (``record_access_denial``) but had no
reader anywhere in ``src/`` or ``tests/``. The fix removes the field and its
write sites entirely instead of bounding dead state. These tests pin the
removal: the write-only state must never come back, ``enqueue`` keeps
queueing inbound messages, and pairing requests keep carrying the sender
profile.
"""

from __future__ import annotations

from agentos.channel_pairing import ChannelPairingStore
from agentos.channels.telegram import TelegramChannel, TelegramChannelConfig
from agentos.channels.types import IncomingMessage


def _channel() -> TelegramChannel:
    return TelegramChannel(TelegramChannelConfig(token="token"))


def _incoming(sender_id: str) -> IncomingMessage:
    return IncomingMessage(
        sender_id=sender_id,
        channel_id="-100999",
        content="hi",
        metadata={"message_id": sender_id},
    )


def test_inbound_updates_leave_no_sender_profile_state() -> None:
    channel = _channel()

    for sender_id in range(1, 201):
        channel.enqueue(_incoming(str(sender_id)))

    # The write-only profile map is gone and must not be resurrected (#1542).
    assert not hasattr(channel, "_known_sender_profiles")
    assert not hasattr(channel, "_remember_sender")
    assert not hasattr(channel, "_store_sender_profile")


def test_enqueue_still_queues_inbound_messages() -> None:
    channel = _channel()

    channel.enqueue(_incoming("42"))

    assert channel._queue.qsize() == 1


def test_record_access_denial_keeps_pairing_without_profile_side_state(tmp_path) -> None:
    channel = TelegramChannel(
        TelegramChannelConfig(name="tg"),
        pairing_store=ChannelPairingStore(tmp_path / "pairing"),
    )
    message = IncomingMessage(sender_id="42", channel_id="42", content="hi", metadata={})

    channel.record_access_denial(message, "not_paired")

    assert not hasattr(channel, "_known_sender_profiles")
    pending = channel.access_snapshot()["pending"]
    assert len(pending) == 1
    assert pending[0]["sender_id"] == "42"
    assert pending[0]["username"] == ""
