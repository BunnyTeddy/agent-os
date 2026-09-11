"""Tests for memory redaction helpers."""

from __future__ import annotations

import pytest

from agentos.memory.redaction import redact_memory_text


@pytest.mark.parametrize(
    ("input_text", "expected_text"),
    [
        # Existing OpenAI / OpenRouter patterns
        ("my key is " + "sk-or-v1-" + "abcdefghijklmnopqrstuvwxyz", "my key is sk-or-***wxyz"),
        ("my key is " + "sk-" + "abcdefghijklmnopqrstuvwxyz", "my key is sk-abc***wxyz"),
        ("api_key=plain-secret", "api_key=***"),
        ("api-key: plain-secret", "api-key: ***"),
        ('api_key="plain-secret"', 'api_key="***"'),
        ("password = plain-secret", "password = ***"),
        ('secret: "my secret"', 'secret: [REDACTED]'),
        # AWS access key IDs
        ("AWS ID: " + "AKIA" + "IOSFODNN7EXAMPLE", "AWS ID: AKIAIO***MPLE"),
        ("AWS ID: " + "ASIA" + "IOSFODNN7EXAMPLE", "AWS ID: ASIAIO***MPLE"),
        # AWS temporary key shapes
        ("AWS ID: " + "ABIA" + "IOSFODNN7EXAMPLE", "AWS ID: ABIAIO***MPLE"),
        ("AWS ID: " + "ACCA" + "IOSFODNN7EXAMPLE", "AWS ID: ACCAIO***MPLE"),
        # GitHub tokens
        (
            "Check token " + "ghp" + "_123456789012345678901234567890123456 in logs",
            "Check token ghp_12***3456 in logs",
        ),
        (
            "fine-grained " + "github_pat" + "_"
            "11a1111111a1111111a1111111a1111111a1111111a"
            "111111a1111111a1111111a1111111a1111111a token",
            "fine-grained github***111a token",
        ),
        (
            "Google key: " + "AIza" + "SyD-example_key_35_characters_long1",
            "Google key: AIzaSy***ong1",
        ),
        # Slack tokens
        (
            "Slack token: " + "xoxb-" + "mock-bot-token-for-testing",
            "Slack token: xoxb-m***ting",
        ),
        (
            "Slack user: " + "xoxp-" + "mock-user-token-for-testing",
            "Slack user: xoxp-m***ting",
        ),
        # JWTs
        (
            "Bearer " + "eyJ" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "Bearer eyJhbG***sw5c",
        ),
        # Authorization Headers
        (
            "Headers: Authorization: Bearer some_token_value",
            "Headers: Authorization: Bearer ***",
        ),
        (
            '{"Authorization": "Bearer some_token_value", "other": 1}',
            '{"Authorization": "Bearer ***", "other": 1}',
        ),
        (
            '{"Authorization": "Bearer ' + "eyJ" + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            'eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature", "other": 1}',
            '{"Authorization": "Bearer ***", "other": 1}',
        ),
        # Field names that shouldn't be redacted
        ("sellToken: 123", "sellToken: 123"),
        ("token_count = 5", "token_count = 5"),
        ("my_token_count = 10", "my_token_count = 10"),
        # PEM Private Keys
        (
            "Before\n"
            "-----" + "BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y...\n"
            "-----" + "END RSA PRIVATE KEY-----\n"
            "After",
            "Before\n«redacted:private-key»\nAfter",
        ),
        (
            "Before\n"
            "-----" + "BEGIN PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y...\n"
            "-----" + "END PRIVATE KEY-----\n"
            "After",
            "Before\n«redacted:private-key»\nAfter",
        ),
    ],
)
def test_redact_memory_text(input_text: str, expected_text: str) -> None:
    assert redact_memory_text(input_text) == expected_text


@pytest.mark.parametrize(
    "input_text",
    [
        "reset_token: 8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
        "csrf_token=8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
        "device_token: 8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
        "verification_token = 8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
        "password_reset_token: 8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
        "app_secret: 8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
        "db_password = hunter2hunter2",
        "stripe_api_key: 8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b",
    ],
)
def test_snake_case_qualified_secret_names_are_redacted(input_text: str) -> None:
    # "_" is a word character, so \b never fires before "token" when a
    # snake_case qualifier is glued to it — the value used to pass through
    # unmasked on the way into durable memory.
    name, _, _value = input_text.partition("=" if "=" in input_text else ":")
    redacted = redact_memory_text(input_text)
    assert "8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b" not in redacted
    assert "hunter2hunter2" not in redacted
    assert redacted.startswith(name.rstrip())


@pytest.mark.parametrize(
    "input_text",
    [
        "sellToken: 123",
        "token_count = 5",
        "my_token_count = 10",
        "reset_token_count = 5",
        "secret_santa_name: bob",
        "the password reset flow is broken",
    ],
)
def test_snake_case_widening_does_not_redact_ordinary_field_names(input_text: str) -> None:
    assert redact_memory_text(input_text) == input_text


def test_hyphen_runs_do_not_backtrack_quadratically() -> None:
    """redact_memory_text runs per transcript message; it must stay linear.

    Every "-" is a word boundary, so an unbounded qualifier chain retried a
    growing prefix at each of the n segment starts. One 100KB line took 22s.
    """
    import time

    text = "8f3a-" * 20000

    start = time.perf_counter()
    redact_memory_text(text)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"redaction took {elapsed:.2f}s on a 100KB hyphen run"


@pytest.mark.parametrize(
    "name",
    ["reset_token", "app_reset_token", "mobile_app_reset_token", "my_mobile_app_reset_token"],
)
def test_qualifier_chains_up_to_the_bound_still_redact(name: str) -> None:
    secret = "8f3a91c2b6d04e7f9a1b5c3d7e2f4a6b"

    assert secret not in redact_memory_text(f"{name}: {secret}")
