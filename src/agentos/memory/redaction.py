"""Shared redaction helpers for memory-derived text."""

from __future__ import annotations

import re

from agentos.redact import redact_sensitive_text

# A leading ``qualifier_``/``qualifier-`` chain is part of the name: ``_`` is a
# word character, so ``\b`` alone never fires before "token" in ``reset_token``
# and the value went into durable memory unmasked. The keyword must still sit
# immediately before the separator, which is what keeps ordinary field names
# such as ``token_count`` or ``my_token_count`` out — the keyword there is
# followed by ``_count``, not by ``:``/``=`` — and camelCase humps are still not
# split, so ``sellToken`` stays an asset name rather than a credential.
#
# The chain is bounded rather than ``*``: every ``-`` is a word boundary, so
# an unbounded chain retries a growing prefix at each of the n segment starts
# in a run like ``"8f3a-" * 20000`` — quadratic, and measured at 22s for one
# 100KB line on a path that runs per transcript message. Four qualifiers is
# well past anything real (``password_reset_token`` uses two).
_KEYWORD_PATTERN = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-]){0,4}(?:api[_-]?key|secret|token|password))"
    r"(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
)


def _replace_keyword(match: re.Match[str]) -> str:
    val = match.group(3)
    stripped_val = val.strip("\"'")
    if "***" in stripped_val or "«redacted" in stripped_val or "[REDACTED]" in stripped_val:
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}[REDACTED]"


def redact_memory_text(text: str) -> str:
    # force=True: AGENTOS_REDACT_SECRETS=0 is an *egress* escape hatch and must
    # not unmask what gets written to durable memory.
    redacted = redact_sensitive_text(text, force=True) or text
    return _KEYWORD_PATTERN.sub(_replace_keyword, redacted)
