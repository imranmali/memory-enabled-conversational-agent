"""Lightweight, dependency-free token estimation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from .types import Message

TokenCounter = Callable[[str], int]

# Per-message overhead (role markers, separators) applied by most chat formats.
MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters per token, minimum 1 for non-empty text)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def message_tokens(message: Message, counter: TokenCounter = estimate_tokens) -> int:
    return counter(message.content) + MESSAGE_OVERHEAD


def total_tokens(messages: Iterable[Message], counter: TokenCounter = estimate_tokens) -> int:
    return sum(message_tokens(m, counter) for m in messages)
