"""Short-term conversational memory: a bounded FIFO buffer of recent messages."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .tokens import TokenCounter, estimate_tokens, message_tokens
from .types import Message

EvictCallback = Callable[[list[Message]], None]


def _check_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    return value


class ShortTermBuffer:
    """Recent messages kept within a message-count and token budget.

    When a limit is exceeded the oldest messages are evicted first. The newest message is
    always retained, even if it alone exceeds ``max_tokens``. Evicted messages are returned
    from :meth:`add` / :meth:`extend` and passed (once per call, oldest first) to ``on_evict``
    so they can be summarized or archived to long-term memory.
    """

    def __init__(
        self,
        max_messages: int | None = 20,
        max_tokens: int | None = 2000,
        token_counter: TokenCounter = estimate_tokens,
        on_evict: EvictCallback | None = None,
    ) -> None:
        self.max_messages = _check_limit("max_messages", max_messages)
        self.max_tokens = _check_limit("max_tokens", max_tokens)
        self.token_counter = token_counter
        self.on_evict = on_evict
        self._messages: deque[Message] = deque()
        self._costs: deque[int] = deque()
        self._total = 0

    # -- mutation -----------------------------------------------------------------------------

    def _append(self, message: Message) -> list[Message]:
        cost = message_tokens(message, self.token_counter)
        self._messages.append(message)
        self._costs.append(cost)
        self._total += cost
        evicted: list[Message] = []
        while self._over_limit():
            evicted.append(self._messages.popleft())
            self._total -= self._costs.popleft()
        return evicted

    def _over_limit(self) -> bool:
        if self.max_messages is not None and len(self._messages) > self.max_messages:
            return True
        return (
            self.max_tokens is not None
            and self._total > self.max_tokens
            and len(self._messages) > 1
        )

    def _notify(self, evicted: list[Message]) -> None:
        if evicted and self.on_evict is not None:
            self.on_evict(list(evicted))

    def add(self, message: Message) -> list[Message]:
        """Append ``message``; return the messages evicted by this call, oldest first."""
        evicted = self._append(message)
        self._notify(evicted)
        return evicted

    def extend(self, messages: Iterable[Message]) -> list[Message]:
        """Append ``messages`` in order; return everything evicted by this call, oldest first."""
        evicted: list[Message] = []
        for message in messages:
            evicted.extend(self._append(message))
        self._notify(evicted)
        return evicted

    def clear(self) -> list[Message]:
        """Remove and return all messages. Does not invoke ``on_evict``."""
        removed = list(self._messages)
        self._messages.clear()
        self._costs.clear()
        self._total = 0
        return removed

    # -- inspection ---------------------------------------------------------------------------

    def messages(self) -> list[Message]:
        """All buffered messages, oldest first (a new list)."""
        return list(self._messages)

    def last(self, n: int) -> list[Message]:
        """The most recent ``n`` messages, oldest first; empty for ``n <= 0``."""
        if n <= 0:
            return []
        return list(self._messages)[-n:]

    def token_count(self) -> int:
        """Total tokens of buffered messages (content estimate plus per-message overhead)."""
        return self._total

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(list(self._messages))

    def __repr__(self) -> str:
        return (
            f"ShortTermBuffer(messages={len(self)}, tokens={self._total}, "
            f"max_messages={self.max_messages}, max_tokens={self.max_tokens})"
        )

    # -- persistence --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict snapshot of limits and messages (JSON-serializable if metadata is)."""
        return {
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
            "messages": [m.to_dict() for m in self._messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> ShortTermBuffer:
        """Rebuild a buffer from :meth:`to_dict` output.

        ``kwargs`` override stored limits and supply ``token_counter`` / ``on_evict``. Messages
        are re-added in order, so limits are re-enforced if overridden smaller.
        """
        params: dict[str, Any] = {
            "max_messages": data.get("max_messages"),
            "max_tokens": data.get("max_tokens"),
        }
        params.update(kwargs)
        buffer = cls(**params)
        buffer.extend(Message.from_dict(copy.deepcopy(dict(m))) for m in data.get("messages", []))
        return buffer
