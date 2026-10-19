from __future__ import annotations

import json

import pytest

from memory_agent.short_term import ShortTermBuffer
from memory_agent.tokens import MESSAGE_OVERHEAD, message_tokens
from memory_agent.types import Message


def msg(content: str, i: int = 0, role: str = "user") -> Message:
    return Message(role=role, content=content, timestamp=1000.0 + i, id=f"m{i}")


def numbered(n: int, start: int = 1) -> list[Message]:
    return [msg(f"message {i}", i) for i in range(start, start + n)]


# -- construction ---------------------------------------------------------------------------


def test_defaults():
    buf = ShortTermBuffer()
    assert buf.max_messages == 20
    assert buf.max_tokens == 2000
    assert len(buf) == 0
    assert buf.token_count() == 0


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", True])
@pytest.mark.parametrize("field", ["max_messages", "max_tokens"])
def test_invalid_limits_rejected(field, bad):
    with pytest.raises(ValueError):
        ShortTermBuffer(**{field: bad})


def test_unbounded_limits_never_evict():
    calls = []
    buf = ShortTermBuffer(max_messages=None, max_tokens=None, on_evict=calls.append)
    evicted = buf.extend(msg("x" * 400, i) for i in range(500))
    assert evicted == []
    assert len(buf) == 500
    assert calls == []


# -- message-count eviction -----------------------------------------------------------------


def test_oldest_message_evicted_on_count_limit():
    buf = ShortTermBuffer(max_messages=3, max_tokens=None)
    m1, m2, m3, m4 = numbered(4)
    for m in (m1, m2, m3):
        assert buf.add(m) == []
    assert buf.add(m4) == [m1]
    assert buf.messages() == [m2, m3, m4]


def test_nothing_evicted_under_limits():
    buf = ShortTermBuffer(max_messages=5, max_tokens=1000)
    assert buf.add(msg("hello")) == []
    assert len(buf) == 1


# -- token-budget eviction ------------------------------------------------------------------


def test_token_accounting_uses_message_tokens():
    buf = ShortTermBuffer(max_messages=None, max_tokens=None)
    ms = [msg("a" * 10, 1), msg("b" * 33, 2)]
    buf.extend(ms)
    assert buf.token_count() == sum(message_tokens(m) for m in ms)
    assert buf.token_count() == 3 + 9 + 2 * MESSAGE_OVERHEAD


def test_token_budget_triggers_fifo_eviction():
    # each message: 40 chars -> 10 tokens + 4 overhead = 14 tokens
    buf = ShortTermBuffer(max_messages=None, max_tokens=30)
    m1, m2, m3 = (msg("x" * 40, i) for i in range(1, 4))
    assert buf.add(m1) == []
    assert buf.add(m2) == []
    assert buf.token_count() == 28
    assert buf.add(m3) == [m1]
    assert buf.messages() == [m2, m3]
    assert buf.token_count() == sum(message_tokens(m) for m in buf.messages()) == 28


def test_token_budget_evicts_multiple_oldest():
    buf = ShortTermBuffer(max_messages=None, max_tokens=50)
    small = [msg("y" * 4, i) for i in range(1, 6)]  # 5 tokens each -> 25
    buf.extend(small)
    big = msg("z" * 120, 9)  # 30 + 4 = 34 tokens
    evicted = buf.add(big)
    assert evicted == small[:2]  # 25 + 34 = 59 -> drop two (49)
    assert buf.token_count() == 49
    assert buf.messages()[-1] is big


def test_oversized_newest_message_retained():
    calls = []
    buf = ShortTermBuffer(max_messages=10, max_tokens=20, on_evict=calls.append)
    m1, m2 = msg("a", 1), msg("b", 2)
    buf.extend([m1, m2])
    huge = msg("h" * 1000, 3)
    evicted = buf.add(huge)
    assert evicted == [m1, m2]
    assert buf.messages() == [huge]
    assert buf.token_count() == message_tokens(huge) > 20
    assert calls == [[m1, m2]]


def test_oversized_message_into_empty_buffer():
    buf = ShortTermBuffer(max_tokens=5)
    huge = msg("h" * 100, 1)
    assert buf.add(huge) == []
    assert buf.messages() == [huge]


def test_custom_token_counter():
    def per_word(text: str) -> int:
        return len(text.split())

    buf = ShortTermBuffer(max_messages=None, max_tokens=15, token_counter=per_word)
    m1 = msg("one two three", 1)  # 3 + 4 = 7
    m2 = msg("four five six", 2)  # 7 -> 14
    m3 = msg("seven", 3)  # 5 -> 19 > 15 -> evict m1
    buf.extend([m1, m2])
    assert buf.token_count() == 14
    assert buf.add(m3) == [m1]
    assert buf.token_count() == 12


def test_both_limits_apply():
    buf = ShortTermBuffer(max_messages=2, max_tokens=1000)
    ms = numbered(3)
    assert buf.extend(ms) == ms[:1]
    assert buf.messages() == ms[1:]


# -- eviction callback ----------------------------------------------------------------------


def test_callback_receives_batch_once_in_order():
    calls = []
    buf = ShortTermBuffer(max_messages=None, max_tokens=30, on_evict=calls.append)
    m1, m2 = msg("a" * 8, 1), msg("b" * 8, 2)  # 6 tokens each
    buf.extend([m1, m2])
    big = msg("c" * 100, 3)  # 29 tokens -> 41 -> evict m1 (35) and m2 (29)
    evicted = buf.add(big)
    assert evicted == [m1, m2]
    assert calls == [[m1, m2]]


def test_callback_not_called_without_eviction():
    calls = []
    buf = ShortTermBuffer(max_messages=3, on_evict=calls.append)
    buf.add(msg("hi", 1))
    buf.extend(numbered(2, start=2))
    assert calls == []


def test_callback_once_per_extend():
    calls = []
    buf = ShortTermBuffer(max_messages=3, max_tokens=None, on_evict=calls.append)
    ms = numbered(5)
    evicted = buf.extend(ms)
    assert evicted == ms[:2]
    assert calls == [ms[:2]]
    assert buf.messages() == ms[2:]


def test_callback_gets_independent_list():
    calls = []
    buf = ShortTermBuffer(max_messages=1, on_evict=calls.append)
    m1, m2 = numbered(2)
    buf.add(m1)
    evicted = buf.add(m2)
    calls[0].clear()
    assert evicted == [m1]


def test_clear_returns_removed_without_callback():
    calls = []
    buf = ShortTermBuffer(max_messages=5, on_evict=calls.append)
    ms = numbered(3)
    buf.extend(ms)
    assert buf.clear() == ms
    assert len(buf) == 0
    assert buf.token_count() == 0
    assert calls == []
    assert buf.clear() == []


# -- inspection ------------------------------------------------------------------------------


def test_messages_returns_copy():
    buf = ShortTermBuffer()
    ms = numbered(2)
    buf.extend(ms)
    got = buf.messages()
    got.append(msg("intruder", 99))
    got.pop(0)
    assert buf.messages() == ms


def test_last_n():
    buf = ShortTermBuffer()
    m1, m2, m3 = numbered(3)
    buf.extend([m1, m2, m3])
    assert buf.last(2) == [m2, m3]
    assert buf.last(1) == [m3]
    assert buf.last(10) == [m1, m2, m3]
    assert buf.last(0) == []
    assert buf.last(-3) == []


def test_len_and_iter():
    buf = ShortTermBuffer()
    ms = numbered(3)
    buf.extend(ms)
    assert len(buf) == 3
    assert list(buf) == ms
    assert [m.id for m in buf] == ["m1", "m2", "m3"]


def test_iteration_safe_during_mutation():
    buf = ShortTermBuffer(max_messages=10)
    buf.extend(numbered(3))
    seen = []
    for m in buf:
        seen.append(m)
        if len(seen) == 1:
            buf.add(msg("late", 50))
    assert len(seen) == 3


def test_repr_mentions_limits():
    assert "max_messages=4" in repr(ShortTermBuffer(max_messages=4))


# -- persistence -----------------------------------------------------------------------------


def test_round_trip_preserves_state():
    buf = ShortTermBuffer(max_messages=7, max_tokens=500)
    ms = [
        Message(role="user", content="hi there", timestamp=1.0, id="a", metadata={"k": [1, 2]}),
        Message(role="assistant", content="hello!", timestamp=2.0, id="b"),
        Message(role="system", content="note", timestamp=3.0, id="c"),
    ]
    buf.extend(ms)
    data = buf.to_dict()
    restored = ShortTermBuffer.from_dict(json.loads(json.dumps(data)))
    assert restored.max_messages == 7
    assert restored.max_tokens == 500
    assert restored.messages() == ms
    assert restored.token_count() == buf.token_count()


def test_round_trip_with_unbounded_limits():
    buf = ShortTermBuffer(max_messages=None, max_tokens=None)
    buf.extend(numbered(3))
    restored = ShortTermBuffer.from_dict(buf.to_dict())
    assert restored.max_messages is None
    assert restored.max_tokens is None
    assert len(restored) == 3


def test_from_dict_does_not_alias_input():
    buf = ShortTermBuffer()
    buf.add(Message(role="user", content="x", id="a", metadata={"tags": ["t"]}))
    data = buf.to_dict()
    restored = ShortTermBuffer.from_dict(data)
    data["messages"][0]["metadata"]["tags"].append("mutated")
    assert restored.messages()[0].metadata == {"tags": ["t"]}


def test_from_dict_with_callback_and_overrides():
    calls = []
    buf = ShortTermBuffer(max_messages=3, max_tokens=None)
    ms = numbered(3)
    buf.extend(ms)
    restored = ShortTermBuffer.from_dict(buf.to_dict(), on_evict=calls.append)
    assert calls == []
    m4 = msg("new", 4)
    restored.add(m4)
    assert calls == [[ms[0]]]

    shrunk = ShortTermBuffer.from_dict(buf.to_dict(), max_messages=1)
    assert shrunk.max_messages == 1
    assert shrunk.messages() == [ms[2]]


def test_to_dict_is_snapshot():
    buf = ShortTermBuffer()
    buf.add(msg("one", 1))
    data = buf.to_dict()
    buf.add(msg("two", 2))
    assert len(data["messages"]) == 1
