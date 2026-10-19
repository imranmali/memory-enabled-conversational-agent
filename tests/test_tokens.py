from memory_agent.tokens import MESSAGE_OVERHEAD, estimate_tokens, message_tokens, total_tokens
from memory_agent.types import Message


def test_empty_and_short():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1


def test_roughly_four_chars_per_token():
    assert estimate_tokens("x" * 400) == 100


def test_message_overhead_and_total():
    msgs = [Message(role="user", content="x" * 8), Message(role="assistant", content="y" * 4)]
    assert message_tokens(msgs[0]) == 2 + MESSAGE_OVERHEAD
    assert total_tokens(msgs) == 2 + 1 + 2 * MESSAGE_OVERHEAD


def test_custom_counter():
    msgs = [Message(role="user", content="one two three")]
    assert total_tokens(msgs, counter=lambda t: len(t.split())) == 3 + MESSAGE_OVERHEAD
