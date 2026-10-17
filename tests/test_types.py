from memory_agent.types import MemoryItem, Message, now


def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.id and abs(m.timestamp - now()) < 5


def test_message_ids_unique():
    assert Message(role="user", content="a").id != Message(role="user", content="a").id


def test_message_round_trip():
    m = Message(role="assistant", content="hello", metadata={"k": 1})
    assert Message.from_dict(m.to_dict()) == m


def test_memory_item_timestamp_defaults():
    item = MemoryItem(user_id="u", content="c", embedding=[1.0], created_at=100.0)
    assert item.last_accessed == 100.0
    assert item.updated_at == 100.0
    assert item.deleted is False and item.access_count == 0


def test_memory_item_round_trip():
    item = MemoryItem(
        user_id="u", content="c", embedding=[0.1, 0.2], kind="fact", metadata={"a": 1}
    )
    assert MemoryItem.from_dict(item.to_dict()) == item


def test_importance_clamped():
    assert MemoryItem(user_id="u", content="c", embedding=[], importance=5).importance == 1.0
    assert MemoryItem(user_id="u", content="c", embedding=[], importance=-1).importance == 0.0


def test_touch_records_access_without_update():
    item = MemoryItem(user_id="u", content="c", embedding=[], created_at=10.0)
    item.touch(at=50.0)
    assert item.access_count == 1
    assert item.last_accessed == 50.0
    assert item.updated_at == 10.0
