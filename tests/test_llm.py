import builtins
import sys
from types import SimpleNamespace

import pytest

from memory_agent.interfaces import LLM
from memory_agent.llm import (
    FACTS_HEADER,
    MEMORIES_HEADER,
    PREVIOUS_SESSION_HEADER,
    REFUSAL_MESSAGE,
    SUMMARY_HEADER,
    AnthropicLLM,
    OfflineLLM,
    build_llm,
    parse_sections,
    to_second_person,
)
from memory_agent.types import Message


def prompt(facts=(), memories=(), previous=None, summary=None):
    parts = ["You are helpful.\n## Some other header\n- ignored"]
    if facts:
        parts.append(FACTS_HEADER + "\n" + "\n".join(f"- {f}" for f in facts))
    if previous:
        parts.append(PREVIOUS_SESSION_HEADER + "\n" + previous)
    if memories:
        parts.append(MEMORIES_HEADER + "\n" + "\n".join(f"- {m}" for m in memories))
    if summary:
        parts.append(SUMMARY_HEADER + "\n" + summary)
    return "\n\n".join(parts)


def user(text):
    return [Message(role="user", content=text)]


# --------------------------------------------------------------------------- sections


def test_parse_sections():
    sections = parse_sections(
        prompt(
            facts=["User's name is Ada", "User loves hiking"],
            memories=["(fact) User lives in Paris"],
            previous="Topics: hiking\nUser: hello",
            summary="User: talked about cats",
        )
    )
    assert sections[FACTS_HEADER] == ["User's name is Ada", "User loves hiking"]
    assert sections[MEMORIES_HEADER] == ["(fact) User lives in Paris"]
    assert sections[PREVIOUS_SESSION_HEADER] == ["Topics: hiking", "User: hello"]
    assert sections[SUMMARY_HEADER] == ["User: talked about cats"]
    assert parse_sections("") == {k: [] for k in sections}


@pytest.mark.parametrize(
    ("fact", "expected"),
    [
        ("User's name is Ada", "your name is Ada"),
        ("User is allergic to cats", "you are allergic to cats"),
        ("User was born in Rome", "you were born in Rome"),
        ("User has two cats", "you have two cats"),
        ("User does yoga", "you do yoga"),
        ("User loves hiking", "you love hiking"),
        ("User really enjoys their coffee", "you really enjoy your coffee"),
        ("User studies physics", "you study physics"),
        ("User watches films", "you watch films"),
        ("User prefers to be called Al", "you prefer to be called Al"),
        ("The meeting is at 5", "The meeting is at 5"),
    ],
)
def test_to_second_person(fact, expected):
    assert to_second_person(fact) == expected


# --------------------------------------------------------------------------- offline model


def test_offline_satisfies_protocol_and_is_deterministic():
    llm = OfflineLLM()
    assert isinstance(llm, LLM)
    system = prompt(facts=["User loves hiking"])
    assert llm.complete(system, user("hmm")) == llm.complete(system, user("hmm"))


def test_what_do_you_remember_lists_facts_and_previous_session():
    llm = OfflineLLM()
    reply = llm.complete(
        prompt(facts=["User's name is Ada", "User loves hiking"], previous="Topics: hiking, maps"),
        user("What do you remember about me?"),
    )
    assert "your name is Ada and you love hiking" in reply
    assert "Last time we talked about hiking and maps" in reply


def test_what_do_you_remember_without_facts():
    reply = OfflineLLM().complete(prompt(), user("what do you know about me"))
    assert "don't know much about you" in reply


def test_name_known_unknown_and_preferred():
    llm = OfflineLLM()
    assert "Ada" in llm.complete(prompt(facts=["User's name is Ada"]), user("What's my name?"))
    unknown = llm.complete(prompt(), user("What's my name?"))
    assert "don't know your name" in unknown
    preferred = llm.complete(
        prompt(facts=["User's name is Robert", "User prefers to be called Bob"]),
        user("do you know my name"),
    )
    assert preferred == "Your name is Bob."


def test_acknowledges_new_facts():
    reply = OfflineLLM().complete(prompt(), user("My name is Ada and I love hiking."))
    assert reply.startswith("Nice to meet you, Ada!")
    assert "I'll remember that you love hiking" in reply
    only_fact = OfflineLLM().complete(prompt(), user("I live in Paris"))
    assert only_fact == "Got it — I'll remember that you live in Paris."


def test_greetings_with_and_without_previous_session():
    llm = OfflineLLM()
    fresh = llm.complete(prompt(), [])
    assert fresh.startswith("Hello!")
    back = llm.complete(prompt(facts=["User's name is Ada"], previous="Topics: hiking"), [])
    assert back.startswith("Welcome back, Ada!") and "hiking" in back
    summary_only = llm.complete(prompt(previous="User: I love hiking."), user("hello"))
    assert "Last time: User: I love hiking." in summary_only
    assert llm.complete(prompt(facts=["User's name is Ada"]), user("hi")).startswith("Hello, Ada")


def test_references_top_relevant_memory():
    llm = OfflineLLM()
    fact = llm.complete(prompt(memories=["(fact) User lives in Paris"]), user("Any travel tips?"))
    assert "you live in Paris" in fact
    message = llm.complete(
        prompt(memories=["(message) Any travel tips?", "(message) I went to Paris in May"]),
        user("Any travel tips?"),
    )
    assert 'earlier you mentioned "I went to Paris in May"' in message
    summary = llm.complete(prompt(memories=["(summary) we planned a Paris trip"]), user("so?"))
    assert "Paris trip" in summary
    untagged = llm.complete(prompt(memories=["Paris notes"]), user("so?"))
    assert "Paris notes" in untagged


def test_generic_reply():
    reply = OfflineLLM().complete(prompt(facts=["User's name is Ada"]), user("The sky is grey."))
    assert "Ada" in reply and "Tell me more" in reply
    assert "Thanks for sharing." in OfflineLLM().complete(prompt(), user("The sky is grey."))


# --------------------------------------------------------------------------- Anthropic model


class FakeCreate:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    beta_create = FakeCreate(response)
    create = FakeCreate(response)
    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=beta_create)),
        messages=SimpleNamespace(create=create),
    )
    return client, beta_create, create


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def test_anthropic_default_request_kwargs():
    client, beta_create, create = fake_client(response(text_block("Hi Ada!")))
    llm = AnthropicLLM(client=client)
    messages = [
        Message(role="system", content="ignored"),
        Message(role="user", content="Hello"),
    ]
    assert llm.complete("SYSTEM", messages) == "Hi Ada!"
    assert create.calls == []
    assert beta_create.calls == [
        {
            "model": "claude-opus-5-5",
            "max_tokens": 16000,
            "system": "SYSTEM",
            "messages": [{"role": "user", "content": "Hello"}],
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "output_config": {"effort": "medium"},
        }
    ]
    assert "thinking" not in beta_create.calls[0]
    assert "temperature" not in beta_create.calls[0]


def test_anthropic_without_fallbacks_or_effort():
    client, beta_create, create = fake_client(response(text_block("ok")))
    llm = AnthropicLLM(
        model="claude-sonnet-5-5", max_tokens=100, effort=None, fallbacks=False, client=client
    )
    llm.complete("S", [Message(role="user", content="Hi")])
    assert beta_create.calls == []
    assert create.calls == [
        {
            "model": "claude-sonnet-5-5",
            "max_tokens": 100,
            "system": "S",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    ]


def test_anthropic_effort_without_fallbacks():
    client, _, create = fake_client(response(text_block("ok")))
    AnthropicLLM(effort="low", fallbacks=False, client=client).complete("S", [])
    assert create.calls[0]["output_config"] == {"effort": "low"}
    assert "betas" not in create.calls[0]


def test_message_normalisation():
    to_api = AnthropicLLM.to_api_messages
    assert to_api([]) == [{"role": "user", "content": "Hello."}]
    assert to_api(
        [
            Message(role="assistant", content="Welcome!"),
            Message(role="user", content="a"),
            Message(role="user", content="b"),
            Message(role="system", content="x"),
            Message(role="assistant", content="   "),
            Message(role="assistant", content="c"),
        ]
    ) == [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Welcome!"},
        {"role": "user", "content": "a\n\nb"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "Please continue."},
    ]


def test_anthropic_joins_only_text_blocks():
    client, _, _ = fake_client(
        response(
            SimpleNamespace(type="thinking", thinking=""),
            text_block("Hello "),
            SimpleNamespace(type="fallback"),
            text_block("there."),
        )
    )
    assert AnthropicLLM(client=client).complete("S", []) == "Hello there."


def test_anthropic_refusal_checked_before_content():
    class Exploding:
        stop_reason = "refusal"

        @property
        def content(self):  # pragma: no cover - must not be read
            raise AssertionError("content read on refusal")

    client, _, _ = fake_client(Exploding())
    assert AnthropicLLM(client=client).complete("S", []) == REFUSAL_MESSAGE


def test_anthropic_invalid_max_tokens():
    with pytest.raises(ValueError):
        AnthropicLLM(max_tokens=0, client=object())


def test_anthropic_missing_sdk(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"memory-enabled-conversational-agent\[anthropic\]"):
        AnthropicLLM()


def test_anthropic_lazy_sdk_client(monkeypatch):
    sentinel = object()
    module = SimpleNamespace(Anthropic=lambda: sentinel)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    assert AnthropicLLM().client is sentinel


# --------------------------------------------------------------------------- factory


def test_build_llm(monkeypatch):
    assert isinstance(build_llm("offline"), OfflineLLM)
    assert isinstance(build_llm(" Offline "), OfflineLLM)
    llm = build_llm("anthropic", client=object(), model="m")
    assert isinstance(llm, AnthropicLLM) and llm.model == "m"
    with pytest.raises(ValueError, match="offline, anthropic"):
        build_llm("gpt")
