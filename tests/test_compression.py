from __future__ import annotations

from collections.abc import Sequence

import pytest

from memory_agent.compression import (
    ELLIPSIS,
    CompressedContext,
    ContextCompressor,
    ExtractiveSummarizer,
    LLMSummarizer,
    RollingSummary,
    prior_summary_message,
    render_transcript,
    truncate_to_tokens,
)
from memory_agent.interfaces import Summarizer
from memory_agent.tokens import MESSAGE_OVERHEAD, estimate_tokens, total_tokens
from memory_agent.types import Message


def msg(role: str, content: str, ts: float = 0.0) -> Message:
    return Message(role=role, content=content, timestamp=ts, id=f"{role}-{ts}-{len(content)}")


def word_counter(text: str) -> int:
    return len(text.split())


def conversation(n: int = 20) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(msg("user", f"Turn {i}: I am planning item number {i} for my garden.", i))
        else:
            out.append(msg("assistant", f"Turn {i}: That sounds like a lovely garden plan.", i))
    return out


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        self.calls.append((system, list(messages)))
        return self.reply


class BrokenLLM:
    def complete(self, system: str, messages: Sequence[Message]) -> str:
        raise RuntimeError("provider down")


class FixedSummarizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[list[Message], int]] = []

    def summarize(self, messages: Sequence[Message], max_tokens: int) -> str:
        self.calls.append((list(messages), max_tokens))
        return self.text


# --------------------------------------------------------------------------- truncation


class TestTruncate:
    def test_within_budget_unchanged(self) -> None:
        assert truncate_to_tokens("short text", 10) == "short text"

    def test_long_text_cut_at_word_boundary(self) -> None:
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 5
        out = truncate_to_tokens(text, 10)
        assert estimate_tokens(out) <= 10
        assert out.endswith(ELLIPSIS)
        body = out[: -len(ELLIPSIS)]
        assert text.startswith(body)
        # the next char in the source after the kept body is whitespace => no partial word
        assert text[len(body)].isspace()

    @pytest.mark.parametrize("budget", [1, 2, 3, 5, 8, 13, 21, 40])
    def test_never_exceeds_budget(self, budget: int) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 10
        assert estimate_tokens(truncate_to_tokens(text, budget)) <= budget

    def test_custom_counter(self) -> None:
        out = truncate_to_tokens("one two three four five six", 3, word_counter)
        assert word_counter(out) <= 3
        assert out.endswith(ELLIPSIS)
        assert out.startswith("one two")

    def test_single_long_word_is_cut_mid_word(self) -> None:
        out = truncate_to_tokens("x" * 100, 5)
        assert estimate_tokens(out) <= 5
        assert out.endswith(ELLIPSIS)

    @pytest.mark.parametrize("budget", [0, -3])
    def test_non_positive_budget(self, budget: int) -> None:
        assert truncate_to_tokens("anything at all", budget) == ""

    def test_empty_text(self) -> None:
        assert truncate_to_tokens("", 5) == ""

    def test_ellipsis_does_not_fit(self) -> None:
        def counter(text: str) -> int:
            return len(text) + (5 if ELLIPSIS in text else 0)

        out = truncate_to_tokens("abcdefgh", 3, counter)
        assert out == "abc"


# --------------------------------------------------------------------------- extractive


class TestExtractiveSummarizer:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(ExtractiveSummarizer(), Summarizer)

    def test_speaker_prefixed_chronological(self) -> None:
        msgs = [
            msg("user", "I adopted a cat named Miso last week.", 1),
            msg("assistant", "Cats make wonderful companions for apartments.", 2),
        ]
        out = ExtractiveSummarizer().summarize(msgs, 200)
        lines = out.splitlines()
        assert lines[0] == "User: I adopted a cat named Miso last week."
        assert lines[1] == "Assistant: Cats make wonderful companions for apartments."

    def test_order_follows_original_not_score(self) -> None:
        msgs = [
            msg("assistant", "The weather report mentions rain.", 1),
            msg("user", "My flight leaves at 7am on 12 March.", 2),
        ]
        out = ExtractiveSummarizer().summarize(msgs, 200)
        assert out.index("Assistant:") < out.index("User:")

    @pytest.mark.parametrize("budget", [1, 5, 12, 30, 60])
    def test_budget_respected(self, budget: int) -> None:
        msgs = conversation(30)
        out = ExtractiveSummarizer().summarize(msgs, budget)
        assert estimate_tokens(out) <= budget

    def test_budget_respected_custom_counter(self) -> None:
        out = ExtractiveSummarizer(word_counter).summarize(conversation(10), 15)
        assert 0 < word_counter(out) <= 15

    def test_empty_input(self) -> None:
        assert ExtractiveSummarizer().summarize([], 100) == ""

    def test_zero_budget(self) -> None:
        assert ExtractiveSummarizer().summarize(conversation(4), 0) == ""

    def test_system_messages_ignored(self) -> None:
        msgs = [
            msg("system", "SECRET SYSTEM INSTRUCTIONS here.", 0),
            msg("user", "I like hiking in the Alps.", 1),
        ]
        out = ExtractiveSummarizer().summarize(msgs, 200)
        assert "SECRET" not in out
        assert "Alps" in out
        assert ExtractiveSummarizer().summarize([msgs[0]], 200) == ""

    def test_deterministic(self) -> None:
        msgs = conversation(25)
        s = ExtractiveSummarizer()
        assert s.summarize(msgs, 40) == s.summarize(msgs, 40)
        assert s.summarize(msgs, 40) == ExtractiveSummarizer().summarize(list(msgs), 40)

    def test_self_disclosure_preferred_over_greeting(self) -> None:
        msgs = [
            msg("user", "Hi there!", 1),
            msg("assistant", "Hello! How can I help you today?", 2),
            msg("user", "My name is Alice.", 3),
        ]
        out = ExtractiveSummarizer().summarize(msgs, estimate_tokens("User: My name is Alice."))
        assert out == "User: My name is Alice."

    def test_digits_preferred_over_filler(self) -> None:
        msgs = [
            msg("user", "ok thanks", 1),
            msg("assistant", "The meeting is at 3pm on Friday.", 2),
            msg("user", "great, cool", 3),
        ]
        budget = estimate_tokens("Assistant: The meeting is at 3pm on Friday.")
        out = ExtractiveSummarizer().summarize(msgs, budget)
        assert "3pm" in out
        assert "thanks" not in out

    def test_scores_expose_boosts(self) -> None:
        msgs = [
            msg("user", "Hello there.", 1),
            msg("user", "I work as a nurse.", 2),
        ]
        scores = dict((line, score) for score, line in ExtractiveSummarizer().score_sentences(msgs))
        assert scores["User: I work as a nurse."] > scores["User: Hello there."]

    def test_sentence_too_long_is_truncated_as_last_resort(self) -> None:
        msgs = [msg("user", "I " + "really " * 50 + "love sailing.", 1)]
        out = ExtractiveSummarizer().summarize(msgs, 8)
        assert out.startswith("User: I really")
        assert out.endswith(ELLIPSIS)
        assert estimate_tokens(out) <= 8

    def test_prior_summary_speakers_preserved(self) -> None:
        prior = prior_summary_message("User: My dog is called Rex.\nAssistant: Noted.")
        out = ExtractiveSummarizer().summarize([prior, msg("user", "I live in Oslo.", 5)], 200)
        assert "User: My dog is called Rex." in out.splitlines()
        assert "User: I live in Oslo." in out.splitlines()
        assert "[Earlier summary]" not in out


# --------------------------------------------------------------------------- LLM


class TestLLMSummarizer:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(LLMSummarizer(FakeLLM("x")), Summarizer)

    def test_uses_llm_output_truncated(self) -> None:
        llm = FakeLLM("The user is Alice, a nurse from Oslo who loves sailing. " * 20)
        msgs = [msg("user", "My name is Alice.", 1), msg("assistant", "Nice to meet you.", 2)]
        out = LLMSummarizer(llm).summarize(msgs, 12)
        assert out.startswith("The user is Alice")
        assert estimate_tokens(out) <= 12
        assert len(llm.calls) == 1
        system, sent = llm.calls[0]
        assert system
        assert len(sent) == 1 and sent[0].role == "user"
        assert "User: My name is Alice." in sent[0].content
        assert "Assistant: Nice to meet you." in sent[0].content
        assert "12 tokens" in sent[0].content

    def test_short_output_returned_as_is(self) -> None:
        out = LLMSummarizer(FakeLLM("  Alice is a nurse.  ")).summarize([msg("user", "hi")], 50)
        assert out == "Alice is a nurse."

    def test_fallback_on_error_default_extractive(self) -> None:
        msgs = [msg("user", "I live in Oslo.", 1)]
        out = LLMSummarizer(BrokenLLM()).summarize(msgs, 50)
        assert out == ExtractiveSummarizer().summarize(msgs, 50)
        assert "Oslo" in out

    def test_fallback_on_error_custom(self) -> None:
        fallback = FixedSummarizer("fallback summary")
        out = LLMSummarizer(BrokenLLM(), fallback=fallback).summarize([msg("user", "x")], 50)
        assert out == "fallback summary"
        assert fallback.calls

    def test_fallback_on_empty_output(self) -> None:
        fallback = FixedSummarizer("from fallback")
        out = LLMSummarizer(FakeLLM("   \n "), fallback=fallback).summarize([msg("user", "x")], 50)
        assert out == "from fallback"

    def test_fallback_output_also_bounded(self) -> None:
        fallback = FixedSummarizer("word " * 200)
        out = LLMSummarizer(BrokenLLM(), fallback=fallback).summarize([msg("user", "x")], 10)
        assert estimate_tokens(out) <= 10

    def test_empty_and_zero_budget(self) -> None:
        llm = FakeLLM("never")
        s = LLMSummarizer(llm)
        assert s.summarize([], 50) == ""
        assert s.summarize([msg("user", "x")], 0) == ""
        assert s.summarize([msg("system", "only system")], 50) == ""
        assert llm.calls == []

    def test_transcript_renders_prior_summary(self) -> None:
        text = render_transcript(
            [prior_summary_message("User: likes tea."), msg("system", "s"), msg("user", "Hi")]
        )
        assert text == "Earlier summary:\nUser: likes tea.\nUser: Hi"


# --------------------------------------------------------------------------- compressor


def _summary_cost(summary: str) -> int:
    return estimate_tokens(summary) + MESSAGE_OVERHEAD if summary else 0


class TestContextCompressor:
    def test_fits_budget_unchanged(self) -> None:
        msgs = conversation(4)
        summary = "User: earlier fact."
        expected = total_tokens(msgs) + _summary_cost(summary)
        result = ContextCompressor().compress(msgs, expected, existing_summary=summary)
        assert isinstance(result, CompressedContext)
        assert result.summary == summary
        assert result.messages == msgs
        assert all(a is b for a, b in zip(result.messages, msgs, strict=True))
        assert result.compressed_count == 0
        assert result.tokens == expected

    def test_fits_without_summary(self) -> None:
        msgs = conversation(2)
        result = ContextCompressor().compress(msgs, 10_000)
        assert result.summary == ""
        assert result.tokens == total_tokens(msgs)

    def test_over_budget_compressed(self) -> None:
        msgs = conversation(30)
        budget = total_tokens(msgs) // 3
        result = ContextCompressor().compress(msgs, budget)
        assert result.tokens <= budget
        assert result.tokens == _summary_cost(result.summary) + total_tokens(result.messages)
        assert result.compressed_count == len(msgs) - len(result.messages)
        assert result.compressed_count > 0
        assert result.summary

    @pytest.mark.parametrize("budget", [1, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200, 400])
    def test_budget_never_exceeded(self, budget: int) -> None:
        msgs = conversation(30)
        result = ContextCompressor().compress(msgs, budget, existing_summary="User: I own a boat.")
        assert result.tokens <= budget
        assert result.tokens == _summary_cost(result.summary) + total_tokens(result.messages)

    @pytest.mark.parametrize("budget", [3, 7, 20, 60])
    def test_budget_never_exceeded_custom_counter(self, budget: int) -> None:
        msgs = conversation(12)
        compressor = ContextCompressor(token_counter=word_counter)
        result = compressor.compress(msgs, budget, existing_summary="earlier stuff")
        cost = (word_counter(result.summary) + MESSAGE_OVERHEAD if result.summary else 0) + (
            total_tokens(result.messages, word_counter)
        )
        assert result.tokens == cost <= budget

    def test_recent_suffix_kept_in_order_and_input_not_mutated(self) -> None:
        msgs = conversation(20)
        snapshot = [m.to_dict() for m in msgs]
        original_list = list(msgs)
        result = ContextCompressor().compress(msgs, 120)
        n = len(result.messages)
        assert n >= 2
        assert result.messages == msgs[-n:]
        assert msgs == original_list
        assert [m.to_dict() for m in msgs] == snapshot

    def test_min_recent_kept_when_fits_whole_budget(self) -> None:
        msgs = [msg("user", "a" * 40, 1), msg("assistant", "b" * 40, 2), msg("user", "c" * 40, 3)]
        # each message costs 10 + 4 = 14; budget 30 -> non-summary share 21 fits only one
        result = ContextCompressor(summary_ratio=0.3, min_recent=2).compress(msgs, 30)
        assert result.messages == msgs[-2:]
        assert result.tokens <= 30

    def test_min_recent_respects_whole_budget(self) -> None:
        msgs = [msg("user", "a" * 40, 1), msg("assistant", "b" * 40, 2)]
        result = ContextCompressor(min_recent=2).compress(msgs, 20)
        assert result.messages == msgs[-1:]
        assert result.tokens <= 20

    def test_existing_summary_carried_forward(self) -> None:
        msgs = conversation(30)
        existing = "User: My passport number ends in 4471."
        compressor = ContextCompressor(summary_ratio=0.5)
        result = compressor.compress(msgs, 200, existing_summary=existing)
        assert result.compressed_count > 0
        assert "4471" in result.summary
        assert result.tokens <= 200

    def test_existing_summary_passed_to_summarizer(self) -> None:
        fixed = FixedSummarizer("S")
        msgs = conversation(10)
        ContextCompressor(summarizer=fixed).compress(msgs, 60, existing_summary="prior info")
        sent, max_tokens = fixed.calls[-1]
        assert "prior info" in sent[0].content
        assert sent[0].metadata.get("summary") is True
        assert max_tokens > 0

    def test_summarizer_output_is_bounded(self) -> None:
        msgs = conversation(10)
        result = ContextCompressor(summarizer=FixedSummarizer("blah " * 500)).compress(msgs, 60)
        assert result.tokens <= 60

    def test_leftover_given_back_to_messages(self) -> None:
        msgs = conversation(10)
        budget = 100
        tiny = ContextCompressor(summarizer=FixedSummarizer("S"), summary_ratio=0.5)
        result = tiny.compress(msgs, budget)
        # without give-back only ~50 tokens of messages would be kept
        assert total_tokens(result.messages) > budget // 2
        assert result.tokens <= budget
        assert result.messages == msgs[-len(result.messages) :]

    def test_single_huge_message_truncated(self) -> None:
        older = msg("user", "older message here", 1)
        huge = msg("user", "word " * 500, 2)
        result = ContextCompressor().compress([older, huge], 50)
        assert len(result.messages) == 1
        clipped = result.messages[0]
        assert clipped.id == huge.id
        assert clipped.content != huge.content
        assert huge.content.startswith(clipped.content[: -len(ELLIPSIS)])
        assert clipped.metadata.get("truncated") is True
        assert huge.metadata == {}
        assert huge.content == "word " * 500
        assert result.summary == ""
        assert result.tokens <= 50
        assert result.compressed_count == 1

    def test_huge_message_with_budget_below_overhead(self) -> None:
        result = ContextCompressor().compress([msg("user", "word " * 50)], MESSAGE_OVERHEAD)
        assert result.messages == []
        assert result.tokens == 0

    def test_only_existing_summary_over_budget(self) -> None:
        result = ContextCompressor().compress([], 20, existing_summary="I like tea. " * 40)
        assert result.messages == []
        assert 0 < result.tokens <= 20

    @pytest.mark.parametrize("budget", [0, -5])
    def test_invalid_budget(self, budget: int) -> None:
        with pytest.raises(ValueError):
            ContextCompressor().compress(conversation(2), budget)

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            ContextCompressor(summary_ratio=1.0)
        with pytest.raises(ValueError):
            ContextCompressor(min_recent=-1)


# --------------------------------------------------------------------------- rolling summary


class TestRollingSummary:
    def test_absorb_bounded(self) -> None:
        rs = RollingSummary(max_tokens=20)
        out = rs.absorb(conversation(20))
        assert out == rs.text
        assert out
        assert estimate_tokens(out) <= 20

    def test_callable_as_on_evict(self) -> None:
        rs = RollingSummary(max_tokens=100)
        out = rs([msg("user", "I am allergic to peanuts.", 1)])
        assert "peanuts" in out
        assert rs.text == out

    def test_accepts_single_message(self) -> None:
        rs = RollingSummary(max_tokens=100)
        assert "Lisbon" in rs.absorb(msg("user", "I moved to Lisbon.", 1))

    def test_prior_content_folded_in(self) -> None:
        rs = RollingSummary(max_tokens=300)
        rs.absorb([msg("user", "My sister is called Hana.", 1)])
        rs.absorb([msg("user", "I started learning piano.", 2)])
        assert "Hana" in rs.text
        assert "piano" in rs.text
        assert rs.text.splitlines()[0].startswith("User: My sister")

    def test_empty_or_system_only_batch_keeps_text(self) -> None:
        rs = RollingSummary(text="User: existing.")
        assert rs.absorb([]) == "User: existing."
        assert rs.absorb([msg("system", "sys")]) == "User: existing."

    def test_custom_summarizer_bounded(self) -> None:
        rs = RollingSummary(summarizer=FixedSummarizer("long " * 100), max_tokens=10)
        assert estimate_tokens(rs.absorb([msg("user", "x")])) <= 10

    def test_reset_and_roundtrip(self) -> None:
        rs = RollingSummary(max_tokens=77)
        rs.absorb([msg("user", "I drive a red bicycle.", 1)])
        data = rs.to_dict()
        restored = RollingSummary.from_dict(data)
        assert restored.text == rs.text
        assert restored.max_tokens == 77
        restored.reset()
        assert restored.text == ""
        assert rs.text  # original untouched

    def test_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError):
            RollingSummary(max_tokens=0)
