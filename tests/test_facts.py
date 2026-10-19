import pytest

from memory_agent.facts import (
    conjugate_third_person,
    extract_facts,
    fact_key,
    normalize_pronouns,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My name is Ada.", ["User's name is Ada"]),
        ("my name's ada lovelace", ["User's name is Ada Lovelace"]),
        ("I'm Ada", ["User's name is Ada"]),
        ("Hi, I'm Ada!", ["User's name is Ada"]),
        ("Call me Bob", ["User prefers to be called Bob"]),
        ("I live in Berlin.", ["User lives in Berlin"]),
        ("I'm from Paris", ["User is from Paris"]),
        ("I come from Lagos", ["User is from Lagos"]),
        ("I work as a nurse", ["User works as a nurse"]),
        ("I work at Acme Corp", ["User works at Acme Corp"]),
        ("I work for the city", ["User works at the city"]),
        ("I work in finance", ["User works in finance"]),
        ("I am a software engineer", ["User is a software engineer"]),
        ("I'm an architect", ["User is an architect"]),
        ("I love hiking!", ["User loves hiking"]),
        ("I like jazz", ["User likes jazz"]),
        ("I enjoy chess", ["User enjoys chess"]),
        ("I prefer tea", ["User prefers tea"]),
        ("I hate mondays", ["User hates mondays"]),
        ("I dislike crowds", ["User dislikes crowds"]),
        ("I really enjoy my morning coffee", ["User really enjoys their morning coffee"]),
        ("I'm allergic to peanuts.", ["User is allergic to peanuts"]),
        ("My birthday is on May 3rd", ["User's birthday is May 3rd"]),
        ("My favourite color is blue", ["User's favorite color is blue"]),
        ("my favorite foods are tacos", ["User's favorite foods is tacos"]),
        ("I love hiking too", ["User loves hiking"]),
    ],
)
def test_single_cues(text, expected):
    assert extract_facts(text) == expected


def test_multiple_clauses_split_in_order():
    assert extract_facts("My name is Ada and I love hiking.") == [
        "User's name is Ada",
        "User loves hiking",
    ]
    assert extract_facts("I live in Berlin, and my favorite food is pizza") == [
        "User lives in Berlin",
        "User's favorite food is pizza",
    ]
    assert extract_facts("my name is Ada Lovelace but call me Ada") == [
        "User's name is Ada Lovelace",
        "User prefers to be called Ada",
    ]


def test_compound_object_stays_together():
    assert extract_facts("I love hiking and climbing") == ["User loves hiking and climbing"]


def test_remember_requests_normalize_pronouns():
    assert extract_facts("Please remember that my sister is called Jane") == [
        "User's sister is called Jane"
    ]
    assert extract_facts("remember that I need to call my mom.") == ["User needs to call their mom"]
    assert extract_facts("Don't forget that I'm vegetarian") == ["User is vegetarian"]
    assert extract_facts("Remember: the meeting is at 5") == []  # no "that" clause cue
    assert extract_facts("remember that the meeting is at 5") == ["The meeting is at 5"]


@pytest.mark.parametrize(
    "text",
    [
        "What is my name?",
        "what do you remember about me",
        "Do you know where I live?",
        "I love hiking, do you?",
        "I don't like rain",
        "I'm not a doctor",
        "I never eat meat",
        "hello there",
        "",
        "   ",
        "I like",
        "I study physics",
        "I'm happy to help",
        "My name is 42!!",
    ],
)
def test_non_facts_are_ignored(text):
    assert extract_facts(text) == []


def test_dedupes_preserving_order():
    assert extract_facts("I love tea. I love tea. My name is Bo.") == [
        "User loves tea",
        "User's name is Bo",
    ]
    assert extract_facts("I love tea.\nI LOVE TEA") == ["User loves tea"]


def test_curly_apostrophes():
    assert extract_facts("I’m allergic to cats") == ["User is allergic to cats"]


@pytest.mark.parametrize(
    ("verb", "expected"),
    [
        ("love", "loves"),
        ("study", "studies"),
        ("play", "plays"),
        ("watch", "watches"),
        ("go", "goes"),
        ("have", "has"),
        ("can", "can"),
        ("am", "is"),
    ],
)
def test_conjugation(verb, expected):
    assert conjugate_third_person(verb) == expected


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        ("my dog is Rex", "User's dog is Rex"),
        ("I am tired of my job", "User is tired of their job"),
        ("I've got two cats", "User has got two cats"),
        ("I was born in Rome", "User was born in Rome"),
        ("I'll move next year", "User will move next year"),
        ("I'd like to visit me mum", "User would like to visit them mum"),
        ("the car is mine", "the car is theirs"),
    ],
)
def test_normalize_pronouns(clause, expected):
    assert normalize_pronouns(clause) == expected


def test_fact_keys():
    assert fact_key("User's name is Ada") == fact_key("User's name is Bob") == "name"
    assert fact_key("User prefers to be called Al") == "preferred-name"
    assert fact_key("User's birthday is May 3") == "birthday"
    assert fact_key("User lives in Paris") == "home"
    assert fact_key("User's favorite Color is blue") == "favorite:color"
    assert fact_key("User loves hiking") is None
