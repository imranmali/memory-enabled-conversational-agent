"""Deterministic extraction of durable user facts from conversational text.

:func:`extract_facts` turns first-person statements into third-person facts, e.g.
``"My name is Ada and I love hiking."`` → ``["User's name is Ada", "User loves hiking"]``.
It is regex-based (English only), offline and deterministic, and deliberately conservative:
questions, negations and statements without a supported cue yield nothing.

:func:`fact_key` identifies *single-valued* facts (name, birthday, home, a favourite ...) so a
newer value can supersede an older one.
"""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["conjugate_third_person", "extract_facts", "fact_key", "normalize_pronouns"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_FIRST_PERSON_OPENER = r"(?:i|i'm|im|i've|i'd|i'll|my|call me|please|remember|don't forget)\b"
_CLAUSE_SPLIT = re.compile(
    rf"\s*(?:[;,]\s*(?:and|but|also|plus|so)?\s*|\s+(?:and|but|also|plus|so)\s+)"
    rf"(?=(?:also\s+)?{_FIRST_PERSON_OPENER})",
    re.IGNORECASE,
)
_QUESTION_OPENER = re.compile(
    r"^(?:what|who|whom|whose|when|where|why|how|which|do|does|did|can|could|would|will|"
    r"should|shall|is|are|am|was|were|have|has|had)\b",
    re.IGNORECASE,
)
_TRAILING = re.compile(r"[\s.!,;:\"')\]]+$")
_LEADING = re.compile(r"^[\s\"'(\[]+")
_TRAILING_FILLER = re.compile(r"\s+(?:too|as well|though|anyway|lol)$", re.IGNORECASE)
_LEADING_FILLER = re.compile(
    r"^(?:(?:oh|well|so|also|and|but|btw|by the way|actually|hi|hello|hey|ok|okay)\b[\s,]*)+",
    re.IGNORECASE,
)

_MODALS = frozenset("can could will would shall should may might must".split())
_IRREGULAR = {"have": "has", "do": "does", "go": "goes", "be": "is", "am": "is"}


def conjugate_third_person(verb: str) -> str:
    """Third-person singular present of an English verb (``love`` → ``loves``)."""
    lower = verb.lower()
    if lower in _MODALS:
        return verb
    if lower in _IRREGULAR:
        return _IRREGULAR[lower]
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if lower.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    return verb + "s"


def _possessives_to_their(text: str) -> str:
    """Rewrite remaining first-person words inside a fact value (``my dog`` → ``their dog``)."""
    replacements = (
        (r"\bmyself\b", "themselves"),
        (r"\bmine\b", "theirs"),
        (r"\bmy\b", "their"),
        (r"\bme\b", "them"),
    )
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


_SUBJECT_FORMS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    (re.compile(r"^my\s+", re.I), lambda m: "User's "),
    (re.compile(r"^(?:i am|i'm|im)\s+", re.I), lambda m: "User is "),
    (re.compile(r"^(?:i have|i've)\s+", re.I), lambda m: "User has "),
    (re.compile(r"^i was\s+", re.I), lambda m: "User was "),
    (re.compile(r"^i'll\s+", re.I), lambda m: "User will "),
    (re.compile(r"^i'd\s+", re.I), lambda m: "User would "),
    (re.compile(r"^i\s+(\w+)", re.I), lambda m: "User " + conjugate_third_person(m.group(1))),
)


def normalize_pronouns(clause: str) -> str:
    """Rewrite a first-person clause about the user into the third person.

    ``"my sister is Jane"`` → ``"User's sister is Jane"``;
    ``"I need to call my mom"`` → ``"User needs to call their mom"``.
    """
    clause = clause.strip()
    for pattern, render in _SUBJECT_FORMS:
        match = pattern.match(clause)
        if match:
            clause = render(match) + clause[match.end() :]
            break
    return _possessives_to_their(clause)


def _clean_value(value: str) -> str:
    value = _TRAILING.sub("", _LEADING.sub("", value))
    value = _TRAILING_FILLER.sub("", value)
    return _TRAILING.sub("", value).strip()


def _title_if_lower(value: str) -> str:
    return value.title() if value == value.lower() else value


_NAME_LIKE = re.compile(r"^[A-Za-z][\w'.-]*(?:\s+[A-Za-z][\w'.-]*){0,3}$")

Rule = tuple[re.Pattern[str], Callable[[re.Match[str]], str | None]]


def _name(match: re.Match[str]) -> str | None:
    value = _clean_value(match.group("v"))
    if not _NAME_LIKE.match(value):
        return None
    return f"User's name is {_title_if_lower(value)}"


def _call_me(match: re.Match[str]) -> str | None:
    value = _clean_value(match.group("v"))
    if not _NAME_LIKE.match(value):
        return None
    return f"User prefers to be called {_title_if_lower(value)}"


_EMBEDDED_QUESTION = re.compile(
    r"^(?:what|where|when|who|whom|whose|which|why|how|whether|if)\b", re.I
)


def _remember(match: re.Match[str]) -> str | None:
    value = _clean_value(match.group("v"))
    if not value or _EMBEDDED_QUESTION.match(value):
        return None
    fact = normalize_pronouns(value)
    return fact[0].upper() + fact[1:]


def _preference(match: re.Match[str]) -> str:
    adverb = match.group("adv")
    verb = conjugate_third_person(match.group("verb").lower())
    prefix = f"User {adverb.lower()} " if adverb else "User "
    return f"{prefix}{verb} {_possessives_to_their(_clean_value(match.group('v')))}"


def _simple(template: str) -> Callable[[re.Match[str]], str]:
    def render(match: re.Match[str]) -> str:
        groups = {
            k: _possessives_to_their(_clean_value(v or "")) for k, v in match.groupdict().items()
        }
        return template.format(**groups)

    return render


_I_AM = r"(?:i am|i'm|im)"
_VALUE = r"(?P<v>.+)"

# Ordered: the first rule matching a clause wins.
_RULES: tuple[Rule, ...] = (
    (
        re.compile(
            rf"^(?:please\s+)?(?:remember|don't forget|do not forget|note)\s+(?:that\s+)?{_VALUE}$",
            re.I,
        ),
        _remember,
    ),
    (
        re.compile(rf"^(?:{_I_AM}\s+)?not\b|\b(?:don't|do not|never|no longer)\b", re.I),
        lambda m: None,
    ),
    (re.compile(rf"\bmy name(?: is|'s)\s+{_VALUE}", re.I), _name),
    (re.compile(rf"\b(?:you can |please )?call me\s+{_VALUE}", re.I), _call_me),
    (re.compile(r"^(?i:i am|i'm|im)\s+(?P<v>[A-Z][\w'-]*)$"), _name),
    (
        re.compile(rf"\bmy birthday(?: is|'s)\s+(?:on\s+)?{_VALUE}", re.I),
        _simple("User's birthday is {v}"),
    ),
    (
        re.compile(rf"\bmy favou?rite\s+(?P<thing>[\w ]+?)\s+(?:is|are)\s+{_VALUE}", re.I),
        _simple("User's favorite {thing} is {v}"),
    ),
    (re.compile(rf"\b{_I_AM}\s+allergic to\s+{_VALUE}", re.I), _simple("User is allergic to {v}")),
    (
        re.compile(rf"\bi\s+(?:live|am living|'m living)\s+in\s+{_VALUE}", re.I),
        _simple("User lives in {v}"),
    ),
    (re.compile(rf"\b(?:{_I_AM}|i come)\s+from\s+{_VALUE}", re.I), _simple("User is from {v}")),
    (re.compile(rf"\bi\s+work\s+as\s+{_VALUE}", re.I), _simple("User works as {v}")),
    (re.compile(rf"\bi\s+work\s+(?:at|for)\s+{_VALUE}", re.I), _simple("User works at {v}")),
    (re.compile(rf"\bi\s+work\s+in\s+{_VALUE}", re.I), _simple("User works in {v}")),
    (
        re.compile(
            r"\bi\s+(?:(?P<adv>really|also|absolutely|totally)\s+)?"
            rf"(?P<verb>like|love|enjoy|prefer|hate|dislike|adore)\s+{_VALUE}",
            re.I,
        ),
        _preference,
    ),
    (re.compile(rf"\b{_I_AM}\s+(?P<a>an?)\s+{_VALUE}", re.I), _simple("User is {a} {v}")),
)


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") or bool(_QUESTION_OPENER.match(sentence.strip()))


def _clauses(sentence: str) -> list[str]:
    parts = _CLAUSE_SPLIT.split(sentence)
    return [p for p in (_LEADING_FILLER.sub("", _clean_value(p)) for p in parts) if p]


def _fact_from_clause(clause: str) -> str | None:
    for pattern, render in _RULES:
        match = pattern.search(clause)
        if match:
            fact = render(match)
            if fact is None:
                return None
            fact = _TRAILING.sub("", fact).strip()
            # Reject cues with nothing after them ("I love").
            return fact if len(fact.split()) >= 3 else None
    return None


def extract_facts(text: str) -> list[str]:
    """Durable third-person facts stated by the user in ``text``, de-duplicated, in order."""
    if not text or not text.strip():
        return []
    normalized = text.replace("’", "'").replace("‘", "'")
    facts: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(normalized):
        sentence = sentence.strip()
        if not sentence or _is_question(sentence):
            continue
        for clause in _clauses(sentence):
            fact = _fact_from_clause(clause)
            if fact and fact.lower() not in seen:
                seen.add(fact.lower())
                facts.append(fact)
    return facts


_KEY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^User's name is\b"), "name"),
    (re.compile(r"^User prefers to be called\b"), "preferred-name"),
    (re.compile(r"^User's birthday is\b"), "birthday"),
    (re.compile(r"^User lives in\b"), "home"),
    (re.compile(r"^User's favorite (?P<thing>[\w ]+?) is\b"), "favorite:{thing}"),
)


def fact_key(fact: str) -> str | None:
    """Key shared by facts that hold a single value at a time, or ``None`` for other facts."""
    for pattern, template in _KEY_PATTERNS:
        match = pattern.match(fact.strip())
        if match:
            return template.format(**{k: v.lower() for k, v in match.groupdict().items()})
    return None
