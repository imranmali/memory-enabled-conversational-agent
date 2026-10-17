import math

import pytest

from memory_agent.embeddings import (
    Embedder,
    HashingEmbedder,
    cosine_similarity,
    normalize,
    tokenize,
)


def test_deterministic_and_dimension():
    a = HashingEmbedder(dim=64).embed_one("The quick brown fox")
    b = HashingEmbedder(dim=64).embed_one("The quick brown fox")
    assert a == b and len(a) == 64


def test_unit_length():
    v = HashingEmbedder().embed_one("I enjoy long walks on the beach")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_empty_text_is_zero_vector():
    assert not any(HashingEmbedder(dim=8).embed_one(""))


def test_invalid_dimension():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_related_beats_unrelated():
    e = HashingEmbedder()
    a, b, c = e.embed(
        [
            "I love hiking in the mountains",
            "hiking mountains is my favourite",
            "tax return deadline",
        ]
    )
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


def test_inflections_match():
    e = HashingEmbedder()
    assert cosine_similarity(e.embed_one("hikes"), e.embed_one("hiking")) > 0


def test_tokenize_drops_stopwords():
    assert tokenize("The cat and THE hat") == ["cat", "hat"]


def test_cosine_basics():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_normalize_zero_vector():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_satisfies_protocol():
    assert isinstance(HashingEmbedder(), Embedder)
