"""Embedding models and vector math.

``HashingEmbedder`` is deterministic and offline (signed feature hashing over
unigrams and bigrams), so the whole agent runs and tests without network access.
Any object satisfying the :class:`Embedder` protocol can be swapped in.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = frozenset(
    "a an and are as at be but by do does for from had has have i if in is it its "
    "me my of on or so that the their them they this to was we were what when where "
    "which who will with you your".split()
)


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """Deterministic bag-of-ngrams embedder using signed feature hashing."""

    def __init__(self, dim: int = 512, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = [_stem(t) for t in tokenize(text)]
        feats = [(t, 1.0) for t in tokens]
        if self.use_bigrams:
            feats += [(f"{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:], strict=False)]
        return feats

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign * weight
        return normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
