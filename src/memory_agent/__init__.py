"""Memory-enabled conversational agent."""

from .embeddings import Embedder, HashingEmbedder, cosine_similarity
from .types import MemoryItem, Message, ScoredMemory

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "MemoryItem",
    "Message",
    "ScoredMemory",
    "cosine_similarity",
]
__version__ = "0.1.0"
