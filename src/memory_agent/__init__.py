"""Memory-enabled conversational agent.

Short-term buffer + long-term vector recall, context compression, relevance scoring and
cross-session sync, tied together by :class:`MemoryAgent`.
"""

from .agent import AgentConfig, AgentReply, MemoryAgent, default_device_id
from .compression import (
    CompressedContext,
    ContextCompressor,
    ExtractiveSummarizer,
    LLMSummarizer,
    RollingSummary,
    truncate_to_tokens,
)
from .embeddings import Embedder, HashingEmbedder, cosine_similarity, tokenize
from .facts import extract_facts, fact_key
from .interfaces import LLM, Scorer, Summarizer, VectorStore
from .llm import (
    FACTS_HEADER,
    MEMORIES_HEADER,
    PREVIOUS_SESSION_HEADER,
    SUMMARY_HEADER,
    AnthropicLLM,
    OfflineLLM,
    build_llm,
    parse_sections,
)
from .memory import LongTermMemory
from .scoring import (
    RelevanceScorer,
    ScoringWeights,
    deduplicate,
    estimate_importance,
    maximal_marginal_relevance,
)
from .short_term import ShortTermBuffer
from .stores import InMemoryVectorStore, SQLiteVectorStore
from .sync import (
    FileSyncBackend,
    InMemorySyncBackend,
    MemorySynchronizer,
    SessionInfo,
    SessionManager,
    SyncBackend,
    SyncReport,
    SyncStorageError,
    resolve_conflict,
)
from .tokens import estimate_tokens, message_tokens, total_tokens
from .types import MemoryItem, Message, ScoredMemory

__all__ = [
    "FACTS_HEADER",
    "LLM",
    "MEMORIES_HEADER",
    "PREVIOUS_SESSION_HEADER",
    "SUMMARY_HEADER",
    "AgentConfig",
    "AgentReply",
    "AnthropicLLM",
    "CompressedContext",
    "ContextCompressor",
    "Embedder",
    "ExtractiveSummarizer",
    "FileSyncBackend",
    "HashingEmbedder",
    "InMemorySyncBackend",
    "InMemoryVectorStore",
    "LLMSummarizer",
    "LongTermMemory",
    "MemoryAgent",
    "MemoryItem",
    "MemorySynchronizer",
    "Message",
    "OfflineLLM",
    "RelevanceScorer",
    "RollingSummary",
    "SQLiteVectorStore",
    "ScoredMemory",
    "Scorer",
    "ScoringWeights",
    "SessionInfo",
    "SessionManager",
    "ShortTermBuffer",
    "Summarizer",
    "SyncBackend",
    "SyncReport",
    "SyncStorageError",
    "VectorStore",
    "build_llm",
    "cosine_similarity",
    "deduplicate",
    "default_device_id",
    "estimate_importance",
    "estimate_tokens",
    "extract_facts",
    "fact_key",
    "maximal_marginal_relevance",
    "message_tokens",
    "parse_sections",
    "resolve_conflict",
    "tokenize",
    "total_tokens",
    "truncate_to_tokens",
]
__version__ = "0.1.0"
