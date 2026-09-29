"""Semantic retrieval over the vector store.

The retriever only orchestrates: turn a query into an embedding, ask the vector
store for neighbours, and wrap the hits into :class:`RetrievedChunk` objects that
carry provenance (document id, page, score) for the evidence section of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ProviderNotConfiguredError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.vector_store.base import VectorStore
from app.rag.vector_store.factory import get_vector_store

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by a similarity search."""

    chunk_id: str
    text: str
    score: float
    source_id: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """Embeds a query and returns the most relevant stored chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for ``query``."""
        query = (query or "").strip()
        if not query:
            return []
        if not self._embedding_provider.is_configured:
            raise ProviderNotConfiguredError(
                "Semantic search needs an embedding provider. Set "
                "EMBEDDING_PROVIDER, EMBEDDING_MODEL and EMBEDDING_API_KEY."
            )

        limit = top_k or get_settings().retrieval_top_k
        embedding = self._embedding_provider.embed_query(query)
        results = self._vector_store.query(embedding, top_k=limit, where=where)

        chunks: list[RetrievedChunk] = []
        for result in results:
            metadata = dict(result.metadata or {})
            page_number = metadata.get("page_number")
            chunks.append(
                RetrievedChunk(
                    chunk_id=result.id,
                    text=result.text,
                    score=result.score,
                    source_id=metadata.get("source_id"),
                    page_number=int(page_number) if page_number is not None else None,
                    metadata=metadata,
                )
            )
        logger.debug("Retrieved %d chunk(s) for query.", len(chunks))
        return chunks


@lru_cache
def get_retriever() -> Retriever:
    """Return a retriever wired to the configured providers."""
    return Retriever(get_embedding_provider(), get_vector_store())


def reset_retriever_cache() -> None:
    """Clear the cache (used by tests and after configuration changes)."""
    get_retriever.cache_clear()
