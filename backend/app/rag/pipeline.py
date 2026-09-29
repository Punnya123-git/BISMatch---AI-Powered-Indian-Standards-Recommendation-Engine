"""RAG pipeline orchestration: indexing and retrieval entry points.

The pipeline is the only place that knows how chunking, embeddings and the
vector store fit together. It contains no FastAPI code and no fabricated data:
if the embedding provider or the standards index is missing, it reports that
state through :meth:`RecommendationPipeline.readiness` instead of inventing
recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

from app.core.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.document_processing.entities import ExtractedPage
from app.rag.chunking import TextChunk, chunk_pages, chunk_text
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.retrieval import RetrievedChunk, Retriever
from app.rag.vector_store.base import VectorRecord, VectorStore
from app.rag.vector_store.factory import get_vector_store

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineReadiness:
    """Whether the retrieval side of the system can currently work."""

    ready: bool
    embedding_provider: str
    vector_store: str
    indexed_chunks: int = 0
    reasons: list[str] = field(default_factory=list)


class RecommendationPipeline:
    """Indexes standards/documents and retrieves evidence for a requirement."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        retriever: Retriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._retriever = retriever

    # --- lazily resolved components --------------------------------------
    @property
    def embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.embedding_provider, self.vector_store)
        return self._retriever

    # --- status -----------------------------------------------------------
    def readiness(self) -> PipelineReadiness:
        """Describe what is missing before real recommendations are possible."""
        reasons: list[str] = []
        indexed_chunks = 0

        embedding_status = self.embedding_provider.describe()
        if not self.embedding_provider.is_configured:
            reasons.append(
                "No embedding provider is configured "
                "(EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_API_KEY)."
            )

        vector_store_status = "unavailable"
        try:
            vector_store = self.vector_store
            vector_store_status = vector_store.provider_name
            indexed_chunks = vector_store.count()
            if indexed_chunks == 0:
                reasons.append(
                    "The standards index is empty. Load and index the Indian "
                    "Standards dataset before requesting recommendations."
                )
        except VectorStoreError as exc:
            reasons.append(str(exc))

        return PipelineReadiness(
            ready=not reasons,
            embedding_provider=embedding_status,
            vector_store=vector_store_status,
            indexed_chunks=indexed_chunks,
            reasons=reasons,
        )

    # --- indexing ---------------------------------------------------------
    def index_chunks(self, chunks: Sequence[TextChunk]) -> int:
        """Embed and store chunks. Returns the number of vectors written."""
        chunks = list(chunks)
        if not chunks:
            return 0
        if not self.embedding_provider.is_configured:
            raise VectorStoreError(
                "Cannot index without an embedding provider. Configure the "
                "embedding provider and retry."
            )
        embeddings = self.embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise VectorStoreError(
                "Embedding provider returned a different number of vectors than chunks."
            )
        records = [
            VectorRecord(
                id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.vector_metadata(),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        return self.vector_store.add(records)

    def index_text(
        self,
        text: str,
        *,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed and store a plain text source."""
        chunks = chunk_text(
            text,
            source_id=source_id,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            metadata=metadata,
        )
        return self.index_chunks(chunks)

    def index_pages(
        self,
        pages: Iterable[ExtractedPage],
        *,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed and store page-aware extracted text."""
        chunks = chunk_pages(
            pages,
            source_id=source_id,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            metadata=metadata,
        )
        return self.index_chunks(chunks)

    # --- retrieval --------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve evidence chunks for a requirement."""
        return self.retriever.retrieve(query, top_k=top_k, where=where)

    def reset_index(self) -> None:
        """Drop every indexed vector (useful when reloading the dataset)."""
        self.vector_store.reset()


@lru_cache
def get_recommendation_pipeline() -> RecommendationPipeline:
    """Return the process-wide pipeline instance."""
    return RecommendationPipeline()


def reset_recommendation_pipeline_cache() -> None:
    """Clear the cache (used by tests and after configuration changes)."""
    get_recommendation_pipeline.cache_clear()
