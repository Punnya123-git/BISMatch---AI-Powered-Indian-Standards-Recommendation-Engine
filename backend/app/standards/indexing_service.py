"""Index the verified Indian Standards dataset into the vector store.

Flow (each step reuses an existing abstraction):

    validated dataset -> searchable documents -> chunks -> embeddings -> Chroma

Nothing is fabricated at any step:

* the embedding provider must be configured, otherwise the service returns the
  explicit status ``embedding_provider_not_configured`` and writes nothing;
* chunking uses :mod:`app.rag.chunking` and writing uses
  :meth:`RecommendationPipeline.index_chunks`, so vectors are always produced by
  the configured provider;
* indexing is never triggered implicitly by a request. It is built either
  explicitly with ``python -m app.rag.index_standards`` or once at process
  start by :mod:`app.standards.bootstrap`, and both write only real vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AppError,
    DatasetNotAvailableError,
    DatasetValidationError,
    VectorStoreError,
)
from app.core.logging import get_logger
from app.rag.chunking import TextChunk, chunk_text
from app.rag.pipeline import RecommendationPipeline, get_recommendation_pipeline
from app.standards.dataset_loader import (
    StandardsDatasetLoader,
    get_standards_dataset_loader,
)
from app.standards.documents import StandardDocument, build_standard_documents

logger = get_logger(__name__)

#: Outcome states of an indexing run (explicit, never implicit).
STATUS_INDEXED = "indexed"
STATUS_EMBEDDING_NOT_CONFIGURED = "embedding_provider_not_configured"
STATUS_DATASET_NOT_AVAILABLE = "dataset_not_available"
STATUS_DATASET_INVALID = "dataset_invalid"
STATUS_VECTOR_STORE_UNAVAILABLE = "vector_store_unavailable"


@dataclass(slots=True)
class IndexingReport:
    """Result of one indexing run, including what was *not* done and why."""

    status: str
    message: str
    dataset_version: str | None = None
    standards_count: int = 0
    document_count: int = 0
    chunk_count: int = 0
    written: int = 0
    indexed_chunks: int = 0
    embedding_provider: str = "unknown"
    vector_store: str = "unknown"
    reasons: list[str] = field(default_factory=list)

    @property
    def indexed(self) -> bool:
        """True only when chunks were actually written to the vector store."""
        return self.status == STATUS_INDEXED


class StandardsIndexingService:
    """Turns the standards dataset into embedded, searchable chunks."""

    def __init__(
        self,
        *,
        loader: StandardsDatasetLoader | None = None,
        pipeline: RecommendationPipeline | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._loader = loader or get_standards_dataset_loader()
        self._pipeline = pipeline or get_recommendation_pipeline()

    @property
    def loader(self) -> StandardsDatasetLoader:
        return self._loader

    @property
    def pipeline(self) -> RecommendationPipeline:
        return self._pipeline

    def prepare_chunks(self, documents: list[StandardDocument]) -> list[TextChunk]:
        """Chunk every prepared document with the configured chunk settings."""
        chunks: list[TextChunk] = []
        for document in documents:
            chunks.extend(
                chunk_text(
                    document.text,
                    source_id=document.source_id,
                    chunk_size=self._settings.chunk_size,
                    chunk_overlap=self._settings.chunk_overlap,
                    metadata=document.metadata,
                )
            )
        return chunks

    @staticmethod
    def _base_report(
        status: str, message: str, *, embedding: str, vector_store: str
    ) -> IndexingReport:
        return IndexingReport(
            status=status,
            message=message,
            embedding_provider=embedding,
            vector_store=vector_store,
        )

    def _component_descriptions(self) -> tuple[str, str]:
        """Describe the configured components without raising."""
        embedding = "unavailable"
        vector_store = "unavailable"
        try:
            embedding = self._pipeline.embedding_provider.describe()
        except AppError as exc:  # pragma: no cover - defensive
            embedding = f"unavailable ({exc.message})"
        try:
            vector_store = self._pipeline.vector_store.provider_name
        except AppError:
            logger.warning("Vector store is unavailable while indexing standards.")
        return embedding, vector_store

    def _stored_chunk_count(self) -> int:
        try:
            return self._pipeline.vector_store.count()
        except AppError:
            return 0

    def stored_chunk_count(self) -> int:
        """How many chunks the vector store currently holds (0 when unavailable)."""
        return self._stored_chunk_count()

    def expected_chunk_count(self) -> int:
        """How many chunks the configured dataset produces.

        Pure preparation - the dataset is read, rendered and chunked, but
        nothing is embedded and nothing is written. Cheap enough to call on
        every process start to decide whether an index is already current.
        """
        dataset = self._loader.load()
        return len(self.prepare_chunks(build_standard_documents(dataset)))

    def is_index_current(self) -> bool:
        """True when the store already holds every chunk of the current dataset.

        Comparing against the *expected* count (instead of "is it non-empty")
        also catches a half-finished previous run, which would otherwise look
        like a finished index.
        """
        try:
            expected = self.expected_chunk_count()
        except AppError:
            return False
        if expected <= 0:
            return False
        return self._stored_chunk_count() == expected

    def index_if_needed(self, *, force: bool = False) -> IndexingReport:
        """Build the index only when the store does not already match the dataset.

        Idempotent by construction: chunk ids are derived from the record and
        the text, and the store upserts, so re-running converges on the same
        collection instead of duplicating it. ``reset`` is deliberately not
        used here - it would also drop vectors belonging to uploaded documents,
        which share the same collection.
        """
        if not force and self.is_index_current():
            embedding, vector_store = self._component_descriptions()
            stored = self._stored_chunk_count()
            return IndexingReport(
                status=STATUS_INDEXED,
                message=(
                    f"The standards index is already up to date: {stored} chunk(s) "
                    f"in the '{vector_store}' store match the configured dataset."
                ),
                embedding_provider=embedding,
                vector_store=vector_store,
                indexed_chunks=stored,
            )
        return self.index()

    def _embedding_reasons(self) -> list[str]:
        """Explain why nothing could be embedded, using the provider's own words.

        The configured provider knows precisely what is missing, so its
        ``reason`` is quoted verbatim. The generic hint is phrased per provider:
        the local ONNX model needs no API key, while the remote providers do.
        """
        provider = self._pipeline.embedding_provider
        detail = getattr(provider, "reason", None)
        if provider.provider_name == "onnx_minilm":
            hint = (
                "Set EMBEDDING_PROVIDER=onnx_minilm (optionally "
                "EMBEDDING_MODEL=all-MiniLM-L6-v2) in the backend environment, "
                "then run this command again. No EMBEDDING_API_KEY is needed."
            )
        else:
            hint = (
                "Set EMBEDDING_PROVIDER, EMBEDDING_MODEL and EMBEDDING_API_KEY in "
                "the backend environment, then run this command again."
            )
        reasons = [reason for reason in (detail, hint) if reason]
        reasons.append("No placeholder or fake embeddings are ever generated.")
        return reasons

    # --- entry point ------------------------------------------------------
    def index(self, *, reset: bool = False) -> IndexingReport:
        """Load, prepare and index the dataset, reporting the honest outcome."""
        embedding, vector_store = self._component_descriptions()

        try:
            dataset = self._loader.load()
        except DatasetNotAvailableError as exc:
            return self._base_report(
                STATUS_DATASET_NOT_AVAILABLE,
                exc.message,
                embedding=embedding,
                vector_store=vector_store,
            )
        except DatasetValidationError as exc:
            report = self._base_report(
                STATUS_DATASET_INVALID,
                exc.message,
                embedding=embedding,
                vector_store=vector_store,
            )
            report.reasons = [str(issue) for issue in (exc.details or [])]
            return report

        documents = build_standard_documents(dataset)
        chunks = self.prepare_chunks(documents)

        if not self._pipeline.embedding_provider.is_configured:
            report = self._base_report(
                STATUS_EMBEDDING_NOT_CONFIGURED,
                (
                    "Nothing was indexed: no embedding provider is configured, so "
                    "the standards could not be embedded. The dataset itself is "
                    "valid and the searchable documents were prepared."
                ),
                embedding=embedding,
                vector_store=vector_store,
            )
            report.dataset_version = dataset.version
            report.standards_count = dataset.count
            report.document_count = len(documents)
            report.chunk_count = len(chunks)
            report.indexed_chunks = self._stored_chunk_count()
            report.reasons = self._embedding_reasons()
            return report

        try:
            if reset:
                self._pipeline.reset_index()
            written = self._pipeline.index_chunks(chunks)
        except VectorStoreError as exc:
            report = self._base_report(
                STATUS_VECTOR_STORE_UNAVAILABLE,
                f"Nothing was indexed: {exc.message}",
                embedding=embedding,
                vector_store=vector_store,
            )
            report.dataset_version = dataset.version
            report.standards_count = dataset.count
            report.document_count = len(documents)
            report.chunk_count = len(chunks)
            return report

        indexed_chunks = self._stored_chunk_count()
        report = self._base_report(
            STATUS_INDEXED,
            (
                f"Indexed {written} chunk(s) from {dataset.count} standard(s) into "
                f"the '{vector_store}' store ({indexed_chunks} chunk(s) stored in "
                "total)."
            ),
            embedding=embedding,
            vector_store=vector_store,
        )
        report.dataset_version = dataset.version
        report.standards_count = dataset.count
        report.document_count = len(documents)
        report.chunk_count = len(chunks)
        report.written = written
        report.indexed_chunks = indexed_chunks
        return report


def get_standards_indexing_service() -> StandardsIndexingService:
    """Return an indexing service wired to the current configuration."""
    return StandardsIndexingService()

