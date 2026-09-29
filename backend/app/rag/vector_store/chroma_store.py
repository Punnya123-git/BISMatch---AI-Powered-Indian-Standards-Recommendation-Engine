"""ChromaDB vector store implementation.

Embeddings are always supplied explicitly (produced by the configured
:class:`~app.rag.embeddings.base.EmbeddingProvider`), so Chroma's bundled
default embedding model is never downloaded or executed. ``chromadb`` is an
optional dependency: it is imported lazily and a missing install is reported as
a :class:`~app.core.exceptions.VectorStoreError` instead of crashing the app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.rag.vector_store.base import VectorQueryResult, VectorRecord, VectorStore

logger = get_logger(__name__)


class _NoEmbeddingFunction:
    """Sentinel used when a no-op Chroma embedding function cannot be built."""


def _no_op_embedding_function() -> Any:
    """A Chroma ``EmbeddingFunction`` that refuses to run.

    Prevents Chroma from silently falling back to its downloadable default
    model. If the installed Chroma version exposes a different interface we
    return the sentinel and rely on explicitly supplied embeddings only.
    """
    try:
        from chromadb.api.types import EmbeddingFunction

        class ExplicitOnlyEmbeddingFunction(EmbeddingFunction):  # type: ignore[misc]
            def __call__(self, input: Any) -> Any:  # noqa: A002 - Chroma's name
                raise VectorStoreError(
                    "Chroma was asked to compute embeddings. Embeddings must be "
                    "produced by the configured embedding provider."
                )

            def name(self) -> str:  # required by newer Chroma versions
                return "explicit-only"

        return ExplicitOnlyEmbeddingFunction()
    except Exception:  # noqa: BLE001 - any version mismatch falls back safely
        return _NoEmbeddingFunction()


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB collection using cosine similarity."""

    provider_name = "chroma"

    def __init__(self, *, collection_name: str, persist_directory: Path) -> None:
        self._collection_name = collection_name
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = self._create_client()
        self._collection = self._create_collection()

    # --- setup -----------------------------------------------------------
    def _create_client(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError(
                "The vector store requires the optional 'chromadb' package. "
                "Install it with: pip install -r requirements-rag.txt"
            ) from exc

        try:
            return chromadb.PersistentClient(path=str(self._persist_directory))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not open the vector store: {exc}") from exc

    def _create_collection(self) -> Any:
        metadata = {"hnsw:space": "cosine"}
        try:
            return self._client.get_or_create_collection(
                name=self._collection_name,
                metadata=metadata,
                embedding_function=_no_op_embedding_function(),
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - older/newer Chroma signatures
            logger.debug("Retrying collection creation without EF: %s", exc)
            try:
                return self._client.get_or_create_collection(
                    name=self._collection_name, metadata=metadata
                )
            except Exception as inner:  # noqa: BLE001
                raise VectorStoreError(
                    f"Could not open collection '{self._collection_name}': {inner}"
                ) from inner

    @property
    def collection_name(self) -> str:
        return self._collection_name

    # --- VectorStore API -------------------------------------------------
    def add(self, records: Sequence[VectorRecord]) -> int:
        records = list(records)
        if not records:
            return 0
        try:
            self._collection.upsert(
                ids=[record.id for record in records],
                embeddings=[record.embedding for record in records],
                documents=[record.text for record in records],
                metadatas=[record.metadata or None for record in records],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not write to the vector store: {exc}") from exc
        return len(records)

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorQueryResult]:
        if top_k <= 0:
            return []
        stored = self.count()
        if stored == 0:
            return []
        try:
            raw = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=min(top_k, stored),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[VectorQueryResult] = []
        for index, record_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            score = 1.0 - float(distance) if distance is not None else 0.0
            results.append(
                VectorQueryResult(
                    id=record_id,
                    text=documents[index] if index < len(documents) else "",
                    score=max(0.0, min(1.0, score)),
                    metadata=metadatas[index] if index < len(metadatas) else {},
                )
            )
        return results

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not read the vector store: {exc}") from exc

    def delete(self, ids: Sequence[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not delete vectors: {exc}") from exc
        return len(ids)

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            logger.debug("Collection '%s' was already absent.", self._collection_name)
        self._collection = self._create_collection()
