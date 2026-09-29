"""Vector store factory: maps ``VECTOR_STORE_PROVIDER`` to an implementation."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.rag.vector_store.base import VectorStore
from app.rag.vector_store.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def _build_vector_store() -> VectorStore:
    settings = get_settings()
    provider = (settings.vector_store_provider or "chroma").lower()
    if provider != "chroma":
        raise ConfigurationError(
            f"Vector store provider '{provider}' is not implemented. "
            "Supported providers: chroma."
        )
    return ChromaVectorStore(
        collection_name=settings.vector_collection_name,
        persist_directory=settings.resolved_vector_db_path,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the configured vector store (cached per process)."""
    store = _build_vector_store()
    logger.info("Vector store: %s", store.provider_name)
    return store


def reset_vector_store_cache() -> None:
    """Clear the cache (used by tests and after configuration changes)."""
    get_vector_store.cache_clear()
