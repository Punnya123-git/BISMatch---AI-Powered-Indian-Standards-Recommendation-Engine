"""Provider factory: maps ``EMBEDDING_PROVIDER`` to an implementation."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.providers.local_onnx import (
    LocalOnnxMiniLMEmbeddingProvider,
)
from app.rag.embeddings.providers.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.rag.embeddings.providers.unconfigured import UnconfiguredEmbeddingProvider

logger = get_logger(__name__)

#: Registry of available embedding providers. Additional providers can be added
#: here or at runtime with :func:`register_embedding_provider`.
_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    # Runs a real sentence model on this machine: no API key, no network calls.
    LocalOnnxMiniLMEmbeddingProvider.provider_name: LocalOnnxMiniLMEmbeddingProvider,
    # Calls any OpenAI-style /embeddings endpoint (needs EMBEDDING_API_KEY).
    OpenAICompatibleEmbeddingProvider.provider_name: OpenAICompatibleEmbeddingProvider,
}


def register_embedding_provider(name: str, provider_cls: type[EmbeddingProvider]) -> None:
    """Register an embedding provider class so it can be selected by name."""
    _PROVIDERS[name.lower()] = provider_cls


def available_embedding_providers() -> list[str]:
    """Names of the embedding providers compiled into this build."""
    return sorted(_PROVIDERS)


def _build_provider(settings: Settings) -> EmbeddingProvider:
    name = (settings.embedding_provider or "unconfigured").lower()
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        return UnconfiguredEmbeddingProvider(
            f"Embedding provider '{name}' is not implemented in this build."
        )
    return provider_cls(settings)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (cached per process)."""
    provider = _build_provider(get_settings())
    logger.info("Embedding provider: %s", provider.describe())
    return provider


def reset_embedding_provider_cache() -> None:
    """Clear the cache (used by tests and after configuration changes)."""
    get_embedding_provider.cache_clear()
