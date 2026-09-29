"""Placeholder embedding provider used until a real backend is chosen.

It never returns fabricated vectors: calling it raises a configuration error so
the system fails loudly instead of indexing meaningless data.
"""

from __future__ import annotations

from typing import Sequence

from app.core.exceptions import ProviderNotConfiguredError
from app.rag.embeddings.base import EmbeddingProvider


class UnconfiguredEmbeddingProvider(EmbeddingProvider):
    """Reports that no embedding provider has been selected yet."""

    provider_name = "unconfigured"

    def __init__(self, reason: str = "No embedding provider has been configured.") -> None:
        self._reason = reason

    @property
    def is_configured(self) -> bool:
        return False

    @property
    def dimension(self) -> int | None:
        return None

    @property
    def reason(self) -> str:
        """Why nothing can be embedded (same attribute name as real providers)."""
        return self._reason

    def _fail(self) -> ProviderNotConfiguredError:
        return ProviderNotConfiguredError(
            f"{self.reason} Set EMBEDDING_PROVIDER, EMBEDDING_MODEL and "
            "EMBEDDING_API_KEY in the backend environment before indexing or "
            "searching standards."
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise self._fail()

    def embed_query(self, text: str) -> list[float]:
        raise self._fail()
