"""Embedding provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    #: Registry key used in settings, e.g. ``"openai_compatible"``.
    provider_name: str = "unconfigured"

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True when the provider has everything it needs to run."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Embedding vector length, when known."""

    @property
    def model(self) -> str | None:
        return None

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    def describe(self) -> str:
        """Short, human readable description used by /api/health."""
        state = "configured" if self.is_configured else "not configured"
        model = f", model={self.model}" if self.model else ""
        return f"{self.provider_name} ({state}{model})"
