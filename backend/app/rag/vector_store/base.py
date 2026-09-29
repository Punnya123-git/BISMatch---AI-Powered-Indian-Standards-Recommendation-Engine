"""Vector store abstraction (ChromaDB is the first implementation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(slots=True)
class VectorRecord:
    """A chunk plus its embedding, ready to be stored."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorQueryResult:
    """One similarity-search hit."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Interface for every vector database backend."""

    provider_name: str = "base"

    @abstractmethod
    def add(self, records: Sequence[VectorRecord]) -> int:
        """Insert or update records. Returns the number written."""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``top_k`` most similar records."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored records."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Delete records by id. Returns the number removed."""

    @abstractmethod
    def reset(self) -> None:
        """Remove every record from the collection."""

    def describe(self) -> str:
        """Short, human readable description used by /api/health."""
        return f"{self.provider_name} ({self.count()} records)"
