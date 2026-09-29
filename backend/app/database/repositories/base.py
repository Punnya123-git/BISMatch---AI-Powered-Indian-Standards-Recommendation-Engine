"""Repository interfaces.

Services depend on these abstractions only, so the storage backend (in-memory
today, PostgreSQL later) can be replaced without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.models.standard import StandardEntity


class StandardRepository(ABC):
    """Read/write access to the Indian Standards catalogue."""

    @abstractmethod
    def add_many(self, standards: Sequence[StandardEntity]) -> int:
        """Insert or update standards. Returns the number written."""

    @abstractmethod
    def get_by_code(self, code: str) -> StandardEntity | None:
        """Fetch a single standard by its canonical code."""

    @abstractmethod
    def list_all(self, *, limit: int | None = None, offset: int = 0) -> list[StandardEntity]:
        """List stored standards."""

    @abstractmethod
    def keyword_search(self, query: str, *, limit: int = 10) -> list[StandardEntity]:
        """Simple lexical search (used as a baseline and for exact codes)."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored standards."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every stored standard."""

    @property
    def is_loaded(self) -> bool:
        """True when at least one standard is available."""
        return self.count() > 0
