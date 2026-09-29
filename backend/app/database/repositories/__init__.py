"""Repository implementations.

``get_standard_repository`` is the single entry point used by services; it
currently returns the in-memory implementation.
"""

from functools import lru_cache

from app.database.repositories.base import StandardRepository
from app.database.repositories.in_memory import InMemoryStandardRepository

__all__ = [
    "InMemoryStandardRepository",
    "StandardRepository",
    "get_standard_repository",
]


@lru_cache
def get_standard_repository() -> StandardRepository:
    """Return the process-wide standards repository."""
    return InMemoryStandardRepository()
