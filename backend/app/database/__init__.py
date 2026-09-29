"""Persistence layer.

Today the project stores standards in memory (:mod:`app.database.repositories`)
and vectors in ChromaDB. When PostgreSQL is introduced, an engine/session
factory lands in :mod:`app.database.connection` and the repository
implementations are swapped behind the same interfaces - no service or route
changes required.
"""

from app.database.repositories.base import StandardRepository
from app.database.repositories.in_memory import InMemoryStandardRepository

__all__ = ["InMemoryStandardRepository", "StandardRepository"]
