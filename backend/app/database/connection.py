"""SQL database integration point (PostgreSQL later).

No SQL driver or ORM is installed yet on purpose: the project must remain
dependency-light until persistence is actually needed. This module defines the
single place where the engine will be created, so adding PostgreSQL later is a
localised change:

1. add ``sqlalchemy`` + ``psycopg`` to ``requirements.txt``,
2. implement :meth:`DatabaseManager.create_engine` using ``DATABASE_URL``,
3. provide a session dependency and swap the repository implementations.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Owns the (future) SQL engine lifecycle."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url

    @property
    def is_configured(self) -> bool:
        """True once ``DATABASE_URL`` is provided."""
        return bool(self._url)

    @property
    def describe(self) -> str:
        return "postgresql-ready (not enabled)" if not self.is_configured else "configured"

    def create_engine(self) -> None:
        """Create the SQL engine. Not implemented until persistence is needed."""
        if not self.is_configured:
            raise ConfigurationError(
                "DATABASE_URL is not set. SQL persistence is optional and not "
                "enabled in this milestone."
            )
        raise NotImplementedError(
            "SQL engine creation will be implemented when PostgreSQL is added."
        )

    def dispose(self) -> None:
        """Release engine resources (no-op until an engine exists)."""
        return None


@lru_cache
def get_database_manager() -> DatabaseManager:
    """Return the process-wide database manager."""
    settings = get_settings()
    manager = DatabaseManager(settings.database_url)
    logger.info("SQL database: %s", manager.describe)
    return manager
