"""Stored-standard entity (PostgreSQL-ready field layout)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.base import Entity


@dataclass(slots=True, kw_only=True)
class StandardVersionEntity:
    """Version/amendment information attached to a standard."""

    year: int | None = None
    revision: str | None = None
    amendments: list[str] = field(default_factory=list)
    status: str | None = None


@dataclass(slots=True, kw_only=True)
class StandardEntity(Entity):
    """A record of an Indian Standard in the knowledge base."""

    code: str
    title: str | None = None
    summary: str | None = None
    category: str | None = None
    keywords: list[str] = field(default_factory=list)
    scope: str | None = None
    version: StandardVersionEntity | None = None
    # JSON columns in the future PostgreSQL schema.
    certification: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    source: dict[str, Any] | None = None
