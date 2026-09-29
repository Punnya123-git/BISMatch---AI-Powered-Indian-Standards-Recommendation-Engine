"""Stored-document entity (PostgreSQL-ready field layout)."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.base import Entity


@dataclass(slots=True, kw_only=True)
class DocumentEntity(Entity):
    """An uploaded procurement/tender document."""

    filename: str
    content_type: str | None = None
    size_bytes: int = 0
    stored_path: str = ""
    page_count: int = 0
    character_count: int = 0
    word_count: int = 0
