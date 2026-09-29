"""Domain models.

These are the in-process representations of persisted entities. They are plain
dataclasses on purpose: the project does not depend on an ORM yet, but the
fields mirror the future PostgreSQL schema so migrating later is mechanical
(replace the dataclass with a SQLAlchemy declarative model and keep the field
names).

Note: this module never contains catalogue data. Indian Standards records must
come from a real, verifiable dataset loaded at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    """Timezone-aware timestamp helper used as the default for ``created_at``."""
    return datetime.now(timezone.utc)


@dataclass(slots=True, kw_only=True)
class Entity:
    """Common identity/timestamp fields shared by stored entities."""

    id: str
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
