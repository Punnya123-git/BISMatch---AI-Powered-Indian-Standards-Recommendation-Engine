"""Domain model exports.

Models are deliberately dataclasses (no ORM dependency yet). When PostgreSQL is
introduced, only the implementations change, not the import paths used by the
services layer.
"""

from app.models.base import Entity, utc_now
from app.models.document import DocumentEntity
from app.models.standard import StandardEntity, StandardVersionEntity

__all__ = [
    "DocumentEntity",
    "Entity",
    "StandardEntity",
    "StandardVersionEntity",
    "utc_now",
]
