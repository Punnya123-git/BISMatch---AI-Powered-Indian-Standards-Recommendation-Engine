"""Extractor interface implemented by every supported file format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.document_processing.entities import ExtractedPage


class DocumentExtractor(ABC):
    """Extracts ordered, page-aware text from a file on disk."""

    #: File extensions this extractor can handle (lower case, with dot).
    supported_extensions: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def extract(self, path: Path) -> list[ExtractedPage]:
        """Return the extracted pages. Raises ``DocumentProcessingError``."""
        raise NotImplementedError
