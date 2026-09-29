"""Plain data structures produced by the document processing package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one page of a source document."""

    page_number: int
    text: str

    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class ExtractedDocument:
    """A processed document: ordered pages plus non-fatal warnings."""

    filename: str
    pages: list[ExtractedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[ExtractedPage]:
        return iter(self.pages)

    @property
    def text(self) -> str:
        """Cleaned document text (pages separated by a blank line)."""
        return "\n\n".join(page.text for page in self.pages if page.text).strip()

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def is_empty(self) -> bool:
        return not self.text
