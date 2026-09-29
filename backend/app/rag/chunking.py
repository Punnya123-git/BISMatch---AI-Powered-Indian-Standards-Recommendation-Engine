"""Deterministic text chunking for retrieval.

The chunker is pure Python and dependency free. It packs paragraphs up to
``chunk_size`` characters and repeats the tail of the previous chunk (up to
``chunk_overlap``) so sentences are not cut off from their context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.core.exceptions import ConfigurationError
from app.document_processing.entities import ExtractedPage

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+")


@dataclass(slots=True)
class TextChunk:
    """A retrievable unit of text with provenance metadata."""

    chunk_id: str
    text: str
    source_id: str
    chunk_index: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def vector_metadata(self) -> dict[str, Any]:
        """Flat metadata suitable for a vector store (no ``None`` values)."""
        metadata: dict[str, Any] = {
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
        }
        if self.page_number is not None:
            metadata["page_number"] = self.page_number
        metadata.update({k: v for k, v in self.metadata.items() if v is not None})
        return metadata


def _chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}::{chunk_index}::{digest}"


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ConfigurationError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ConfigurationError("chunk_overlap must be >= 0 and smaller than chunk_size.")


def _split_oversized(block: str, chunk_size: int) -> list[str]:
    """Split a single oversized paragraph into sentence/word sized pieces."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(block):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(sentence) <= chunk_size:
            current = sentence
            continue
        words = sentence.split(" ")
        buffer = ""
        for word in words:
            candidate = f"{buffer} {word}".strip() if buffer else word
            if len(candidate) > chunk_size and buffer:
                pieces.append(buffer)
                buffer = word
            else:
                buffer = candidate
        current = buffer
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str,
    *,
    source_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    page_number: int | None = None,
    start_index: int = 0,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Split ``text`` into overlapping :class:`TextChunk` objects."""
    _validate(chunk_size, chunk_overlap)
    base_metadata = dict(metadata or {})

    blocks: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            blocks.append(paragraph)
        else:
            blocks.extend(_split_oversized(paragraph, chunk_size))

    chunks: list[TextChunk] = []
    buffer = ""
    index = start_index
    for block in blocks:
        candidate = f"{buffer}\n\n{block}" if buffer else block
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(
                TextChunk(
                    chunk_id=_chunk_id(source_id, index, buffer),
                    text=buffer,
                    source_id=source_id,
                    chunk_index=index,
                    page_number=page_number,
                    metadata=dict(base_metadata),
                )
            )
            index += 1
            buffer = (
                f"{buffer[-chunk_overlap:]}\n\n{block}" if chunk_overlap else block
            )
        else:
            buffer = block
    if buffer:
        chunks.append(
            TextChunk(
                chunk_id=_chunk_id(source_id, index, buffer),
                text=buffer,
                source_id=source_id,
                chunk_index=index,
                page_number=page_number,
                metadata=dict(base_metadata),
            )
        )
    return chunks


def chunk_pages(
    pages: Iterable[ExtractedPage],
    *,
    source_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Chunk page-aware text, keeping each page's number in the metadata."""
    chunks: list[TextChunk] = []
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            source_id=source_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page.page_number,
            start_index=len(chunks),
            metadata=metadata,
        )
        chunks.extend(page_chunks)
    return chunks


def chunks_to_texts(chunks: Sequence[TextChunk]) -> list[str]:
    """Convenience helper for embedding providers."""
    return [chunk.text for chunk in chunks]
