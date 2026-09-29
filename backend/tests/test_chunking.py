"""Unit tests for the chunking utility."""

from __future__ import annotations

import pytest

from app.core.exceptions import ConfigurationError
from app.document_processing.entities import ExtractedPage
from app.rag.chunking import chunk_pages, chunk_text


def test_chunk_text_respects_max_size() -> None:
    text = "\n\n".join(f"Paragraph {index} " + "x" * 120 for index in range(20))
    chunks = chunk_text(text, source_id="doc-1", chunk_size=300, chunk_overlap=50)

    assert chunks
    for chunk in chunks:
        assert len(chunk.text) <= 300 + 50  # size plus overlap prefix
        assert chunk.source_id == "doc-1"


def test_chunk_ids_and_indexes_are_unique() -> None:
    text = "\n\n".join("Sentence " * 40 for _ in range(6))
    chunks = chunk_text(text, source_id="doc-2", chunk_size=200, chunk_overlap=20)

    ids = {chunk.chunk_id for chunk in chunks}
    indexes = [chunk.chunk_index for chunk in chunks]
    assert len(ids) == len(chunks)
    assert indexes == list(range(len(chunks)))


def test_chunk_pages_keeps_page_numbers() -> None:
    pages = [
        ExtractedPage(page_number=1, text="First page content " * 20),
        ExtractedPage(page_number=2, text="Second page content " * 20),
    ]
    chunks = chunk_pages(pages, source_id="doc-3", chunk_size=150, chunk_overlap=10)

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert chunks[-1].page_number == 2


def test_chunk_metadata_is_flat_for_vector_stores() -> None:
    chunks = chunk_text(
        "Some text",
        source_id="doc-4",
        page_number=3,
        metadata={"standard_code": "IS TEST", "empty": None},
    )
    metadata = chunks[0].vector_metadata()

    assert metadata["page_number"] == 3
    assert metadata["standard_code"] == "IS TEST"
    assert "empty" not in metadata


def test_invalid_overlap_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        chunk_text("text", source_id="doc-5", chunk_size=100, chunk_overlap=100)
