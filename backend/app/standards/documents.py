"""Convert validated standards records into searchable, embeddable documents.

Only fields that exist in the dataset are rendered: a ``null`` field produces no
line at all, so the text can never contain information the dataset does not hold
(no title, scope, amendment or relation is ever written from memory).

This module is the single place that decides how a standard becomes text, which
keeps the vector store contents explainable: every chunk is traceable back to
columns of ``data/standards/standards.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.standards.dataset_schema import StandardDatasetRecord, StandardsDataset

#: Word used to mark standards records in the vector store (vs. uploaded docs).
RECORD_TYPE = "standard"

_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9]+")


@dataclass(slots=True)
class StandardDocument:
    """A standard rendered as text plus the metadata stored with its vectors."""

    standard_number: str
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def slugify_designation(standard_number: str) -> str:
    """Vector-store safe form of a designation.

    ``IS/IEC 60034-1:2022`` -> ``IS-IEC-60034-1-2022``. Only used inside ids;
    the real designation is always kept in the metadata.
    """
    slug = _SLUG_PATTERN.sub("-", standard_number or "").strip("-").upper()
    return slug or "STANDARD"


def standard_source_id(standard_number: str) -> str:
    """Stable chunk-source id for a standard, e.g. ``standard::IS-12615-2018``."""
    return f"{RECORD_TYPE}::{slugify_designation(standard_number)}"


def _line(label: str, value: Any) -> str | None:
    """Render ``Label: value`` only when the value is actually present."""
    if value is None:
        return None
    text = str(value).strip()
    return f"{label}: {text}" if text else None


def _reference_line(reference: Any) -> str:
    parts = [reference.code.strip()]
    if reference.title:
        parts.append(f"({reference.title.strip()})")
    if reference.relation:
        parts.append(f"[{reference.relation.strip()}]")
    return " ".join(parts)


def _certification_line(certification: Any) -> str | None:
    parts = [
        f"{name} = {value.strip()}"
        for name, value in (
            ("scheme", certification.scheme),
            ("marking", certification.marking),
            ("notes", certification.notes),
        )
        if isinstance(value, str) and value.strip()
    ]
    return "Certification: " + " | ".join(parts) if parts else None


def standard_to_searchable_text(record: StandardDatasetRecord) -> str:
    """Build the search/embedding document for one standard record."""
    lines: list[str | None] = [
        _line("Standard Number", record.standard_number),
        _line("Title", record.title),
        _line("Scope", record.scope),
        _line("Product Category", record.product_category),
        _line("Status", record.status),
        _line("Revision", record.revision),
        _line("Year", record.year),
        _line("Amendments", "; ".join(record.amendments)),
        _line("Related Standards", ", ".join(record.related_standards)),
        _line(
            "References",
            "; ".join(_reference_line(reference) for reference in record.references),
        ),
        _certification_line(record.certification) if record.certification else None,
        _line(
            "Source Organization",
            record.source.organization if record.source else None,
        ),
    ]
    return "\n".join(line for line in lines if line)


def standard_chunk_metadata(
    record: StandardDatasetRecord, *, dataset_version: str | None = None
) -> dict[str, Any]:
    """Flat, vector-store friendly metadata for every chunk of a standard."""
    metadata: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "standard_number": record.standard_number,
        # Kept as a separate key because evidence reporting reads `standard_code`.
        "standard_code": record.standard_number,
        "product_category": record.product_category,
    }
    if record.title:
        metadata["title"] = record.title
    if record.year is not None:
        metadata["year"] = record.year
    if record.source and record.source.organization:
        metadata["source"] = record.source.organization
    if dataset_version:
        metadata["dataset_version"] = dataset_version
    return metadata


def build_standard_document(
    record: StandardDatasetRecord, *, dataset_version: str | None = None
) -> StandardDocument:
    """Convert one record into a :class:`StandardDocument`."""
    return StandardDocument(
        standard_number=record.standard_number,
        source_id=standard_source_id(record.standard_number),
        text=standard_to_searchable_text(record),
        metadata=standard_chunk_metadata(record, dataset_version=dataset_version),
    )


def build_standard_documents(dataset: StandardsDataset) -> list[StandardDocument]:
    """Convert a validated dataset into documents, skipping empty renderings."""
    documents = [
        build_standard_document(record, dataset_version=dataset.version)
        for record in dataset.records
    ]
    return [document for document in documents if document.text.strip()]
