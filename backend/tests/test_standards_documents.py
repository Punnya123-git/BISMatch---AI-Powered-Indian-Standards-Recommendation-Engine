"""Searchable document conversion tests (dataset record -> embeddable text)."""

from __future__ import annotations

from pathlib import Path

from app.standards.dataset_loader import StandardsDatasetLoader
from app.standards.dataset_schema import validate_standards_dataset
from app.standards.documents import (
    build_standard_document,
    build_standard_documents,
    standard_chunk_metadata,
    standard_source_id,
    standard_to_searchable_text,
)

_OPTIONAL_LABELS = (
    "Title:",
    "Scope:",
    "Status:",
    "Revision:",
    "Amendments:",
    "Related Standards:",
    "References:",
    "Certification:",
)


def test_document_only_renders_fields_present_in_the_dataset(
    dataset_payload_factory,
) -> None:
    """With everything except the designation unverified, nothing extra appears."""
    record = validate_standards_dataset(dataset_payload_factory()).records[0]

    assert standard_to_searchable_text(record).splitlines() == [
        "Standard Number: IS 9999:2000",
        "Product Category: rotating electrical machines",
        "Year: 2000",
        "Source Organization: Bureau of Indian Standards",
    ]


def test_document_renders_verified_values(
    dataset_payload_factory, standard_record_factory
) -> None:
    record = validate_standards_dataset(
        dataset_payload_factory(
            standard_record_factory(
                title="Test-only title",
                scope="Test-only scope",
                status="current",
                revision="Second Revision",
                amendments=["Amendment No. 1", "Amendment No. 2"],
                related_standards=["IS 1111:2001", "IS 2222:2002"],
                references=[
                    {
                        "code": "IS 5678:1990",
                        "title": "Test reference",
                        "relation": "normative",
                    }
                ],
                certification={
                    "scheme": "BIS certification scheme",
                    "marking": "ISI Mark",
                },
            )
        )
    ).records[0]

    text = standard_to_searchable_text(record)
    assert "Title: Test-only title" in text
    assert "Scope: Test-only scope" in text
    assert "Status: current" in text
    assert "Revision: Second Revision" in text
    assert "Amendments: Amendment No. 1; Amendment No. 2" in text
    assert "Related Standards: IS 1111:2001, IS 2222:2002" in text
    assert "References: IS 5678:1990 (Test reference) [normative]" in text
    assert (
        "Certification: scheme = BIS certification scheme | marking = ISI Mark" in text
    )


def test_shipped_style_record_has_no_invented_lines(
    dataset_payload_factory, standard_record_factory
) -> None:
    record = validate_standards_dataset(
        dataset_payload_factory(standard_record_factory(standard_number="IS 12615:2018"))
    ).records[0]
    text = standard_to_searchable_text(record)
    assert not any(label in text for label in _OPTIONAL_LABELS)


def test_chunk_metadata_is_vector_store_friendly(dataset_payload_factory) -> None:
    record = validate_standards_dataset(dataset_payload_factory()).records[0]
    metadata = standard_chunk_metadata(record, dataset_version="9.9.9")

    assert metadata == {
        "record_type": "standard",
        "standard_number": "IS 9999:2000",
        "standard_code": "IS 9999:2000",
        "product_category": "rotating electrical machines",
        "year": 2000,
        "source": "Bureau of Indian Standards",
        "dataset_version": "9.9.9",
    }
    assert all(value is not None for value in metadata.values())


def test_metadata_omits_values_the_dataset_does_not_hold() -> None:
    payload = {
        "standards": [
            {
                "standard_number": "IS 9999",
                "product_category": "rotating electrical machines",
                "source": {"organization": "Bureau of Indian Standards"},
            }
        ]
    }
    record = validate_standards_dataset(payload).records[0]
    metadata = standard_chunk_metadata(record, dataset_version=None)

    assert "title" not in metadata
    assert "year" not in metadata
    assert "dataset_version" not in metadata


def test_source_id_is_safe_for_a_vector_store() -> None:
    assert standard_source_id("IS/IEC 60034-2-1:2024") == "standard::IS-IEC-60034-2-1-2024"
    source_id = standard_source_id("IS 12615:2018")
    assert source_id == "standard::IS-12615-2018"
    assert " " not in source_id and "/" not in source_id


def test_build_documents_covers_every_shipped_record(shipped_dataset_path: Path) -> None:
    dataset = StandardsDatasetLoader(path=shipped_dataset_path).load()
    documents = build_standard_documents(dataset)

    assert len(documents) == dataset.count
    assert all(document.text.startswith("Standard Number: ") for document in documents)
    assert all(
        document.metadata["dataset_version"] == dataset.version
        for document in documents
    )
    assert all(document.source_id.startswith("standard::") for document in documents)

    single = build_standard_document(dataset.records[0], dataset_version=dataset.version)
    assert single.standard_number == dataset.records[0].standard_number
    assert single.text == standard_to_searchable_text(dataset.records[0])
