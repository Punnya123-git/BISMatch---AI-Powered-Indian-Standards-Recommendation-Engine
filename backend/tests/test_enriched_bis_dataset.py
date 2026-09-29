"""Guards for the enriched BIS catalogue.

``data/standards/standards_enriched_bis_verified.json`` uses the *draft* envelope
(``dataset_version`` / ``description`` / ``source_note`` instead of a ``dataset``
object) and writes ``references`` as plain designation strings. These tests pin
down two things:

* the loader really reads those 20 verified records - including the BIS text and
  the cross-references - instead of silently dropping them;
* the leniency stays cosmetic: only strings are normalised, and anything the
  draft does not say is still empty (and unknown keys are still rejected).

Fast by design: JSON + schema + text rendering only, no model and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.exceptions import DatasetValidationError
from app.standards.dataset_loader import StandardsDatasetLoader
from app.standards.dataset_schema import (
    CertificationRecord,
    StandardReferenceRecord,
    designation_year,
    validate_standards_dataset,
)
from app.standards.documents import build_standard_documents

#: The designations carried by the enriched draft (rotating electrical machines).
EXPECTED_DESIGNATIONS = {
    "IS/IEC 60034-1:2022",
    "IS 12615:2018",
    "IS 996:2009",
    "IS 9283:2024",
    "IS 7538:1996",
    "IS 14582:2021",
    "IS 14578:1999",
    "IS 8151:2024",
    "IS 18073:2023",
    "IS/IEC 60034-2-1:2024",
    "IS 4029:2010",
    "IS 9320:2025",
    "IS 7572:1974",
    "IS 12075:2024",
    "IS 12065:2025",
    "IS 1231:2019",
    "IS 2223:1983",
    "IS 6362:1995",
    "IS/IEC 60034-5:2020",
    "IS 13529:2021",
}

#: Every designation whose BIS record names an adopted IEC counterpart, with the
#: exact strings the draft lists for it.
EXPECTED_REFERENCES = {
    "IS/IEC 60034-1:2022": ["IEC 60034-1:2022"],
    "IS 12615:2018": ["IEC 60034-30-1:2014 (modified)"],
    "IS/IEC 60034-2-1:2024": ["IEC 60034-2-1:2024"],
    "IS 12075:2024": ["IEC 60034-14:2018 (modified)"],
    "IS 12065:2025": ["IEC 60034-9:2021 (modified)"],
    "IS 6362:1995": ["IEC 60034-6"],
    "IS/IEC 60034-5:2020": ["IEC 60034-5:2020"],
    "IS 13529:2021": ["IEC 60034-26:2006 (modified)"],
}

#: The one record whose BIS material mentions certification, plus that statement.
EXPECTED_CERTIFICATION = {
    "IS 9283:2024": "BIS product-manual material is listed for IS 9283:2024.",
}


@pytest.fixture(scope="session")
def enriched_dataset_path() -> Path:
    """Path to the enriched BIS draft that ships with the repository."""
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "standards"
        / "standards_enriched_bis_verified.json"
    )


@pytest.fixture(scope="session")
def enriched_payload(enriched_dataset_path: Path) -> dict:
    return json.loads(enriched_dataset_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def enriched_dataset(enriched_payload, enriched_dataset_path):
    return validate_standards_dataset(enriched_payload, path=str(enriched_dataset_path))


def _by_designation(dataset) -> dict[str, object]:
    return {record.standard_number: record for record in dataset.records}


def test_enriched_dataset_validates_and_describes_itself(enriched_dataset) -> None:
    assert enriched_dataset.count == 20
    assert enriched_dataset.version == "1.1.0-bis-enriched-draft"
    # The draft's own description/source note are preserved, not dropped.
    assert any("PS 108" in note for note in enriched_dataset.metadata.notes)
    # Draft level fields that the file does not state stay empty.
    assert enriched_dataset.metadata.organization is None


def test_every_enriched_record_carries_its_bis_text(enriched_dataset) -> None:
    """The point of the enriched file: the BIS title, scope and status are there."""
    for record in enriched_dataset.records:
        assert record.title and record.title.strip()
        assert record.scope and record.scope.strip()
        assert record.status and record.status.strip()
        assert record.product_category == "rotating electrical machines"
        assert record.source.organization == "Bureau of Indian Standards"
        assert record.year == designation_year(record.standard_number)


def test_cross_references_are_read_and_normalised(enriched_dataset) -> None:
    records = _by_designation(enriched_dataset)
    for designation, expected in EXPECTED_REFERENCES.items():
        references = records[designation].references
        assert [reference.code for reference in references] == expected
        for reference in references:
            # Only the designation exists in the draft; nothing else is invented.
            assert isinstance(reference, StandardReferenceRecord)
            assert reference.title is None
            assert reference.relation is None

    for designation, record in records.items():
        if designation not in EXPECTED_REFERENCES:
            assert record.references == []


def test_certification_statement_is_kept_as_a_note(enriched_dataset) -> None:
    records = _by_designation(enriched_dataset)
    for designation, statement in EXPECTED_CERTIFICATION.items():
        certification = records[designation].certification
        assert isinstance(certification, CertificationRecord)
        assert certification.notes == statement
        assert certification.scheme is None
        assert certification.marking is None
    for designation, record in records.items():
        if designation not in EXPECTED_CERTIFICATION:
            assert record.certification is None


def test_loader_reads_the_enriched_file_into_entities(
    enriched_dataset_path: Path,
) -> None:
    """The end-to-end path used by the API and the indexer really works."""
    loader = StandardsDatasetLoader(path=enriched_dataset_path)
    dataset, entities = loader.load_entities()
    assert dataset.count == len(entities) == 20
    by_code = {entity.code: entity for entity in entities}
    motor = by_code["IS/IEC 60034-1:2022"]
    assert motor.title and motor.scope
    assert motor.version is not None and motor.version.status
    assert motor.references == [{"code": "IEC 60034-1:2022", "title": None, "relation": None}]
    assert by_code["IS 9283:2024"].certification == [
        {
            "scheme": None,
            "marking": None,
            "notes": EXPECTED_CERTIFICATION["IS 9283:2024"],
        }
    ]


def test_searchable_text_renders_the_new_fields(enriched_dataset) -> None:
    documents = {
        document.standard_number: document
        for document in build_standard_documents(enriched_dataset)
    }
    assert len(documents) == 20
    text = documents["IS 12615:2018"].text
    assert text.startswith("Standard Number: IS 12615:2018")
    assert "Title: Line-Operated Three-Phase" in text
    assert "Scope:" in text
    assert "References: IEC 60034-30-1:2014 (modified)" in text
    # A title/scope from BIS is never presented as a requirement.
    assert "Requirement:" not in text
    assert "Certification: notes =" in documents["IS 9283:2024"].text


def test_draft_envelope_maps_onto_dataset_metadata(standard_record_factory) -> None:
    dataset = validate_standards_dataset(
        {
            "schema_version": "1.0",
            "dataset_version": "2.0.0-draft",
            "description": "Draft catalogue.",
            "source_note": "Checked against BIS.",
            "standards": [standard_record_factory()],
        },
        path="draft.json",
    )
    assert dataset.version == "2.0.0-draft"
    assert dataset.metadata.notes == ["Draft catalogue.", "Checked against BIS."]


def test_draft_leniency_does_not_accept_unknown_keys_or_bad_shapes(
    standard_record_factory,
) -> None:
    """Relaxing the envelope must not turn the validator into a rubber stamp."""
    with pytest.raises(DatasetValidationError):
        validate_standards_dataset(
            {"schema_version": "1.0", "standard": [standard_record_factory()]},
            path="typo.json",
        )
    with pytest.raises(DatasetValidationError):
        validate_standards_dataset(
            {
                "schema_version": "1.0",
                "standards": [standard_record_factory(references="IEC 60034-1")],
            },
            path="bad-references.json",
        )
    with pytest.raises(DatasetValidationError):
        validate_standards_dataset(
            {"schema_version": "1.0", "standards": [standard_record_factory(certification="   ")]},
            path="blank-certification.json",
        )
