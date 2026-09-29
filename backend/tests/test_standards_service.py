"""Unit tests for the standards repository and service (no catalogue data)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.database.repositories.in_memory import InMemoryStandardRepository
from app.models.standard import StandardEntity, StandardVersionEntity
from app.services.standards_service import StandardsService
from app.standards.dataset_loader import StandardsDatasetLoader


def _example_entity() -> StandardEntity:
    """A test-only fixture record; it is not shipped catalogue data."""
    return StandardEntity(
        id="test-1",
        code="IS 00000 : 2000",
        title="Test-only placeholder record for repository behaviour",
        summary="Used exclusively by unit tests.",
        keywords=["testing", "repository"],
        version=StandardVersionEntity(year=2000),
    )


def test_repository_starts_empty() -> None:
    repository = InMemoryStandardRepository()
    assert repository.count() == 0
    assert repository.is_loaded is False


def test_repository_add_and_lookup() -> None:
    repository = InMemoryStandardRepository()
    assert repository.add_many([_example_entity()]) == 1

    assert repository.is_loaded is True
    found = repository.get_by_code("is 00000 : 2000")
    assert found is not None
    assert found.title.startswith("Test-only")


def test_repository_keyword_search_finds_exact_code() -> None:
    repository = InMemoryStandardRepository()
    repository.add_many([_example_entity()])

    results = repository.keyword_search("IS 00000 : 2000", limit=5)
    assert [item.code for item in results] == ["IS 00000 : 2000"]


def test_service_reports_dataset_status_honestly() -> None:
    """With no dataset configured the service must say exactly that."""
    service = StandardsService(
        repository=InMemoryStandardRepository(),
        loader=StandardsDatasetLoader(
            settings=SimpleNamespace(standards_dataset_path=None)
        ),
    )
    status = service.dataset_status()

    assert status.available is False
    assert status.loaded is False
    assert status.count == 0
    assert status.dataset_version is None
    assert "STANDARDS_DATASET_PATH" in status.message
    assert status.issues


def test_service_loads_the_configured_dataset(shipped_dataset_path: Path) -> None:
    """The verified dataset populates the repository and stays searchable."""
    repository = InMemoryStandardRepository()
    service = StandardsService(
        repository=repository,
        loader=StandardsDatasetLoader(path=shipped_dataset_path),
    )

    status = service.ensure_loaded()
    assert status.available is True
    assert status.loaded is True
    assert status.count == 20
    assert repository.count() == 20
    assert status.dataset_version
    assert status.organization == "Bureau of Indian Standards"

    # Loading twice does not duplicate records.
    assert service.ensure_loaded().count == 20

    record = service.get_standard("is 12615:2018")
    assert record.code == "IS 12615:2018"
    assert record.title is None
    assert record.source.organization == "Bureau of Indian Standards"


def test_repository_search_finds_standards_by_designation(
    shipped_dataset_path: Path,
) -> None:
    service = StandardsService(
        repository=InMemoryStandardRepository(),
        loader=StandardsDatasetLoader(path=shipped_dataset_path),
    )
    service.ensure_loaded()

    iec_matches = service.keyword_search("IEC 60034", limit=10)
    assert {record.code for record in iec_matches} == {
        "IS/IEC 60034-1:2022",
        "IS/IEC 60034-2-1:2024",
        "IS/IEC 60034-5:2020",
    }

    assert service.keyword_search("", limit=10) == []
    assert service.keyword_search("nothing-matches-this", limit=10) == []


def test_catalog_payload_reports_safe_metadata(shipped_dataset_path: Path) -> None:
    service = StandardsService(
        repository=InMemoryStandardRepository(),
        loader=StandardsDatasetLoader(path=shipped_dataset_path),
    )
    catalog = service.catalog(limit=3)

    assert catalog.available is True
    assert catalog.count == 20
    assert len(catalog.standards) == 3
    assert catalog.dataset.available is True
    assert catalog.dataset.dataset_version
    assert catalog.dataset.message

