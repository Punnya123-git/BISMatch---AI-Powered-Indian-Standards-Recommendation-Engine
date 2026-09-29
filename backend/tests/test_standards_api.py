"""Standards catalogue + health endpoint tests (API contract level)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_health_service, get_standards_service
from app.database.repositories.in_memory import InMemoryStandardRepository
from app.main import app
from app.services.health_service import HealthService
from app.services.standards_service import StandardsService
from app.standards.dataset_loader import StandardsDatasetLoader

_INTERNAL_PATH_FRAGMENTS = ("standards.json", ".json", "data/", "data\\\\")


@pytest.fixture
def override_standards_service():
    """Temporarily replace the service the /api/standards route depends on."""

    def _apply(service: StandardsService) -> None:
        app.dependency_overrides[get_standards_service] = lambda: service

    yield _apply
    app.dependency_overrides.pop(get_standards_service, None)


@pytest.fixture
def override_health_service():
    """Temporarily replace the service the /api/health route depends on."""

    def _apply(service: HealthService) -> None:
        app.dependency_overrides[get_health_service] = lambda: service

    yield _apply
    app.dependency_overrides.pop(get_health_service, None)


def _service_with_dataset(path: Path) -> StandardsService:
    """A service with its own repository and an explicit dataset file."""
    return StandardsService(
        repository=InMemoryStandardRepository(),
        loader=StandardsDatasetLoader(path=path),
    )


def test_standards_endpoint_returns_the_loaded_catalogue(client: TestClient) -> None:
    response = client.get("/api/standards")
    assert response.status_code == 200

    payload = response.json()
    assert payload["available"] is True
    assert payload["count"] == 20
    assert len(payload["standards"]) == 20

    dataset = payload["dataset"]
    assert dataset["available"] is True
    assert dataset["dataset_version"]
    assert dataset["organization"] == "Bureau of Indian Standards"
    assert dataset["product_category"] == "rotating electrical machines"
    assert dataset["issues"] == []

    standard = payload["standards"][0]
    assert standard["code"]
    assert standard["title"] is None  # never invented
    assert standard["related_standards"] == []
    assert standard["source"]["organization"] == "Bureau of Indian Standards"
    assert standard["version"]["year"] is not None


def test_standards_endpoint_exposes_no_internal_paths(client: TestClient) -> None:
    body = client.get("/api/standards").text
    for fragment in _INTERNAL_PATH_FRAGMENTS:
        assert fragment not in body


def test_standards_endpoint_supports_paging(client: TestClient) -> None:
    first = client.get("/api/standards", params={"limit": 5}).json()
    second = client.get("/api/standards", params={"limit": 5, "offset": 5}).json()

    assert len(first["standards"]) == 5
    assert len(second["standards"]) == 5
    assert first["count"] == second["count"] == 20
    assert {item["code"] for item in first["standards"]}.isdisjoint(
        {item["code"] for item in second["standards"]}
    )


def test_standards_endpoint_searches_by_text(client: TestClient) -> None:
    payload = client.get("/api/standards", params={"q": "IEC 60034"}).json()
    assert {item["code"] for item in payload["standards"]} == {
        "IS/IEC 60034-1:2022",
        "IS/IEC 60034-2-1:2024",
        "IS/IEC 60034-5:2020",
    }

    exact = client.get("/api/standards", params={"q": "IS 12615:2018"}).json()
    assert [item["code"] for item in exact["standards"]] == ["IS 12615:2018"]


def test_standards_endpoint_validates_the_query(client: TestClient) -> None:
    response = client.get("/api/standards", params={"limit": 0})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_standards_endpoint_reports_a_missing_dataset(
    client: TestClient, override_standards_service, tmp_path: Path
) -> None:
    override_standards_service(_service_with_dataset(tmp_path / "absent.json"))

    response = client.get("/api/standards")
    assert response.status_code == 503
    assert response.json()["code"] == "standards_dataset_unavailable"


def test_standards_endpoint_reports_an_invalid_dataset(
    client: TestClient, override_standards_service, tmp_path: Path
) -> None:
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps({"standards": [{"standard_number": "", "product_category": "x"}]}),
        encoding="utf-8",
    )
    override_standards_service(_service_with_dataset(path))

    response = client.get("/api/standards")
    assert response.status_code == 500

    payload = response.json()
    assert payload["code"] == "standards_dataset_invalid"
    assert payload["details"]


def test_health_reports_catalogue_and_index_counts(client: TestClient) -> None:
    payload = client.get("/api/health").json()

    assert payload["standards_available"] == 20
    assert payload["indexed_chunks"] == 0  # nothing is indexed without a provider

    dataset = payload["dataset"]
    assert dataset["available"] is True
    assert dataset["count"] == 20
    assert dataset["dataset_version"]
    assert dataset["organization"] == "Bureau of Indian Standards"

    dependencies = {item["name"]: item for item in payload["dependencies"]}
    assert dependencies["standards_dataset"]["configured"] is True


def test_health_stays_up_when_the_dataset_cannot_be_loaded(
    client: TestClient, override_health_service, tmp_path: Path
) -> None:
    service = _service_with_dataset(tmp_path / "absent.json")
    override_health_service(HealthService(standards_service=service))

    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["standards_available"] == 0
    assert payload["dataset"]["available"] is False
    assert payload["dataset"]["issues"]
    assert payload["dataset"]["message"]

