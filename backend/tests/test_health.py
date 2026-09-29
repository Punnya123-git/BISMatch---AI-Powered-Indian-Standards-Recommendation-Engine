"""Health endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]
    assert payload["environment"] == "test"


def test_health_reports_dependency_status(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    dependencies = {item["name"]: item for item in payload["dependencies"]}

    assert {"llm_provider", "embedding_provider", "vector_store"} <= set(dependencies)
    # Nothing is configured yet, and the API must say so rather than fail.
    assert dependencies["llm_provider"]["configured"] is False
    assert dependencies["embedding_provider"]["configured"] is False


def test_root_endpoint_exists(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"
