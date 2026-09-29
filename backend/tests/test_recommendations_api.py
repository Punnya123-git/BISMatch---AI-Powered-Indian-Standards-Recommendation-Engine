"""Recommendation endpoint tests.

These tests pin down the honesty contract: the API must always report what it
could and could not do (no dataset, no provider, no reasoning stage) and must
never return invented recommendations.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_analyze_without_dataset_returns_explicit_status(client: TestClient) -> None:
    response = client.post(
        "/api/recommendations/analyze",
        json={"requirement": "Supply of 43 grade ordinary Portland cement in 50 kg bags"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"dataset_unavailable", "not_configured", "placeholder"}
    assert payload["recommendations"] == []
    assert payload["message"]
    assert payload["pipeline"] is not None
    assert payload["pipeline"]["ready"] is False
    assert payload["pipeline"]["indexed_chunks"] == 0


def test_analyze_validates_short_requirement(client: TestClient) -> None:
    response = client.post("/api/recommendations/analyze", json={"requirement": "ab"})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_analyze_with_unknown_document_id_warns(client: TestClient) -> None:
    response = client.post(
        "/api/recommendations/analyze",
        json={
            "requirement": "Supply of mild steel plates for structural work",
            "document_id": "does-not-exist",
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["recommendations"] == []
    assert any("does-not-exist" in warning for warning in payload["warnings"])


def test_analyze_accepts_uploaded_document(
    client: TestClient, sample_text_bytes: bytes
) -> None:
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("requirement.txt", sample_text_bytes, "text/plain")},
    )
    document_id = upload.json()["metadata"]["document_id"]

    response = client.post(
        "/api/recommendations/analyze",
        json={"requirement": "Cement for concrete works", "document_id": document_id},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["document_id"] == document_id
    assert payload["recommendations"] == []
