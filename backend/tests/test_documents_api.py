"""Document upload / extraction endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_upload_text_document(client: TestClient, sample_text_bytes: bytes) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("requirement.txt", sample_text_bytes, "text/plain")},
    )
    assert response.status_code == 201

    payload = response.json()
    assert payload["metadata"]["document_id"]
    assert payload["metadata"]["filename"] == "requirement.txt"

    document = payload["document"]
    assert document["page_count"] == 1
    assert document["word_count"] > 10
    assert "cement" in document["text"].lower()


def test_upload_pdf_document(client: TestClient, sample_pdf_bytes: bytes) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("tender.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201

    document = response.json()["document"]
    assert document["page_count"] == 1
    assert "cement" in document["text"].lower()


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("drawing.dwg", b"binary", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_file_type"


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "document_processing_error"
