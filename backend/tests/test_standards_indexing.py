"""Standards indexing tests.

Embeddings and vector storage are exercised with *test doubles* defined here:
production code never fabricates vectors, so the "not configured" path is
asserted to write nothing at all, while the write path is verified with a
deterministic, clearly-labelled stub provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from app.core.exceptions import VectorStoreError
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.pipeline import RecommendationPipeline
from app.rag.vector_store.base import VectorQueryResult, VectorRecord, VectorStore
from app.standards.dataset_loader import StandardsDatasetLoader
from app.standards.indexing_service import (
    STATUS_DATASET_INVALID,
    STATUS_DATASET_NOT_AVAILABLE,
    STATUS_EMBEDDING_NOT_CONFIGURED,
    STATUS_INDEXED,
    STATUS_VECTOR_STORE_UNAVAILABLE,
    StandardsIndexingService,
)


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic test-only provider: no model, no network, no pretence."""

    provider_name = "stub-test-provider"

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def dimension(self) -> int | None:
        return 4

    @property
    def model(self) -> str | None:
        return "stub-test-vectors"

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text) % 7), float(sum(map(ord, text)) % 13), 1.0, 0.5]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class InMemoryVectorStore(VectorStore):
    """Minimal test double (the real store is ``ChromaVectorStore``)."""

    provider_name = "in-memory-test-store"

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def add(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            self._records[record.id] = record
        return len(records)

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorQueryResult]:
        stored = list(self._records.values())[:top_k]
        return [
            VectorQueryResult(id=item.id, text=item.text, score=1.0, metadata=item.metadata)
            for item in stored
        ]

    def count(self) -> int:
        return len(self._records)

    def delete(self, ids: Sequence[str]) -> int:
        return sum(1 for record_id in ids if self._records.pop(record_id, None))

    def reset(self) -> None:
        self._records.clear()


class BrokenVectorStore(InMemoryVectorStore):
    """Fails on write, so the failure must be reported instead of swallowed."""

    def add(self, records: Sequence[VectorRecord]) -> int:
        raise VectorStoreError("simulated vector store failure")


def _service(
    dataset_path: Path,
    *,
    provider: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
) -> StandardsIndexingService:
    pipeline = RecommendationPipeline(embedding_provider=provider, vector_store=store)
    return StandardsIndexingService(
        loader=StandardsDatasetLoader(path=dataset_path), pipeline=pipeline
    )


def _dataset_file(
    tmp_path: Path, payload: dict[str, Any], name: str = "dataset.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_dataset_reports_not_available(tmp_path: Path) -> None:
    service = _service(tmp_path / "absent.json", store=InMemoryVectorStore())
    report = service.index()

    assert report.status == STATUS_DATASET_NOT_AVAILABLE
    assert report.indexed is False
    assert "not found" in report.message


def test_invalid_dataset_reports_every_problem(
    tmp_path: Path, dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 1111:2001", year=1999)
    )
    service = _service(_dataset_file(tmp_path, payload), store=InMemoryVectorStore())
    report = service.index()

    assert report.status == STATUS_DATASET_INVALID
    assert any("year" in reason for reason in report.reasons)
    assert report.written == 0


def test_unconfigured_embedding_provider_writes_nothing(shipped_dataset_path: Path) -> None:
    """The honest-failure path: no provider configured means no vectors at all."""
    store = InMemoryVectorStore()
    service = _service(shipped_dataset_path, store=store)
    report = service.index()

    assert report.status == STATUS_EMBEDDING_NOT_CONFIGURED
    assert report.indexed is False
    assert report.written == 0
    assert report.indexed_chunks == 0
    assert store.count() == 0
    # The dataset is still valid and fully prepared - nothing was hidden.
    assert report.standards_count == 20
    assert report.document_count == 20
    assert report.chunk_count == 20
    assert report.dataset_version
    assert any("EMBEDDING_PROVIDER" in reason for reason in report.reasons)


def test_indexing_with_a_configured_provider_writes_chunks(
    shipped_dataset_path: Path,
) -> None:
    store = InMemoryVectorStore()
    service = _service(
        shipped_dataset_path, provider=StubEmbeddingProvider(), store=store
    )
    report = service.index()

    assert report.status == STATUS_INDEXED
    assert report.indexed is True
    assert report.standards_count == 20
    assert report.chunk_count == 20
    assert report.written == 20
    assert report.indexed_chunks == 20
    assert store.count() == 20

    stored = next(iter(store._records.values()))
    assert stored.metadata["record_type"] == "standard"
    assert stored.metadata["standard_number"]
    assert stored.metadata["standard_code"] == stored.metadata["standard_number"]
    assert stored.metadata["product_category"] == "rotating electrical machines"
    assert stored.metadata["source"] == "Bureau of Indian Standards"
    assert stored.metadata["dataset_version"] == report.dataset_version
    assert stored.text.startswith("Standard Number: ")
    assert " " not in stored.id


def test_indexing_is_idempotent_and_reset_clears_previous_vectors(
    shipped_dataset_path: Path,
) -> None:
    store = InMemoryVectorStore()
    service = _service(
        shipped_dataset_path, provider=StubEmbeddingProvider(), store=store
    )

    first = service.index()
    second = service.index()
    assert first.written == second.written == 20
    assert store.count() == 20

    reset_report = service.index(reset=True)
    assert reset_report.status == STATUS_INDEXED
    assert store.count() == 20


def test_vector_store_failure_is_reported(shipped_dataset_path: Path) -> None:
    service = _service(
        shipped_dataset_path, provider=StubEmbeddingProvider(), store=BrokenVectorStore()
    )
    report = service.index()

    assert report.status == STATUS_VECTOR_STORE_UNAVAILABLE
    assert report.indexed is False
    assert report.written == 0
    assert "simulated vector store failure" in report.message


def test_chroma_roundtrip_stores_searchable_metadata(
    shipped_dataset_path: Path, tmp_path: Path
) -> None:
    """The real Chroma adapter, exercised with the test-only stub provider."""
    pytest.importorskip("chromadb")
    from app.rag.vector_store.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(
        collection_name="standards_index_test",
        persist_directory=tmp_path / "chroma",
    )
    provider = StubEmbeddingProvider()
    service = _service(shipped_dataset_path, provider=provider, store=store)

    report = service.index(reset=True)
    assert report.status == STATUS_INDEXED
    assert store.count() == 20

    hits = store.query(provider.embed_query("IS 12615:2018"), top_k=3)
    assert hits
    assert hits[0].metadata["standard_number"]
    assert hits[0].metadata["dataset_version"] == report.dataset_version

