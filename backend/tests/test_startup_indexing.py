"""Start-up initialisation of the verified standards index.

Deployment used to index during the *build*; the index is now built by the
running process (see :mod:`app.standards.bootstrap`). These tests pin the
behaviour that makes that safe:

* the index is built from the real dataset with real vectors, never skipped or
  faked;
* a start-up that already matches the dataset is left alone (idempotent);
* a start-up that cannot embed never writes vectors, never raises, and reports
  the reason - the app must start and explain itself through /api/health.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.pipeline import RecommendationPipeline
from app.rag.vector_store.base import VectorQueryResult, VectorRecord, VectorStore
from app.standards.bootstrap import ensure_standards_index
from app.standards.dataset_loader import StandardsDatasetLoader
from app.standards.indexing_service import (
    STATUS_EMBEDDING_NOT_CONFIGURED,
    STATUS_INDEXED,
    StandardsIndexingService,
)


class CountingProvider(EmbeddingProvider):
    """Deterministic test-only provider that records how often it was used."""

    provider_name = "startup-test-provider"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def dimension(self) -> int | None:
        return 4

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text) % 7), float(sum(map(ord, text)) % 13), 1.0, 0.5]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class MemoryVectorStore(VectorStore):
    """Minimal test double; the real store is ``ChromaVectorStore``."""

    provider_name = "startup-test-store"

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}

    def add(self, records: Sequence[VectorRecord]) -> int:
        records = list(records)
        for record in records:
            self.records[record.id] = record
        return len(records)

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        where: dict[str, object] | None = None,
    ) -> list[VectorQueryResult]:
        return [
            VectorQueryResult(id=item.id, text=item.text, score=1.0, metadata={})
            for item in list(self.records.values())[:top_k]
        ]

    def count(self) -> int:
        return len(self.records)

    def delete(self, ids: Sequence[str]) -> int:
        return sum(1 for record_id in ids if self.records.pop(record_id, None))

    def reset(self) -> None:
        self.records.clear()


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


# --- building the index at start-up ---------------------------------------
def test_startup_indexes_the_verified_dataset(shipped_dataset_path: Path) -> None:
    """A cold start must end with the real catalogue searchable."""
    store = MemoryVectorStore()
    provider = CountingProvider()
    report = ensure_standards_index(
        service=_service(shipped_dataset_path, provider=provider, store=store)
    )

    assert report.status == STATUS_INDEXED
    assert report.indexed is True
    assert report.chunk_count == 20
    assert report.indexed_chunks == 20
    assert store.count() == 20
    # Real vectors were produced by the provider, not invented by the service.
    assert provider.calls == 1
    assert all(len(record.embedding) == 4 for record in store.records.values())


def test_startup_is_idempotent_and_does_not_reembed(shipped_dataset_path: Path) -> None:
    """A second start over an up-to-date index must not touch the store."""
    store = MemoryVectorStore()
    provider = CountingProvider()
    service = _service(shipped_dataset_path, provider=provider, store=store)

    first = ensure_standards_index(service=service)
    second = ensure_standards_index(service=service)

    assert first.indexed_chunks == second.indexed_chunks == 20
    assert second.status == STATUS_INDEXED
    assert store.count() == 20
    assert provider.calls == 1  # only the first start embedded anything
    assert "already up to date" in second.message


def test_half_written_index_is_repaired(shipped_dataset_path: Path) -> None:
    """A partial index must not be mistaken for a finished one."""
    store = MemoryVectorStore()
    provider = CountingProvider()
    service = _service(shipped_dataset_path, provider=provider, store=store)
    service.index()
    # Simulate a run that died half way through.
    for record_id in list(store.records)[:5]:
        store.records.pop(record_id)
    assert store.count() == 15
    assert service.is_index_current() is False

    report = ensure_standards_index(service=service)

    assert report.status == STATUS_INDEXED
    assert store.count() == 20
    assert provider.calls == 2



# --- honest failure paths ---------------------------------------------------
def test_startup_without_a_provider_writes_nothing(shipped_dataset_path: Path) -> None:
    """No usable provider => no vectors, no exception, an honest report."""
    store = MemoryVectorStore()
    report = ensure_standards_index(
        service=_service(shipped_dataset_path, store=store)
    )

    assert report.indexed is False
    assert report.status == STATUS_EMBEDDING_NOT_CONFIGURED
    assert report.indexed_chunks == 0
    assert store.count() == 0
    assert report.reasons


def test_startup_never_raises_when_the_dataset_is_missing(tmp_path: Path) -> None:
    """A broken catalogue must degrade to a report, not crash the process."""
    report = ensure_standards_index(
        service=_service(
            tmp_path / "does-not-exist.json", provider=CountingProvider()
        )
    )

    assert report.indexed is False
    assert report.status == "dataset_not_available"
    # The operator still learns *why* from the report, without a traceback.
    assert "does-not-exist.json" in report.message


def test_startup_never_raises_on_an_unexpected_error(shipped_dataset_path: Path) -> None:
    """Even a programming error must not take the web process down."""

    class Exploding(MemoryVectorStore):
        def count(self) -> int:
            raise RuntimeError("simulated catastrophic store failure")

    report = ensure_standards_index(
        service=_service(
            shipped_dataset_path, provider=CountingProvider(), store=Exploding()
        )
    )

    assert report.indexed is False
    assert "simulated catastrophic store failure" in report.message


# --- expectations about the shipped catalogue ------------------------------
def test_expected_chunk_count_matches_the_shipped_catalogue(
    shipped_dataset_path: Path,
) -> None:
    service = _service(shipped_dataset_path, provider=CountingProvider())
    assert service.expected_chunk_count() == 20


def test_empty_store_is_never_current(shipped_dataset_path: Path) -> None:
    service = _service(shipped_dataset_path, store=MemoryVectorStore())
    assert service.is_index_current() is False


# --- real store round trip --------------------------------------------------
def test_chroma_startup_produces_a_searchable_index(
    shipped_dataset_path: Path, tmp_path: Path
) -> None:
    """The real Chroma adapter, driven through the start-up path."""
    pytest.importorskip("chromadb")
    from app.rag.vector_store.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(
        collection_name="startup_index_test", persist_directory=tmp_path / "chroma"
    )
    service = _service(shipped_dataset_path, provider=CountingProvider(), store=store)

    report = ensure_standards_index(service=service)

    assert report.status == STATUS_INDEXED
    assert store.count() == 20
    hits = store.query([0.0, 0.0, 1.0, 0.5], top_k=3)
    assert hits
    assert all(hit.metadata.get("standard_number") for hit in hits)


def test_force_rebuilds_an_up_to_date_index(shipped_dataset_path: Path) -> None:
    store = MemoryVectorStore()
    provider = CountingProvider()
    service = _service(shipped_dataset_path, provider=provider, store=store)
    ensure_standards_index(service=service)

    ensure_standards_index(force=True, service=service)

    assert store.count() == 20
    assert provider.calls == 2

