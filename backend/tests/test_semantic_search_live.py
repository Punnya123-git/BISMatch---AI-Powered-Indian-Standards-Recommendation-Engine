"""Opt-in check that semantic search really works on this machine.

Skipped by default: it loads the real MiniLM model (first run may need the
one-off weight download) and reads the real index in ``data/vector_store``, so
it is slower than the rest of the suite and depends on an indexing run:

    cd backend
    python -m app.rag.index_standards          # once
    $env:RUN_LIVE_RAG_TESTS = "1"; pytest tests/test_semantic_search_live.py

The components are built explicitly from the project paths instead of the
environment, because ``tests/conftest.py`` redirects the data directory (and
forces ``EMBEDDING_PROVIDER=unconfigured``) for the rest of the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag.embeddings.providers.local_onnx import LocalOnnxMiniLMEmbeddingProvider
from app.rag.pipeline import RecommendationPipeline
from app.rag.vector_store.chroma_store import ChromaVectorStore
from app.rag.verify_search import run_probes

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_RAG_TESTS") != "1",
        reason="set RUN_LIVE_RAG_TESTS=1 to run the real model against the real index",
    ),
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_DATASET = _DATA_DIR / "standards" / "standards_enriched_bis_verified.json"


@pytest.fixture(scope="module")
def live_pipeline() -> RecommendationPipeline:
    """Pipeline wired to the real index and the local model, not to test env."""
    settings = Settings(
        _env_file=None,
        data_dir=_DATA_DIR,
        vector_db_path=_DATA_DIR / "vector_store",
        embedding_provider="onnx_minilm",
        embedding_model=None,
        standards_dataset_path=_DATASET,
    )
    pipeline = RecommendationPipeline(
        embedding_provider=LocalOnnxMiniLMEmbeddingProvider(settings),
        vector_store=ChromaVectorStore(
            collection_name=settings.vector_collection_name,
            persist_directory=settings.resolved_vector_db_path,
        ),
        settings=settings,
    )
    readiness = pipeline.readiness()
    if not readiness.ready:
        pytest.skip(
            "the live index is not available: "
            + "; ".join(readiness.reasons)
            + " - run 'python -m app.rag.index_standards' first"
        )
    return pipeline


def test_plain_language_questions_retrieve_their_standard(live_pipeline) -> None:
    report = run_probes(top_k=5, pipeline=live_pipeline)
    failures = [result for result in report.results if not result.passed]
    assert not failures, "\n".join(
        f"{result.expected} <- '{result.query}' returned "
        f"{', '.join(result.top) or '(nothing)'}"
        for result in failures
    )


def test_the_top_hit_is_usually_already_the_answer(live_pipeline) -> None:
    """Semantic quality, not just presence somewhere in the top-k."""
    report = run_probes(top_k=5, pipeline=live_pipeline)
    assert report.first_rank_hits >= len(report.results) - 1, [
        (result.expected, result.rank, result.top) for result in report.results
    ]


def test_vectors_come_from_a_real_model(live_pipeline) -> None:
    """Similar in meaning >> similar in wording, which hashing could not do."""
    provider = live_pipeline.embedding_provider
    close = provider.embed_query("permissible noise level of a machine")
    far = provider.embed_query("packing of cement in moisture proof bags")
    text = (
        "IS 12065:2025 Permissible noise levels for rotating electrical machines "
        "(measurement of sound power level)"
    )
    (document,) = provider.embed_documents([text])

    def cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    assert cosine(close, document) > cosine(far, document)
