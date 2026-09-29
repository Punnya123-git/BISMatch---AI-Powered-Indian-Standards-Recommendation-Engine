"""Retrieval-Augmented Generation building blocks.

Responsibilities are deliberately split so each can evolve independently:

* :mod:`app.rag.chunking`      - split text into overlapping chunks
* :mod:`app.rag.embeddings`    - embedding provider abstraction + factory
* :mod:`app.rag.vector_store`  - vector store abstraction (ChromaDB first)
* :mod:`app.rag.retrieval`     - query -> relevant chunks
* :mod:`app.rag.ranking`       - retrieved chunks -> ranked standard candidates
* :mod:`app.rag.pipeline`      - orchestration of indexing and retrieval
"""

from app.rag.chunking import TextChunk, chunk_pages, chunk_text
from app.rag.pipeline import PipelineReadiness, RecommendationPipeline
from app.rag.ranking import (
    RankedStandard,
    RankingOutcome,
    explain_match,
    rank_standards,
)
from app.rag.retrieval import RetrievedChunk, Retriever

__all__ = [
    "PipelineReadiness",
    "RankedStandard",
    "RankingOutcome",
    "RecommendationPipeline",
    "RetrievedChunk",
    "Retriever",
    "TextChunk",
    "chunk_pages",
    "chunk_text",
    "explain_match",
    "rank_standards",
]
