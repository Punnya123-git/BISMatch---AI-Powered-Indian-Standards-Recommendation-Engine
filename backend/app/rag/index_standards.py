"""Index the standards dataset into ChromaDB.

Usage:

    cd backend
    python -m app.rag.index_standards [--dataset PATH] [--reset]

Steps performed (and printed):

    1. load the dataset          4. prepare searchable documents
    2. validate the dataset      5. check the embedding configuration
    3. show the number of        6. index into ChromaDB
       standards                 7. print the final chunk count

This is the *only* way the index is built: the API never indexes on start-up, so
a demo can never silently depend on vectors that were never created.

Exit codes: 0 indexed, 1 dataset missing or invalid, 2 embedding provider not
configured, 3 vector store unavailable.
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.standards.dataset_loader import StandardsDatasetLoader
from app.standards.indexing_service import (
    STATUS_DATASET_INVALID,
    STATUS_DATASET_NOT_AVAILABLE,
    STATUS_EMBEDDING_NOT_CONFIGURED,
    STATUS_INDEXED,
    STATUS_VECTOR_STORE_UNAVAILABLE,
    IndexingReport,
    StandardsIndexingService,
)

_EXIT_CODES = {
    STATUS_INDEXED: 0,
    STATUS_DATASET_NOT_AVAILABLE: 1,
    STATUS_DATASET_INVALID: 1,
    STATUS_EMBEDDING_NOT_CONFIGURED: 2,
    STATUS_VECTOR_STORE_UNAVAILABLE: 3,
}

_HEADLINE = {
    STATUS_INDEXED: "Indexed",
    STATUS_EMBEDDING_NOT_CONFIGURED: "BLOCKED: embedding provider not configured",
    STATUS_DATASET_NOT_AVAILABLE: "FAILED: standards dataset not available",
    STATUS_DATASET_INVALID: "FAILED: standards dataset is invalid",
    STATUS_VECTOR_STORE_UNAVAILABLE: "FAILED: vector store unavailable",
}


def _describe_dataset_path(loader: StandardsDatasetLoader) -> str:
    try:
        return str(loader.resolve_path())
    except AppError as exc:
        return f"unresolved ({exc.message})"


def print_report(loader: StandardsDatasetLoader, report: IndexingReport) -> None:
    """Print the run in the same order as the pipeline itself."""
    rows = [
        ("1. Dataset path", _describe_dataset_path(loader)),
        ("   Dataset version", report.dataset_version or "not declared"),
        ("2. Validation", "passed" if report.status != STATUS_DATASET_INVALID else "FAILED"),
        ("3. Standards loaded", str(report.standards_count)),
        ("4. Documents prepared", str(report.document_count)),
        ("   Chunks prepared", str(report.chunk_count)),
        ("5. Embedding provider", report.embedding_provider),
        ("   Vector store", report.vector_store),
        ("6. Result", _HEADLINE.get(report.status, report.status)),
        ("7. Indexed chunks in store", str(report.indexed_chunks)),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label.ljust(width)} : {value}")

    print()
    print(report.message)
    if report.reasons:
        print()
        for reason in report.reasons:
            print(f"  - {reason}")
    if not report.indexed:
        print()
        print("No vectors were written, so retrieval will stay unavailable.")


def main(argv: list[str] | None = None) -> int:
    """Run the indexing command; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.index_standards",
        description="Index the verified Indian Standards dataset into ChromaDB.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override STANDARDS_DATASET_PATH for this run.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vectors in the collection before indexing.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    loader = StandardsDatasetLoader(path=args.dataset) if args.dataset else StandardsDatasetLoader()
    service = StandardsIndexingService(loader=loader)
    report = service.index(reset=args.reset)

    print_report(loader, report)
    return _EXIT_CODES.get(report.status, 1)


if __name__ == "__main__":
    sys.exit(main())
