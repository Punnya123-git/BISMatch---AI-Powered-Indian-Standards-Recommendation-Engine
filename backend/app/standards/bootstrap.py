"""Build the verified standards index once, safely, at process start.

Why this exists
---------------
Indexing used to be a *build-time* step (``python -m app.rag.index_standards``
in the deploy pipeline). That couples the deployment to things a build cannot
guarantee: network access to download the ONNX weights, a writable vector
directory that survives into the runtime, and a catalogue path that resolves
the same way in both phases. When any of those wobbled the deploy failed even
though the application code was fine - and with an ephemeral filesystem the
index built during the build was not even present at runtime.

Initialising at start-up removes that coupling: the build only installs
dependencies, and the running process prepares its own index.

Honesty rules
-------------
This module changes *when* the index is built, never *what* is built:

* it reuses :class:`~app.standards.indexing_service.StandardsIndexingService`,
  so vectors always come from the configured embedding provider;
* it never fabricates vectors and never substitutes deterministic output. If
  the provider is unusable it reports the failure and writes nothing;
* it is idempotent - an index that already matches the dataset is left alone;
* it never raises. A process must start and serve ``/api/health`` reporting the
  true state, rather than crash-looping because the catalogue could not be
  embedded.
"""

from __future__ import annotations

from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.standards.indexing_service import (
    STATUS_DATASET_NOT_AVAILABLE,
    IndexingReport,
    StandardsIndexingService,
    get_standards_indexing_service,
)

logger = get_logger(__name__)


def _failure_report(message: str, reason: str) -> IndexingReport:
    """A report describing a start-up indexing failure, never an exception."""
    return IndexingReport(
        status=STATUS_DATASET_NOT_AVAILABLE,
        message=message,
        reasons=[reason],
    )


def ensure_standards_index(
    *,
    force: bool = False,
    service: StandardsIndexingService | None = None,
) -> IndexingReport:
    """Make sure the verified catalogue is searchable, and report the outcome.

    Returns the indexing report describing what happened. This function does
    not raise: any unexpected error is turned into a failed report so the
    application can still start and explain itself through ``/api/health``.
    """
    indexing = service or get_standards_indexing_service()
    try:
        report = indexing.index_if_needed(force=force)
    except AppError as exc:
        logger.warning("Standards index could not be prepared: %s", exc.message)
        return _failure_report(
            "The standards index is not available: " + exc.message,
            exc.message,
        )
    except Exception as exc:  # noqa: BLE001 - start-up must never crash the app
        logger.exception("Unexpected failure while preparing the standards index.")
        return _failure_report(
            "The standards index could not be prepared because of an unexpected "
            f"error: {exc}",
            str(exc),
        )

    if report.indexed:
        logger.info("Standards index: %s", report.message)
    else:
        # Honest degradation: the service starts and /api/health reports why.
        logger.warning("Standards index not built. %s", report.message)
        for reason in report.reasons:
            logger.warning("  - %s", reason)
    return report
