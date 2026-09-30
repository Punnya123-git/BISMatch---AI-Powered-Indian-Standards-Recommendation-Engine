"""FastAPI application factory.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.error_handlers import register_error_handlers
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.services.standards_service import get_standards_service
from app.standards.bootstrap import ensure_standards_index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare directories, load the standards catalogue and log readiness.

    The verified catalogue is *read* here, and the vector index is built here
    too, by :func:`app.standards.bootstrap.ensure_standards_index`. Indexing
    used to be a build-time step only, which made every deployment depend on
    the build being able to download model weights and write the vector store;
    the running process now owns that step (see ``AUTO_INDEX_ON_STARTUP``).

    The honesty guarantee is unchanged: the index is only ever built from the
    configured dataset and the configured embedding provider, and startup
    indexing never raises, so a failure shows up in ``/api/health`` instead of
    a crash-looping container. Readiness still comes from the real stored chunk
    count, so the API can never claim a searchable index that was not created.
    """
    settings: Settings = get_settings()
    try:
        settings.ensure_directories()
    except OSError as exc:
        # A read-only or unwritable data directory must not stop the process:
        # /api/health reports the real state of every component.
        logger.warning("Could not prepare the data directories: %s", exc)
    logger.info(
        "%s v%s starting (environment=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    logger.info("Data directory: %s", settings.data_dir)

    try:
        status = get_standards_service().ensure_loaded()
        logger.info("Standards catalogue: %s", status.message)
    except AppError as exc:
        # The API must still start; /api/health reports the exact problem.
        logger.warning("Standards catalogue not loaded: %s", exc.message)

    if settings.auto_index_on_startup:
        ensure_standards_index()
    else:
        logger.info(
            "Automatic start-up indexing is disabled (AUTO_INDEX_ON_STARTUP=false); "
            "build the index with: python -m app.rag.index_standards"
        )

    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered recommendation engine for identifying applicable Indian "
            "Standards for procurement specifications (SIH PS-108)."
        ),
        docs_url=settings.docs_url,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Small landing payload so hitting the base URL is not a 404."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": settings.docs_url,
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
