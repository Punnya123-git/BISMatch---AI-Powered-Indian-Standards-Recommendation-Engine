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

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare directories, load the standards catalogue and log readiness.

    Only the *dataset file* is read here. Embedding/indexing never happens at
    start-up: the vector index is built explicitly with
    ``python -m app.rag.index_standards`` so the API can never appear to have a
    searchable index that was never created.
    """
    settings: Settings = get_settings()
    settings.ensure_directories()
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
