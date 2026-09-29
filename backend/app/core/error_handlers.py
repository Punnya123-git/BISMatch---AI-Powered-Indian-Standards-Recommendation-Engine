"""Translate application exceptions into consistent HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)


def _payload(code: str, message: str, details: object | None = None) -> dict:
    return ErrorResponse(code=code, message=message, details=details).model_dump()


def register_error_handlers(app: FastAPI) -> None:
    """Attach handlers so clients always receive an :class:`ErrorResponse`."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_payload(
                "validation_error",
                "The request payload failed validation.",
                exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_payload("internal_error", "An unexpected error occurred."),
        )
