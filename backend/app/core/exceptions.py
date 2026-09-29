"""Domain level exceptions shared by services and the API layer.

The API layer only needs to translate these into HTTP responses (see
:mod:`app.core.error_handlers`), which keeps business logic framework free.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every expected, recoverable application error."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ConfigurationError(AppError):
    """Raised when required configuration (keys, models, paths) is missing."""

    status_code = 500
    code = "configuration_error"


class ProviderNotConfiguredError(ConfigurationError):
    """Raised when an AI/embedding provider is requested but not configured."""

    status_code = 503
    code = "provider_not_configured"


class LLMProviderError(AppError):
    """Raised when an LLM provider call fails (transport, HTTP, malformed body).

    ``status_code`` is a *class* attribute on purpose: ``AppError.__init__``
    accepts only ``message`` and ``details``, so passing ``status_code=`` to the
    constructor is a ``TypeError`` and used to mask the real provider failure.
    Subclass this and let the class attribute carry the status.
    """

    status_code = 502
    code = "llm_provider_error"


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM provider rejects a call with HTTP 429.

    A distinct type because it is the one provider failure that is *expected*
    under a shared/free key: the caller should fall back to the deterministic
    ranking immediately rather than burn the request budget on retries.
    """

    status_code = 503
    code = "llm_rate_limited"


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded file cannot be processed."""

    status_code = 400
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    """Raised when an upload exceeds the configured size limit."""

    status_code = 413
    code = "file_too_large"


class DocumentProcessingError(AppError):
    """Raised when a document cannot be parsed or cleaned."""

    status_code = 422
    code = "document_processing_error"


class VectorStoreError(AppError):
    """Raised for vector store failures or missing optional dependencies."""

    status_code = 503
    code = "vector_store_error"


class DatasetNotAvailableError(AppError):
    """Raised when the Indian Standards dataset has not been loaded yet."""

    status_code = 503
    code = "standards_dataset_unavailable"


class DatasetValidationError(AppError):
    """Raised when a standards dataset file exists but is not valid.

    ``details`` carries the list of concrete problems found, so a broken dataset
    fails loudly and the reason can be printed by the CLI or logged at startup.
    """

    status_code = 500
    code = "standards_dataset_invalid"


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    status_code = 404
    code = "not_found"
