"""OpenAI-compatible ``/embeddings`` provider (OpenAI, Gemini, Jina, vLLM, ...).

Anything that exposes ``POST {base_url}/embeddings`` with the OpenAI body shape
works, so this provider needs no SDK - only ``httpx``. Requests happen lazily
(never while the app starts up), and the provider reports ``not configured`` -
instead of guessing - until both a model and credentials are present.

Two honest defaults worth knowing:

* the vector length is verified locally: a batch whose vectors disagree (or that
  contradicts ``EMBEDDING_DIMENSION``) is an error, never something callers have
  to double-check;
* ``input_type`` is only sent for the hosts that require it, because sending an
  unrequested field to a strict endpoint is how 400s happen.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import ProviderNotConfiguredError, VectorStoreError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

#: Registry key selected through ``EMBEDDING_PROVIDER``.
PROVIDER_NAME = "openai_compatible"

#: Used when ``EMBEDDING_BASE_URL`` is empty.
DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: Providers that require an extra ``input_type`` field in the request body.
_INPUT_TYPE_HOSTS = ("jina.ai", "cohere", "openai.com")

#: Transient failures (429/5xx/timeouts) are retried this many times in total.
_MAX_ATTEMPTS = 3

#: Per-request network timeout, in seconds.
_TIMEOUT = 60.0


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embeds text through an OpenAI-compatible embeddings endpoint."""

    provider_name = PROVIDER_NAME

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        #: Tests inject a client bound to a mock transport.
        self._client = client

    # --- configuration -----------------------------------------------------
    @property
    def model(self) -> str | None:
        model = (self._settings.embedding_model or "").strip()
        return model or None

    @property
    def base_url(self) -> str:
        base = (self._settings.embedding_base_url or "").strip() or DEFAULT_BASE_URL
        return base.rstrip("/")

    @property
    def api_key(self) -> str | None:
        key = (self._settings.embedding_api_key or "").strip()
        return key or None

    @property
    def is_configured(self) -> bool:
        return bool(self.model and self.api_key)

    @property
    def reason(self) -> str | None:
        """Why the provider is unusable, for status endpoints and the CLI."""
        if self.is_configured:
            return None
        missing = [
            name
            for name, value in (
                ("EMBEDDING_MODEL", self.model),
                ("EMBEDDING_API_KEY", self.api_key),
            )
            if not value
        ]
        return (
            "EMBEDDING_PROVIDER is 'openai_compatible' but "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing."
        )

    @property
    def dimension(self) -> int | None:
        return self._settings.embedding_dimension

    @property
    def input_type_field(self) -> bool:
        """Whether the endpoint expects an explicit ``input_type`` field."""
        base = self.base_url.lower()
        return any(host in base for host in _INPUT_TYPE_HOSTS)

    def describe(self) -> str:
        state = "configured" if self.is_configured else f"not configured ({self.reason})"
        return f"{self.provider_name} ({state}, model={self.model or 'none'})"

    # --- embedding ---------------------------------------------------------
    def _send(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """POST once, using the injected client when one was provided (tests)."""
        if self._client is not None:
            return self._client.post(url, json=payload, headers=headers)
        with httpx.Client(timeout=_TIMEOUT) as client:
            return client.post(url, json=payload, headers=headers)

    def _request_payload(self, texts: Sequence[str], input_type: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.input_type_field:
            payload["input_type"] = input_type
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST the payload, retrying transient failures with a small backoff."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._send(url, payload, headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status < 500 and status != 429:
                    raise VectorStoreError(
                        "The embedding endpoint rejected the request with HTTP "
                        f"{status}: {exc.response.text[:300]}"
                    ) from exc
                last_error = exc
            except httpx.HTTPError as exc:  # timeouts, connection resets, ...
                last_error = exc
            if attempt < _MAX_ATTEMPTS:
                delay = 0.5 * attempt
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s - retrying in %.1fs",
                    attempt,
                    _MAX_ATTEMPTS,
                    last_error,
                    delay,
                )
                time.sleep(delay)
        raise VectorStoreError(
            f"Could not reach the embedding endpoint ({last_error}). No vectors were "
            "fabricated; fix EMBEDDING_BASE_URL / credentials and retry."
        )

    @staticmethod
    def _extract_embeddings(payload: dict[str, Any]) -> list[list[float]]:
        items = payload.get("data")
        if not isinstance(items, list) or not items:
            raise VectorStoreError(
                "The embedding endpoint returned no 'data' entries, so nothing could "
                "be embedded."
            )
        vectors: list[list[float]] = []
        for item in sorted(items, key=lambda entry: entry.get("index", 0)):
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise VectorStoreError(
                    "The embedding endpoint returned an entry without a usable "
                    "'embedding' vector."
                )
            vectors.append([float(value) for value in embedding])
        return vectors

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        if not self.is_configured:
            raise ProviderNotConfiguredError(
                f"{self.reason} Set EMBEDDING_MODEL and EMBEDDING_API_KEY (and "
                "optionally EMBEDDING_BASE_URL) in backend/.env."
            )
        vectors = self._extract_embeddings(
            self._post(self._request_payload(items, input_type))
        )
        if len(vectors) != len(items):
            raise VectorStoreError(
                f"The embedding endpoint returned {len(vectors)} vector(s) for "
                f"{len(items)} text(s); refusing to guess which chunk is which."
            )
        lengths = {len(vector) for vector in vectors}
        if len(lengths) > 1:
            raise VectorStoreError(
                "The embedding endpoint returned inconsistent vector lengths "
                f"({sorted(lengths)})."
            )
        expected = self.dimension
        if expected is not None and lengths != {expected}:
            raise VectorStoreError(
                f"EMBEDDING_DIMENSION is {expected} but the endpoint returned "
                f"{lengths.pop()}-dimensional vectors. Fix the configuration before "
                "indexing, because a mixed-dimension index is useless."
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]
