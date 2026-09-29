"""OpenAI-compatible LLM provider implementation."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.ai.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    ProviderNotConfiguredError,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "openai_compatible"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: Retries for *transient* failures only (transport errors, 5xx).
_MAX_ATTEMPTS = 3
_TIMEOUT = 60.0

#: HTTP statuses worth one more attempt.
_RETRY_STATUSES = frozenset({500, 502, 503, 504})


class OpenAICompatibleLLMProvider(LLMProvider):
    """Executes completions via an OpenAI-compatible chat completions endpoint."""

    provider_name = PROVIDER_NAME

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    @property
    def model(self) -> str | None:
        model = (self._settings.llm_model or "").strip()
        return model or None

    @property
    def base_url(self) -> str:
        base = (self._settings.llm_base_url or "").strip() or DEFAULT_BASE_URL
        return base.rstrip("/")

    @property
    def api_key(self) -> str | None:
        key = (self._settings.llm_api_key or "").strip()
        return key or None

    @property
    def is_configured(self) -> bool:
        return bool(self.model and self.api_key)

    @property
    def reason(self) -> str | None:
        if self.is_configured:
            return None
        missing = [
            name
            for name, value in (
                ("LLM_MODEL", self.model),
                ("LLM_API_KEY", self.api_key),
            )
            if not value
        ]
        return (
            "LLM_PROVIDER is 'openai_compatible' but "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing."
        )

    def describe(self) -> str:
        state = "configured" if self.is_configured else f"not configured ({self.reason})"
        return f"{self.provider_name} ({state}, model={self.model or 'none'})"

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the chat completions endpoint with automatic retry on transient errors."""
        if not self.is_configured:
            raise ProviderNotConfiguredError(
                self.reason
                or "LLM_PROVIDER 'openai_compatible' requires LLM_MODEL and LLM_API_KEY."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        temperature = (
            request.temperature
            if request.temperature is not None
            else self._settings.llm_temperature
        )
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else self._settings.llm_max_tokens
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._send(url, payload, headers)
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning(
                    "LLM request to %s failed (attempt %d/%d): %s",
                    url,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
                continue

            if response.status_code == 200:
                try:
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise LLMProviderError(
                            "LLM provider returned empty choices in completion response."
                        )
                    first_choice = choices[0]
                    message = first_choice.get("message") or {}
                    text = message.get("content") or ""
                    finish_reason = first_choice.get("finish_reason")
                    usage = data.get("usage") or {}
                    return LLMResponse(
                        text=text,
                        provider=self.provider_name,
                        model=data.get("model") or self.model,
                        finish_reason=finish_reason,
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    )
                except (ValueError, KeyError) as exc:
                    raise LLMProviderError(
                        f"Malformed response payload from LLM provider: {exc}"
                    ) from exc

            # HTTP 429 is a rate/token budget rejection, not a transient fault.
            # Retrying inside the same budget window is guaranteed to fail again
            # and wastes latency, so this fails fast and lets the caller fall back.
            if response.status_code == 429:
                raise LLMRateLimitError(
                    "LLM provider rate limit reached (HTTP 429): "
                    f"{response.text[:300]}"
                )

            # Retry on transient 5xx status codes only.
            if response.status_code in _RETRY_STATUSES:
                logger.warning(
                    "LLM request returned status %d (attempt %d/%d): %s",
                    response.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                    response.text[:200],
                )
                last_error = LLMProviderError(
                    f"LLM provider returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                continue

            # Non-retryable error
            raise LLMProviderError(
                f"LLM provider returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        raise LLMProviderError(
            f"LLM request failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _send(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        if self._client is not None:
            return self._client.post(url, json=payload, headers=headers)
        with httpx.Client(timeout=_TIMEOUT) as client:
            return client.post(url, json=payload, headers=headers)
