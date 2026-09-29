"""Embedding provider selection and failure modes.

The whole demo hinges on one property: embeddings come either from a real model
or from nowhere - never from a placeholder. These tests cover the registry, the
local (no API key) provider's configuration rules, and the remote provider's
request/response handling through a mock transport. No model weights are loaded
here; that is what ``tests/test_semantic_search_live.py`` does.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderNotConfiguredError, VectorStoreError
from app.rag.embeddings import factory
from app.rag.embeddings.factory import (
    UnconfiguredEmbeddingProvider,
    available_embedding_providers,
)
from app.rag.embeddings.providers.local_onnx import (
    DEFAULT_MODEL,
    MODEL_DIMENSION,
    LocalOnnxMiniLMEmbeddingProvider,
)
from app.rag.embeddings.providers.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)


def _settings(**overrides: object) -> Settings:
    """Settings built from keywords only - never from the developer's .env."""
    return Settings(_env_file=None, **overrides)


def _embeddings_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ]
        },
    )


# --- registry --------------------------------------------------------------
def test_both_providers_are_registered() -> None:
    names = available_embedding_providers()
    assert "onnx_minilm" in names  # local, no API key
    assert "openai_compatible" in names  # any OpenAI-style endpoint


def test_unknown_provider_name_reports_itself_without_embedding() -> None:
    provider = factory._build_provider(_settings(embedding_provider="mistral"))
    assert isinstance(provider, UnconfiguredEmbeddingProvider)
    assert "mistral" in (provider.reason or "")
    with pytest.raises(ProviderNotConfiguredError):
        provider.embed_query("anything")


# --- local ONNX provider (the one the project actually runs on) ------------
def test_local_provider_describes_the_real_model() -> None:
    provider = LocalOnnxMiniLMEmbeddingProvider(_settings(embedding_model=None))
    assert provider.provider_name == "onnx_minilm"
    assert provider.model == DEFAULT_MODEL
    assert provider.dimension == MODEL_DIMENSION
    # chromadb is installed in this environment, so the model is usable.
    assert provider.is_configured is True
    assert provider.reason is None
    assert "configured" in provider.describe()


def test_local_provider_refuses_a_model_it_cannot_run() -> None:
    """A typo must not quietly embed with a different model than requested."""
    provider = LocalOnnxMiniLMEmbeddingProvider(
        _settings(embedding_model="text-embedding-3-small")
    )
    assert provider.is_configured is False
    assert "not supported" in (provider.reason or "")
    assert DEFAULT_MODEL in (provider.reason or "")
    with pytest.raises(VectorStoreError):
        provider.embed_documents(["some text"])


def test_local_provider_returns_no_vectors_for_no_text() -> None:
    assert LocalOnnxMiniLMEmbeddingProvider(_settings()).embed_documents([]) == []


# --- OpenAI-compatible provider -------------------------------------------
def test_remote_provider_is_not_configured_without_key() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        _settings(embedding_model="text-embedding-3-small", embedding_api_key=None)
    )
    assert provider.is_configured is False
    assert "EMBEDDING_API_KEY" in (provider.reason or "")
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        provider.embed_query("query")
    # The message must say exactly what to set, and where.
    assert "backend/.env" in excinfo.value.message


def _remote(handler, **overrides: object) -> OpenAICompatibleEmbeddingProvider:
    config: dict[str, object] = {
        "embedding_model": "text-embedding-3-small",
        "embedding_api_key": "test-key",
        "embedding_base_url": "https://api.example.test/v1",
    }
    config.update(overrides)
    return OpenAICompatibleEmbeddingProvider(
        _settings(**config),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_remote_request_shape_and_response_ordering() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                # Deliberately out of order: the 'index' field decides the result.
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    provider = _remote(handler)
    assert provider.embed_documents(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]

    request = seen[0]
    assert str(request.url) == "https://api.example.test/v1/embeddings"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.read())
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["first", "second"]
    # Third-party endpoints are not sent 'input_type' they did not ask for.
    assert "input_type" not in body


def test_remote_sends_input_type_only_to_hosts_that_expect_it() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        return _embeddings_response([[0.5, 0.5]])

    _remote(handler, embedding_base_url="https://api.jina.ai/v1").embed_query("q")
    assert "input_type" in seen[0]
    _remote(handler, embedding_base_url=None).embed_query("q")
    assert "input_type" in seen[1]  # OpenAI itself is the other allowed host


def test_remote_dimension_mismatch_is_an_error() -> None:
    provider = _remote(
        lambda request: _embeddings_response([[0.1, 0.2, 0.3]]),
        embedding_dimension=384,
    )
    with pytest.raises(VectorStoreError, match="EMBEDDING_DIMENSION"):
        provider.embed_query("query")


def test_remote_count_mismatch_is_an_error() -> None:
    provider = _remote(lambda request: _embeddings_response([[0.1, 0.2]]))
    with pytest.raises(VectorStoreError, match="1 vector"):
        provider.embed_documents(["one", "two"])


def test_remote_retries_transient_failures_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.embeddings.providers.openai_compatible.time.sleep", lambda _s: None
    )
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="rate limited")
        return _embeddings_response([[0.7, 0.7]])

    assert _remote(handler).embed_query("query") == [0.7, 0.7]
    assert calls["n"] == 3


def test_remote_gives_up_after_retries_without_inventing_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.rag.embeddings.providers.openai_compatible.time.sleep", lambda _s: None
    )
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(VectorStoreError, match="Could not reach"):
        _remote(handler).embed_query("query")
    assert calls["n"] == 3


def test_remote_rejects_a_client_error_without_retrying() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")

    with pytest.raises(VectorStoreError, match="401"):
        _remote(handler).embed_query("query")
    assert calls["n"] == 1

    assert LocalOnnxMiniLMEmbeddingProvider(_settings()).embed_documents([]) == []
