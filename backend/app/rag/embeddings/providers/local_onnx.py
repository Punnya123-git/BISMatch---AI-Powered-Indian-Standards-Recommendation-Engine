"""Local, offline sentence-embedding provider (no API key, no network).

The model is ``all-MiniLM-L6-v2`` (384 dimensional, cosine normalised), executed
with ONNX Runtime. It is the same sentence-transformer model ChromaDB ships an
ONNX build of, so this provider reuses the ``chromadb`` installation instead of
pulling in a second (multi-gigabyte) dependency such as PyTorch.

Honesty rules that matter for the demo:

* the vectors are produced by a real neural network, not by hashing, and this
  module cannot silently degrade to something else: if ONNX Runtime or the model
  weights are missing, the provider reports ``not configured`` with the reason;
* weights are downloaded once by Chroma into the user cache directory. After that
  first download the provider works fully offline.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.core.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

#: Registry key selected through ``EMBEDDING_PROVIDER``.
PROVIDER_NAME = "onnx_minilm"

#: The only model this provider can run (it has exactly one ONNX build).
DEFAULT_MODEL = "all-MiniLM-L6-v2"

#: Output length of that model.
MODEL_DIMENSION = 384


class LocalOnnxMiniLMEmbeddingProvider(EmbeddingProvider):
    """Embeds text locally with the MiniLM ONNX model bundled by ``chromadb``."""

    provider_name = PROVIDER_NAME

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._function: Any | None = None
        self._unusable_reason: str | None = None

    # --- configuration -----------------------------------------------------
    @property
    def model(self) -> str | None:
        model = (self._settings.embedding_model or "").strip()
        return model or DEFAULT_MODEL

    def _check(self) -> str | None:
        """Return why the model cannot be used here, or ``None`` when usable."""
        if self._unusable_reason is not None:
            return self._unusable_reason
        model = (self._settings.embedding_model or "").strip()
        if model and model.lower() != DEFAULT_MODEL:
            self._unusable_reason = (
                f"EMBEDDING_MODEL '{model}' is not supported by this provider; it "
                f"runs '{DEFAULT_MODEL}' only. Choose a provider that serves the "
                "model you want (for example EMBEDDING_PROVIDER=openai_compatible)."
            )
            return self._unusable_reason
        try:
            import chromadb.utils.embedding_functions  # noqa: F401 - availability
        except Exception as exc:  # noqa: BLE001 - broken/incomplete install
            self._unusable_reason = (
                "The local MiniLM model needs the optional 'chromadb' package. "
                "Install it with: pip install -r requirements-rag.txt"
                f" (import failed: {exc})"
            )
            return self._unusable_reason
        return None

    @property
    def is_configured(self) -> bool:
        return self._check() is None

    @property
    def reason(self) -> str | None:
        """Why the provider is unusable, for status endpoints and the CLI."""
        return self._check()

    @property
    def dimension(self) -> int | None:
        return MODEL_DIMENSION

    # --- embedding ---------------------------------------------------------
    def _embedder(self) -> Any:
        """Return (and cache) the model, loading the weights on first use."""
        reason = self._check()
        if reason is not None:
            raise VectorStoreError(reason)
        if self._function is None:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            try:
                self._function = ONNXMiniLM_L6_V2()
            except Exception as exc:  # noqa: BLE001 - download/load failures
                raise VectorStoreError(
                    "Could not load the local MiniLM embedding model "
                    f"({exc}). The weights are downloaded once into "
                    "'~/.cache/chroma/onnx_models'; no fake embeddings are "
                    "generated in the meantime."
                ) from exc
            logger.info(
                "Loaded local embedding model '%s' (%d dimensions).",
                DEFAULT_MODEL,
                MODEL_DIMENSION,
            )
        return self._function

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = [text for text in texts]
        if not items:
            return []
        vectors = self._embedder()(items)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
