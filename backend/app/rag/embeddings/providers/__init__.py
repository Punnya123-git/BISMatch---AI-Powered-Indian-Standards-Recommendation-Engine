"""Embedding provider implementations.

Two providers ship with the backend:

* :mod:`~app.rag.embeddings.providers.local_onnx` - runs the MiniLM ONNX model on
  this machine (default: works with no key and no network after one weight
  download);
* :mod:`~app.rag.embeddings.providers.openai_compatible` - calls any OpenAI-style
  ``/embeddings`` endpoint (OpenAI, Gemini compatibility mode, Jina, vLLM, ...).

Add another provider by subclassing ``EmbeddingProvider`` and registering it in
:mod:`app.rag.embeddings.factory`.
"""

from app.rag.embeddings.providers.local_onnx import (
    LocalOnnxMiniLMEmbeddingProvider,
)
from app.rag.embeddings.providers.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.rag.embeddings.providers.unconfigured import UnconfiguredEmbeddingProvider

__all__ = [
    "LocalOnnxMiniLMEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "UnconfiguredEmbeddingProvider",
]
