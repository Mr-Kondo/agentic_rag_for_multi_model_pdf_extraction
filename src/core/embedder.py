"""
Embedder backends for chunk and query vectorization.

Two backends are supported:
  - SentenceTransformerEmbedder: uses HuggingFace sentence-transformers directly (default)
  - OllamaEmbedder: calls Ollama's embed API — useful for models like
    kun432/cl-nagoya-ruri-large which are served via Ollama

Both implement the same BaseEmbedder interface so ChunkStore is backend-agnostic.

Selecting a backend in settings.json:
  {
    "models": { "embedder": "kun432/cl-nagoya-ruri-large" },
    "embedder": {
      "backend": "ollama",
      "query_prefix": "クエリ: ",
      "passage_prefix": "文章: "
    }
  }
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# BASE PROTOCOL
# ═══════════════════════════════════════════════════════════


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol for embedding backends used by ChunkStore."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        """Encode document passages (stored in vector DB)."""
        ...

    def encode_query(self, query: str) -> list[list[float]]:
        """Encode a single query string."""
        ...


# ═══════════════════════════════════════════════════════════
# SENTENCE TRANSFORMERS BACKEND (default)
# ═══════════════════════════════════════════════════════════


class SentenceTransformerEmbedder:
    """
    Embedding backend using HuggingFace sentence-transformers.

    Loads the model locally. Compatible with any model published on HuggingFace
    that is supported by sentence-transformers, including multilingual-e5-* models.

    Args:
        model_id: HuggingFace model ID, e.g. "intfloat/multilingual-e5-small"
        query_prefix: Prefix to prepend to query strings (e.g. "query: ")
        passage_prefix: Prefix to prepend to passage strings (e.g. "passage: ")
        batch_size: Batch size for encoding
    """

    def __init__(
        self,
        model_id: str,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._dim: int = self._model.get_sentence_embedding_dimension()
        log.info("SentenceTransformerEmbedder loaded: %s (dim=%d)", model_id, self._dim)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self._passage_prefix}{t}" for t in texts]
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

    def encode_query(self, query: str) -> list[list[float]]:
        prefixed = f"{self._query_prefix}{query}"
        return self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


# ═══════════════════════════════════════════════════════════
# OLLAMA BACKEND
# ═══════════════════════════════════════════════════════════


class OllamaEmbedder:
    """
    Embedding backend using the Ollama embed API.

    Designed for models served via Ollama, such as:
      - kun432/cl-nagoya-ruri-large (Japanese-specialized, JMTEB Retrieval: 73.02)

    ruri-large requires Japanese-style prefixes:
      - query: "クエリ: "
      - passage: "文章: "

    Args:
        model_id: Ollama model name, e.g. "kun432/cl-nagoya-ruri-large"
        base_url: Ollama server URL
        query_prefix: Prefix for query encoding
        passage_prefix: Prefix for passage encoding
        batch_size: Number of texts per Ollama embed() call
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        query_prefix: str = "クエリ: ",
        passage_prefix: str = "文章: ",
        batch_size: int = 32,
    ) -> None:
        import ollama

        self._model_id = model_id
        self._client = ollama.Client(host=base_url)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._dim: int | None = None
        log.info("OllamaEmbedder initialized: %s (base_url=%s)", model_id, base_url)

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            # Probe dimension with a single call
            sample = self._client.embed(model=self._model_id, input=["probe"])
            self._dim = len(sample.embeddings[0])
            log.info("OllamaEmbedder dimension probed: %d", self._dim)
        return self._dim

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self._passage_prefix}{t}" for t in texts]
        return self._encode_batched(prefixed)

    def encode_query(self, query: str) -> list[list[float]]:
        prefixed = f"{self._query_prefix}{query}"
        return self._encode_batched([prefixed])

    def _encode_batched(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in batches, collecting all embeddings."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embed(model=self._model_id, input=batch)
            all_embeddings.extend(response.embeddings)
        return all_embeddings


# ═══════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════


def create_embedder(
    model_id: str,
    backend: str = "sentence_transformers",
    query_prefix: str | None = None,
    passage_prefix: str | None = None,
    batch_size: int = 32,
    ollama_base_url: str = "http://localhost:11434",
) -> SentenceTransformerEmbedder | OllamaEmbedder:
    """
    Factory function that creates the appropriate embedder backend.

    Args:
        model_id: Model identifier (HuggingFace path or Ollama model name)
        backend: "sentence_transformers" or "ollama"
        query_prefix: Override query prefix (uses backend default if None)
        passage_prefix: Override passage prefix (uses backend default if None)
        batch_size: Batch size for encoding
        ollama_base_url: Ollama server URL (only used when backend="ollama")

    Returns:
        Configured embedder instance

    Raises:
        ValueError: If backend is unknown
    """
    if backend == "ollama":
        return OllamaEmbedder(
            model_id=model_id,
            base_url=ollama_base_url,
            query_prefix=query_prefix if query_prefix is not None else "クエリ: ",
            passage_prefix=passage_prefix if passage_prefix is not None else "文章: ",
            batch_size=batch_size,
        )
    elif backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_id=model_id,
            query_prefix=query_prefix if query_prefix is not None else "query: ",
            passage_prefix=passage_prefix if passage_prefix is not None else "passage: ",
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown embedder backend: {backend!r}. Choose 'sentence_transformers' or 'ollama'.")
