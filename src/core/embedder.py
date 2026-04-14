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
        retry_trim_enabled: bool = True,
        retry_trim_min_chars: int = 128,
    ) -> None:
        import ollama

        self._model_id = model_id
        self._client = ollama.Client(host=base_url)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._retry_trim_enabled = retry_trim_enabled
        self._retry_trim_min_chars = max(1, int(retry_trim_min_chars))
        self._retry_trim_ratios = (1.0, 0.75, 0.5, 0.35, 0.25)
        self._dim: int | None = None
        log.info("OllamaEmbedder initialized: %s (base_url=%s)", model_id, base_url)

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            # Probe dimension with a single call
            sample = self._client.embed(model=self._model_id, input=["probe"])
            if not sample.embeddings or not sample.embeddings[0]:
                raise ValueError(
                    f"Ollama model {self._model_id!r} returned empty embeddings. Is the model loaded? Run: ollama pull <model>"
                )
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
        """Encode texts in batches with fallback for context-length failures."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self._client.embed(model=self._model_id, input=batch, truncate=True)
                all_embeddings.extend(response.embeddings)
            except Exception as exc:
                if not self._is_context_length_error(exc):
                    raise

                log.warning(
                    "Embed batch failed; fallback to single: batch_start=%d batch_size=%d error=%s",
                    i,
                    len(batch),
                    exc,
                )
                for offset, text in enumerate(batch):
                    embedding = self._encode_single_with_retry(text, global_index=i + offset)
                    all_embeddings.append(embedding)
        return all_embeddings

    def _encode_single_with_retry(self, text: str, global_index: int) -> list[float]:
        """Encode one text, progressively trimming when context-length errors occur."""
        original_len = len(text)
        trim_lengths = [original_len]

        if self._retry_trim_enabled:
            for ratio in self._retry_trim_ratios[1:]:
                trim_lengths.append(max(self._retry_trim_min_chars, int(original_len * ratio)))

        # Preserve order and deduplicate lengths while clipping to original length.
        seen: set[int] = set()
        unique_lengths: list[int] = []
        for length in trim_lengths:
            clipped = min(original_len, length)
            if clipped not in seen:
                seen.add(clipped)
                unique_lengths.append(clipped)

        last_error: Exception | None = None
        for candidate_len in unique_lengths:
            candidate_text = text[:candidate_len]
            try:
                response = self._client.embed(model=self._model_id, input=[candidate_text], truncate=True)
                if not response.embeddings or not response.embeddings[0]:
                    raise ValueError("Ollama embed returned empty embedding in single retry mode")
                if candidate_len < original_len:
                    log.warning(
                        "Embed single retry succeeded after trim: index=%d chars=%d->%d",
                        global_index,
                        original_len,
                        candidate_len,
                    )
                return response.embeddings[0]
            except Exception as exc:
                last_error = exc
                if not self._is_context_length_error(exc):
                    raise
                log.warning(
                    "Embed single retry length error: index=%d chars=%d error=%s",
                    global_index,
                    candidate_len,
                    exc,
                )

        # If we exhausted retries, re-raise the last context-length exception.
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected embed retry state: no attempts were executed")

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        """Return True if an exception indicates embed context-length overflow."""
        return "input length exceeds the context length" in str(exc).lower()


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
    retry_trim_enabled: bool = True,
    retry_trim_min_chars: int = 128,
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
        retry_trim_enabled: Whether to retry single embeds with progressive trims
        retry_trim_min_chars: Minimum chars preserved during retry trim

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
            retry_trim_enabled=retry_trim_enabled,
            retry_trim_min_chars=retry_trim_min_chars,
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
