"""
Model client management for Ollama-backed inference.

Provides a lightweight ModelCache that returns configured Ollama client
instances. Model loading and memory management are delegated to the Ollama
server process.

The HF_HOME environment variable is still set to ensure the sentence-transformers
embedding model is cached in the project-local ./models directory.
"""

import logging
import os
from pathlib import Path
from typing import Any

import ollama

from src.core.config import config

log = logging.getLogger(__name__)

# Configure HuggingFace cache directory for the embedder model.
# LLM inference is handled by Ollama; this path is only used by
# sentence-transformers (intfloat/multilingual-e5-small).
MODEL_CACHE_DIR = Path.home() / ".models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR.resolve())


class ModelCache:
    """
    Manages Ollama client instances for text and vision inference.

    All LLM inference is delegated to a running Ollama server. This class
    acts as a lightweight factory that provides a shared client and validates
    that requested models are available on the server.

    Attributes:
        _base_url: Ollama server base URL from configuration
        _client: Shared ollama.Client instance
    """

    def __init__(self) -> None:
        self._base_url: str = config.get_ollama_base_url()
        self._client: ollama.Client = ollama.Client(host=self._base_url)

    def load_text_model(self, model_id: str) -> ollama.Client:
        """
        Return the shared Ollama client for text inference.

        Verifies the model is available on the server. Does not load weights
        into this process; the Ollama server manages memory.

        Args:
            model_id: Ollama model name, e.g. "qwen3:8b"

        Returns:
            Configured ollama.Client instance
        """
        self._ensure_model_available(model_id)
        return self._client

    def load_vision_model(self, model_id: str) -> ollama.Client:
        """
        Return the shared Ollama client for vision-language inference.

        Verifies the model is available on the server. Multimodal models
        accept image bytes through the same chat API as text models.

        Args:
            model_id: Ollama model name, e.g. "qwen2.5vl:7b"

        Returns:
            Configured ollama.Client instance
        """
        self._ensure_model_available(model_id)
        return self._client

    def _ensure_model_available(self, model_id: str) -> None:
        """
        Check that a model is available in Ollama, logging a warning if not.

        Args:
            model_id: Ollama model name to check

        Raises:
            RuntimeError: If the Ollama server is not reachable
        """
        try:
            models_response = self._client.list()
            available = {m.model for m in models_response.models}
            # Normalize: Ollama appends ":latest" when no tag is given
            base_id = model_id.split(":")[0]
            if model_id not in available and f"{base_id}:latest" not in available:
                log.warning(
                    "Model '%s' not found in Ollama. Run: ollama pull %s",
                    model_id,
                    model_id,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama server at {self._base_url}. Ensure Ollama is running: ollama serve"
            ) from exc


# Global model cache instance
_model_cache = ModelCache()
