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
from typing import Any, Optional, Union

import ollama

from src.core.config import config

log = logging.getLogger(__name__)

# Configure HuggingFace cache directory for the embedder model.
# LLM inference is handled by Ollama; this path is only used by
# sentence-transformers backends (non-default configuration).
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
        self._request_timeout_seconds: float = config.get_ollama_request_timeout_seconds()
        self._client: ollama.Client = ollama.Client(
            host=self._base_url,
            timeout=self._request_timeout_seconds,
        )
        self._vllm_models: dict[str, Any] = {}  # Cache vLLM LLM instances (heavy)

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

    def load_vision_model(self, model_id: str) -> Union[ollama.Client, Any]:
        """
        Return Ollama client or vLLM instance for vision-language inference.

        For Ollama models: verifies model is available and returns shared client.
        For vLLM models (HuggingFace IDs): loads vLLM LLM instance (lazy, cached).

        Args:
            model_id: Model identifier. Ollama format (e.g. "qwen2.5vl:7b")
                     or HuggingFace repo ID (e.g. "ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227")

        Returns:
            ollama.Client for Ollama models, or vLLM LLM instance for HF models
        """
        # vLLM for HF hub models (contain '/' in ID)
        if "/" in model_id:
            return self._load_vllm_model(model_id)
        # Ollama for standard model names
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

    def _load_vllm_model(self, model_id: str) -> Any:
        """
        Load a vLLM LLM instance for HuggingFace model IDs.

        Lazy-loads and caches the model. Only loads on first access;
        subsequent calls return the cached instance.

        Args:
            model_id: HuggingFace model ID, e.g. "ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227"

        Returns:
            vLLM LLM instance (cached)

        Raises:
            ImportError: If vLLM is not installed
            ValueError: If model cannot be loaded from HuggingFace
        """
        if model_id in self._vllm_models:
            log.debug(f"Using cached vLLM model: {model_id}")
            return self._vllm_models[model_id]

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("vLLM not installed. Install with: pip install vllm==0.11.0") from e

        log.info(f"Loading vLLM model: {model_id} (this may take a few minutes)")
        try:
            llm = LLM(
                model=model_id,
                dtype="bfloat16",  # Ricoh published in BF16
                enforce_eager=True,  # For Mac compatibility
                gpu_memory_utilization=0.7,
            )
            self._vllm_models[model_id] = llm
            log.info(f"✓ vLLM model loaded: {model_id}")
            return llm
        except Exception as e:
            raise ValueError(
                f"Failed to load vLLM model '{model_id}': {e}. Ensure HuggingFace token is configured and model is accessible."
            ) from e


# Global model cache instance
_model_cache = ModelCache()
