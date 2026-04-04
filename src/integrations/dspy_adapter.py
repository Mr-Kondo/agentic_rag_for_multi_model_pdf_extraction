"""
dspy_ollama_adapter.py
======================
DSPy configuration helper for Ollama-backed language models.

DSPy has native support for Ollama via the LiteLLM backend. This module
provides a thin wrapper that configures DSPy to use a specific Ollama
model and exposes a convenience factory function consistent with the
project's configuration patterns.

Usage:
    from src.integrations.dspy_adapter import configure_ollama_lm
    import dspy

    configure_ollama_lm("qwen2.5:7b")

    signature = dspy.Signature("question -> answer")
    predictor = dspy.ChainOfThought(signature)
    result = predictor(question="What is 2+2?")

Integration with BaseLoadableModel pattern:
    class ValidatorAgent(BaseLoadableModel):
        def _do_load(self):
            configure_ollama_lm(self.model_id)
            self._predictor = dspy.ChainOfThought(ValidatorSignature)
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

log = logging.getLogger(__name__)

# Default Ollama base URL (overridden by configure_ollama_lm via config)
_DEFAULT_BASE_URL = "http://localhost:11434"


def configure_ollama_lm(
    model_id: str,
    base_url: str = _DEFAULT_BASE_URL,
    max_tokens: int = 512,
    temperature: float = 0.0,
    **kwargs: Any,
) -> dspy.LM:
    """
    Configure DSPy to use an Ollama-hosted model in one call.

    DSPy's built-in LiteLLM backend supports Ollama via the
    "ollama_chat/<model>" prefix. This function constructs the
    appropriate LM object and registers it globally with dspy.configure().

    Args:
        model_id: Ollama model name, e.g. "qwen2.5:7b"
        base_url: Ollama server base URL (default: http://localhost:11434)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for deterministic output)
        **kwargs: Additional keyword arguments forwarded to dspy.LM

    Returns:
        The configured dspy.LM instance

    Example:
        lm = configure_ollama_lm("qwen2.5:7b")
        predictor = dspy.ChainOfThought("question -> answer")
        result = predictor(question="What is the capital of France?")
    """
    log.info("Configuring DSPy with Ollama model: %s (base_url=%s)", model_id, base_url)

    lm = dspy.LM(
        f"ollama_chat/{model_id}",
        api_base=base_url,
        api_key="ollama",
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    dspy.configure(lm=lm)
    log.info("DSPy configured with Ollama model: %s", model_id)
    return lm
