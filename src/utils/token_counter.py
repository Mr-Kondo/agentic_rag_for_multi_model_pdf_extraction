"""
Unified token counting utilities for Langfuse usage tracking.

Provides accurate token counting using either:
  1. The model's native tokenizer (preferred, exact)
  2. tiktoken cl100k_base encoding (fallback, approximate)

Usage:
    from src.utils.token_counter import count_tokens, count_tokens_with_tokenizer

    # With a loaded MLX tokenizer (exact)
    n = count_tokens_with_tokenizer("こんにちは世界", tokenizer)

    # Standalone fallback (approximate via tiktoken)
    n = count_tokens("こんにちは世界")
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_tiktoken_encoding: Any | None = None
_tiktoken_load_attempted: bool = False


def _get_tiktoken_encoding() -> Any | None:
    """Lazily load and cache the tiktoken cl100k_base encoding."""
    global _tiktoken_encoding, _tiktoken_load_attempted
    if _tiktoken_load_attempted:
        return _tiktoken_encoding
    _tiktoken_load_attempted = True
    try:
        import tiktoken

        _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        log.warning("tiktoken unavailable; falling back to word-based estimation: %s", e)
        _tiktoken_encoding = None
    return _tiktoken_encoding


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken cl100k_base encoding.

    This is a model-agnostic approximation suitable for Langfuse usage tracking.
    For exact counts with a specific model, use ``count_tokens_with_tokenizer``.

    Args:
        text: Input text to tokenize.

    Returns:
        Token count (tiktoken-based, or character-length fallback).
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Last-resort fallback: rough char-based estimate for CJK text
    return max(1, len(text) // 2)


def count_tokens_with_tokenizer(text: str, tokenizer: Any) -> int:
    """
    Count tokens using a model's native tokenizer (exact).

    Falls back to ``count_tokens()`` (tiktoken) if the tokenizer
    is unavailable or raises an error.

    Args:
        text: Input text to tokenize.
        tokenizer: A transformers-compatible tokenizer with ``.encode()`` method.

    Returns:
        Exact token count from the model tokenizer, or tiktoken approximation.
    """
    if not text:
        return 0
    if tokenizer is None:
        return count_tokens(text)
    try:
        ids = tokenizer.encode(text)
        return len(ids)
    except Exception:
        return count_tokens(text)
