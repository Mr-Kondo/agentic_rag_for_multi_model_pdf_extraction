"""Tests for src.utils.token_counter module."""

import pytest

from src.utils.token_counter import (
    _get_tiktoken_encoding,
    count_tokens,
    count_tokens_with_tokenizer,
)


class TestCountTokens:
    """Tests for the tiktoken-based count_tokens function."""

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_english_text(self):
        result = count_tokens("Hello, world!")
        assert result > 0
        assert isinstance(result, int)

    def test_japanese_text(self):
        result = count_tokens("こんにちは世界")
        assert result > 0
        # Japanese text should produce multiple tokens (not just 1 from split())
        assert result >= 2

    def test_mixed_language_text(self):
        result = count_tokens("日本語テキスト and English text mixed together")
        assert result > 0

    def test_long_text(self):
        text = "token " * 100
        result = count_tokens(text)
        assert result > 50

    def test_tiktoken_encoding_loads(self):
        enc = _get_tiktoken_encoding()
        assert enc is not None


class TestCountTokensWithTokenizer:
    """Tests for count_tokens_with_tokenizer using mock tokenizer."""

    def test_empty_string_returns_zero(self):
        assert count_tokens_with_tokenizer("", None) == 0

    def test_none_tokenizer_falls_back(self):
        result = count_tokens_with_tokenizer("Hello, world!", None)
        # Should fall back to tiktoken
        assert result > 0

    def test_with_mock_tokenizer(self):
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

        tokenizer = MockTokenizer()
        result = count_tokens_with_tokenizer("one two three", tokenizer)
        assert result == 3

    def test_tokenizer_error_falls_back(self):
        class BrokenTokenizer:
            def encode(self, text):
                raise RuntimeError("tokenizer error")

        tokenizer = BrokenTokenizer()
        result = count_tokens_with_tokenizer("Hello, world!", tokenizer)
        # Should fall back to tiktoken without raising
        assert result > 0

    def test_japanese_with_mock_tokenizer(self):
        class CharTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        tokenizer = CharTokenizer()
        result = count_tokens_with_tokenizer("日本語テスト", tokenizer)
        assert result == 6


class TestTiktokenVsWordCount:
    """Demonstrate tiktoken is more accurate than split() for CJK text."""

    def test_japanese_token_count_differs_from_split(self):
        text = "自然言語処理は人工知能の一分野です"
        tiktoken_count = count_tokens(text)
        word_count = len(text.split())
        # split() returns 1 for spaceless Japanese; tiktoken should return more
        assert word_count == 1
        assert tiktoken_count > word_count
