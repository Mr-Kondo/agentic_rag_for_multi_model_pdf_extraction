"""Unit tests for the DSPy Ollama adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.integrations.dspy_adapter import configure_ollama_lm


@patch("src.integrations.dspy_adapter.dspy.configure")
@patch("src.integrations.dspy_adapter.dspy.LM")
def test_configure_ollama_lm_sets_dspy_lm(lm_cls_mock, configure_mock):
    """configure_ollama_lm should build a dspy.LM with the ollama_chat prefix and call dspy.configure."""
    fake_lm = MagicMock()
    lm_cls_mock.return_value = fake_lm

    result = configure_ollama_lm("qwen2.5:7b", base_url="http://localhost:11434", max_tokens=256, temperature=0.0)

    lm_cls_mock.assert_called_once_with(
        "ollama_chat/qwen2.5:7b",
        api_base="http://localhost:11434",
        api_key="ollama",
        max_tokens=256,
        temperature=0.0,
    )
    configure_mock.assert_called_once_with(lm=fake_lm)
    assert result is fake_lm


@patch("src.integrations.dspy_adapter.dspy.configure")
@patch("src.integrations.dspy_adapter.dspy.LM")
def test_configure_ollama_lm_forwards_extra_kwargs(lm_cls_mock, configure_mock):
    """configure_ollama_lm should forward extra kwargs to dspy.LM."""
    fake_lm = MagicMock()
    lm_cls_mock.return_value = fake_lm

    result = configure_ollama_lm(
        "qwen3:8b",
        base_url="http://localhost:11434",
        max_tokens=512,
        temperature=0.0,
        top_p=0.9,
    )

    lm_cls_mock.assert_called_once_with(
        "ollama_chat/qwen3:8b",
        api_base="http://localhost:11434",
        api_key="ollama",
        max_tokens=512,
        temperature=0.0,
        top_p=0.9,
    )
    configure_mock.assert_called_once_with(lm=fake_lm)
    assert result is fake_lm
