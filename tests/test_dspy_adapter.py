"""Unit tests for the DSPy MLX adapter."""

from __future__ import annotations

from unittest.mock import patch

from src.integrations.dspy_adapter import MLXLM


class _DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt is True
        assert tokenize is False
        return "formatted-prompt"


@patch("src.integrations.dspy_adapter.load")
def test_mlxlm_initializes_dspy_kwargs(load_mock):
    """MLXLM should preserve DSPy LM kwargs such as temperature/max_tokens."""
    load_mock.return_value = (object(), _DummyTokenizer())

    lm = MLXLM("mlx-community/Qwen2.5-3B-Instruct-4bit", max_tokens=256, temperature=0.0)

    assert lm.kwargs["temperature"] == 0.0
    assert lm.kwargs["max_tokens"] == 256


@patch("src.integrations.dspy_adapter.generate")
@patch("src.integrations.dspy_adapter.load")
def test_mlxlm_translates_generation_kwargs_for_mlx(load_mock, generate_mock):
    """Adapter should translate DSPy kwargs to mlx-lm kwargs without leaking unsupported ones."""
    tokenizer = _DummyTokenizer()
    load_mock.return_value = (object(), tokenizer)
    generate_mock.return_value = "adapter-output"

    lm = MLXLM("mlx-community/Qwen2.5-3B-Instruct-4bit", max_tokens=128, temperature=0.0)

    result = lm(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
        temperature=0.7,
        n=2,
        stop=["DONE"],
    )

    assert result == ["adapter-output"]

    _, kwargs = generate_mock.call_args
    assert kwargs["prompt"] == "formatted-prompt"
    assert kwargs["max_tokens"] == 64
    assert kwargs["temp"] == 0.7
    assert kwargs["verbose"] is False
    assert "temperature" not in kwargs
    assert "n" not in kwargs
    assert "stop" not in kwargs

    assert lm.history[-1]["kwargs"]["temp"] == 0.7

